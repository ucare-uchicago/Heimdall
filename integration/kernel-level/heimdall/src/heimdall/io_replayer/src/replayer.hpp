/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#ifndef __REPLAYER_H__
#define __REPLAYER_H__

#include <stdint.h>
#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <fstream>
#include <string>
#include <fcntl.h>

#define LARGEST_REQUEST_BYTES (64*1024*1024)
#define MEM_ALIGN 4096
#define SINGLE_IO_LIMIT 1024*1024

extern int64_t *DISKSZ;

inline uint64_t get_ns_ts() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

//aaargh, I should have used this when reading
struct TraceOp {
    double timestamp;
    uint64_t offset;
    uint64_t size;
    uint8_t op;
    uint8_t device;
};

class Trace {
private:
    uint64_t io_rejections=0;
    uint64_t unique_io_rejections=0;
    uint64_t never_completed_ios=0;

    uint32_t ndevices;
    uint32_t *nr_workers;
    std::atomic<uint64_t> *jobtracker;
    uint64_t *trace_line_count;
    uint64_t start_timestamp;

    //the traces themselves
    double **req_timestamps;
    uint64_t **req_offsets;
    uint64_t **req_sizes;
    uint8_t **req_ops;
    int *dev_fds;

    //failover stuff
    std::atomic<uint64_t> late_ios[3];
    std::atomic<uint64_t> io_count[3];
    std::atomic<uint64_t> fails[3];
    std::atomic<uint64_t> unique_fails[3];
    std::atomic<uint64_t> never_finished[3];

    /*log format:
    * 1: timestamp in ms
    * 2: latency in us
    * 3: r/w type [0 for w, 1 for r] (this is opposite of fio)
    * 4: I/O size in bytes
    * 5: offset in bytes
    * 6: IO submission time (not used)
    */
    std::ofstream outfile;  
    std::mutex io_mutex;

    void allocate_trace() {
        req_timestamps = new double*[ndevices];
        req_offsets = new uint64_t*[ndevices];
        req_sizes = new uint64_t*[ndevices];
        req_ops = new uint8_t*[ndevices];
    }

public:
    Trace(char* dev_string) {
        ndevices = 0;
        char *token;
        std::string dev_names[12];
        token = strtok(dev_string, "-");
        while (token) {
            dev_names[ndevices] = std::string(token);
            ndevices++;
            token = strtok(NULL, "-");
        }

        dev_fds = new int[ndevices];
        for (int i = 0 ; i < ndevices ; i++) {
            dev_fds[i] = open(dev_names[i].c_str(), O_DIRECT | O_RDWR | O_LARGEFILE);
            if (dev_fds[i] < 0) {
                printf("Cannot open %s\n", dev_names[i].c_str());
                exit(1);
            }
            printf("Opened device %s at idx %d\n", dev_names[i].c_str(), i);
        }

        nr_workers = new uint32_t[ndevices];
        jobtracker = new std::atomic<uint64_t>[ndevices];
        trace_line_count = new uint64_t[ndevices]{0};
        ndevices = ndevices;
        allocate_trace();
        for (int i = 0 ; i < 3 ; i++) {
            std::atomic_init(&late_ios[i], (uint64_t)0);
            std::atomic_init(&io_count[i], (uint64_t)0);
            std::atomic_init(&fails[i], (uint64_t)0);
            std::atomic_init(&unique_fails[i], (uint64_t)0);
            std::atomic_init(&never_finished[i], (uint64_t)0);
        }
    }

    ~Trace() {
        delete nr_workers;
        delete jobtracker;

        for (int i = 0 ; i < ndevices ; i++) {
            delete req_timestamps[i] ;
            delete req_offsets[i];
            delete req_sizes[i];
            delete req_ops[i];
        }

        delete trace_line_count;
        delete req_timestamps;
        delete req_offsets;
        delete req_sizes;
        delete req_ops;

        outfile.close();
    }

    uint8_t get_ndevices() {
        return ndevices;
    }

    uint8_t get_fd(uint8_t dev) {
        return dev_fds[dev];
    }

    int* get_fds() {
        return dev_fds;
    }

    void closefds() {
        for (int i = 0 ; i < ndevices ; i++)
            close(dev_fds[i]);
    }

    void set_output_file(std::string filename) {
        outfile = std::ofstream(filename);
    }

    void parse_file(uint8_t device, char* trace_path) {
        trace_line_count[device] = 0;
        std::string cstr = std::string(trace_path);
        std::ifstream in(cstr);
        std::string line;
        while(std::getline(in, line)) {
            trace_line_count[device]++;
        }
        in.clear();
        in.seekg(0);

        printf("Trace of device %d has %lu lines\n", device, trace_line_count[device]);

        req_timestamps[device] = new double[trace_line_count[device]];
        req_offsets[device] = new uint64_t[trace_line_count[device]];
        req_sizes[device] = new uint64_t[trace_line_count[device]];
        req_ops[device] = new uint8_t[trace_line_count[device]];
        
        double timestamp;
        int trash;
        uint64_t offset, size;
        uint32_t op_type; //0 is read, 1 write
        uint64_t max_size=0;
        for (int i = 0 ; i < trace_line_count[device] ; i++) {
            std::getline(in, line);
            //printf("parsing %s\n", line.c_str());
            sscanf(line.c_str(), "%lf %d %lu %lu %u", 
                &timestamp, &trash, &offset, &size, &op_type);

            //in >> timestamp >> trash >> offset >> size >> op_type;
            req_timestamps[device][i] = timestamp;
            req_offsets[device][i] = offset;
            req_sizes[device][i] = size;
            req_ops[device][i] = op_type;
            //printf("%f, %lu, %lu, %d\n", timestamp, offset, size, op_type);
            if(size > max_size) {
                max_size = size;
            }
        }

        printf("Max size %lu MB\n", (uint64_t)(max_size/1e6));
    }

    /* 1: timestamp in ms
    * 2: latency in us
    * 3: r/w type [0 for w, 1 for r] (this is opposite of fio)
    * 4: I/O size in bytes
    * 5: offset in bytes
    * 6: IO submission time (in ms)
    * 7: Device index
    */
   //ts comes in as us
    void write_output_line(uint64_t ts, uint32_t latency, uint8_t op,
            uint64_t size, uint64_t offset, uint64_t submission, 
            uint32_t device, double timestamps) {
        std::lock_guard<std::mutex> lk(io_mutex);
        char buf[1024]; 
        sprintf(buf, "%.3ld,%d,%d,%ld,%lu,%.3ld,%u,%f", ts, latency, !op, 
                size, offset, submission, device, timestamps);
        outfile << std::string(buf) << std::endl;
    }

    TraceOp get_line(uint8_t device) {
        uint64_t line_n = jobtracker[device].fetch_add(1, std::memory_order_seq_cst);

        TraceOp traceop; 
        traceop.timestamp = line_n >= trace_line_count[device] ? -1 : req_timestamps[device][line_n];
        if (traceop.timestamp == -1)
            return traceop;

        // the offset should be adjusted according to the device size.
        int64_t tmp_oft = req_offsets[device][line_n];
        tmp_oft %= DISKSZ[device];
        tmp_oft = tmp_oft / 4096 * 4096; // make sure offset is 4KB aligned
        traceop.offset = tmp_oft;


        // traceop.offset = req_offsets[device][line_n],
        traceop.size = req_sizes[device][line_n];
        traceop.op = req_ops[device][line_n];
        return traceop;
    }

    void add_late_op(uint32_t dev) {
        late_ios[dev].fetch_add(1, std::memory_order_seq_cst);
    }

    void print_stats() {
        uint64_t total_lines = 0;
        for (int i = 0 ; i < ndevices ; i++) {
            total_lines += trace_line_count[i];
        }

        for (int i = 0 ; i < ndevices ; i++) {
            uint64_t lio = std::atomic_load(&late_ios[i]);
            uint64_t total = std::atomic_load(&io_count[i]);
            //printf("Device %d had %lu IOs, %lu late (%f%%)\n", i, total, lio, (lio/(float)total)*100);
            uint64_t f = std::atomic_load(&fails[i]);
            uint64_t nf = std::atomic_load(&never_finished[i]);
        }

    }

    void add_fail(uint32_t dev){
        std::atomic_fetch_add(&fails[dev], (uint64_t)1);
    }

    void add_unique_fail(uint32_t dev) {
        std::atomic_fetch_add(&unique_fails[dev], (uint64_t)1);
    }

    void add_never_finished(uint32_t dev) {
        std::atomic_fetch_add(&never_finished[dev], (uint64_t)1);
    }

    void add_io_count(int dev) {
        std::atomic_fetch_add(&io_count[dev], (uint64_t)1);
    }

    uint64_t get_io_count(int dev) {
        return std::atomic_fetch_add(&io_count[dev], (uint64_t)0);
    }

};


#endif