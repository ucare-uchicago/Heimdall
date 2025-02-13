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


#include <errno.h>
#include <iostream>
#include "op_replayers.hpp"


#define MAX_FAIL 1

int is_warmup = 0;

static int sleep_until(uint64_t next) {
    uint64_t now = get_ns_ts();
    int64_t diff = next - now;

    //if 0 or negative, we need to issue
    if(diff <= 0) {
        //we're late by at least 2 us
        if (diff <= -2000) return 1;
        return 0; //late but not that much
    }
    else 
        std::this_thread::sleep_for(std::chrono::nanoseconds(diff));
    return 0;
}

void baseline_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf) {
    int ret;
    int *fds = trace->get_fds();
    //read
    if(trace_op.op == 0) {
        trace->add_io_count(device);
        ret = pread(fds[device], buf, trace_op.size, trace_op.offset);
    } else if(trace_op.op == 1) {
        trace->add_io_count(device);
        ret = pwrite(fds[device], buf, trace_op.size, trace_op.offset);
    } else {
        printf("Wrong OP code! %d\n", trace_op.op);
    }

}


void heimdall_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf) {
    int ret, i;
    int *fds = trace->get_fds();
    int device_num = 2;
    bool success = false;
    //read
    if(trace_op.op == 0) {
        trace->add_io_count(device);
        ret = pread(fds[device], buf, trace_op.size, trace_op.offset);
        if (ret < 0) {
            trace->add_fail(device);
            // if reject, redirect to secondary device.
            device = ++device % device_num;

            trace->add_io_count(device);
            trace->add_unique_fail(device);
            ret = pread(fds[device], buf, trace_op.size, 0);    // redirect to the secondary device
        }
    } else if(trace_op.op == 1) {
        trace->add_io_count(device);
        if (is_warmup != 1) {  // formal replaying
            ret = pwrite(fds[device], buf, trace_op.size, trace_op.offset);
        }
        // ret = pwrite(fds[device], buf, trace_op.size, 0);
    } else {
        printf("Wrong OP code! %d\n", trace_op.op);
    }
}


void random_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf) {
    int ret;
    int *fds = trace->get_fds();

    auto clock = std::chrono::steady_clock::now();
    uint32_t time = std::chrono::duration_cast<std::chrono::microseconds>(clock.time_since_epoch()).count();

    std::srand(static_cast<unsigned int>(time));
    int random_device = std::rand() % 2;

    //read
    if(trace_op.op == 0) {
        trace->add_io_count(random_device);
        ret = pread(fds[random_device], buf, trace_op.size, trace_op.offset);
    } else if(trace_op.op == 1) {
        trace->add_io_count(device);
        ret = pwrite(fds[device], buf, trace_op.size, trace_op.offset);
    } else {
        printf("Wrong OP code! %d\n", trace_op.op);
    }
}


void* replayer_fn(void* arg) {
    Thread_arg *targ = (Thread_arg*) arg;
    Trace *trace = targ->trace;
    uint32_t device = targ->device;
    std::string type = targ->type;
    is_warmup = targ->is_warmup; // TODO: fix me in the future (warm up is not an ugly way and didn't identify the true bug)
    TraceOp trace_op;
    char *buf;
    int device_num = 2;

    // is_warmup
    // if (is_warmup == 1) {
    //     printf("Heimdall Warmup \n");    // TODO: fix me in the future
    // }

    // to update main thread status
    std::shared_ptr<double> main_status = std::make_shared<double>(0);;

    if (posix_memalign((void**)&buf, MEM_ALIGN, LARGEST_REQUEST_BYTES)) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }

    //start together
    pthread_barrier_wait(targ->sync_barrier);
    int is_late;
    while (1) {
        trace_op = trace->get_line(device);
        if (trace_op.timestamp == -1) {
            break;
        }
        //timestamp in op is in microsecond float, so convert to nano
        uint64_t next = targ->start_ts + (uint64_t)(trace_op.timestamp*1000);
        if(sleep_until(next) == 1)
            trace->add_late_op(device);

        uint64_t submission = get_ns_ts();
        auto begin = std::chrono::steady_clock::now();

        // realize trace_op
        targ->executor(trace_op, trace, targ->device, buf);
        auto end = std::chrono::steady_clock::now();
        uint32_t elaps =  std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        uint64_t end_ts = get_ns_ts();
        //store results
        trace->write_output_line(end_ts/1000000, elaps, trace_op.op,
                trace_op.size, trace_op.offset, submission/1000000,
                device, trace_op.timestamp);

    }
    free(buf);
    return 0;
}