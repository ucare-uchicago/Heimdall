#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/fs.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "atomic.h"

// gcc io_replayer.c -o io_replayer -lpthread
// sudo ./io_replayer /dev/nvme0n1 /home/daniar/trace-ACER/TRACETOREPLAY/msft-most-iops-10-mins/out-rerated-10.0/azure.trace /home/daniar/trace-ACER/TRACEREPLAYED/nvme0n1/azure.trace
enum {READ_IO = 1, WRITE_IO = 0};


// format: ts_record(ms),latency(us),io_type(r=1/w=0),size(B),offset,ts_submit(ms),size(B)
FILE *out_file;  
int nr_workers = 64;
int64_t jobtracker = 0;
int block_size = 1;  // by default, one sector (512 bytes)
char device[200];
char tracefile[200];
char logfile[200];
int fd = -1;
int64_t DISKSZ = 0;

int64_t nr_tt_ios;
int64_t latecount = 0;
int64_t slackcount = 0;
uint64_t starttime;
void *buff;
int respecttime = 1;

int64_t *oft;
int *reqsize;
int *reqflag;
float *timestamp;

pthread_mutex_t lock;  // only for writing to logfile, TODO

static int64_t get_disksz(int devfd) {
    int64_t sz;

    ioctl(devfd, BLKGETSIZE64, &sz);
    printf("Disk size is %" PRId64 " MB\n", sz / 1024 / 1024);
    printf("    in Bytes %" PRId64 " B\n", sz );

    return sz;
}

int64_t read_trace(char ***req, char *tracefile) {
    char line[1024];
    int64_t nr_lines = 0, i = 0;
    int ch;

    // first, read the number of lines
    FILE *trace = fopen(tracefile, "r");
    if (trace == NULL) {
        printf("Cannot open trace file: %s!\n", tracefile);
        exit(1);
    }

    while (!feof(trace)) {
        ch = fgetc(trace);
        if (ch == '\n') {
            nr_lines++;
        }
    }
    printf("there are [%lu] IOs in total in trace:%s\n", nr_lines, tracefile);

    rewind(trace);

    // then, start parsing
    if ((*req = malloc(nr_lines * sizeof(char *))) == NULL) {
        fprintf(stderr, "memory allocation error (%d)!\n", __LINE__);
        exit(1);
    }

    while (fgets(line, sizeof(line), trace) != NULL) {
        line[strlen(line) - 1] = '\0';
        if (((*req)[i] = malloc((strlen(line) + 1) * sizeof(char))) == NULL) {
            fprintf(stderr, "memory allocation error (%d)!\n", __LINE__);
            exit(1);
        }

        strcpy((*req)[i], line);
        i++;
    }
    fclose(trace);

    return nr_lines;
}

void parse_io(char **reqs, int total_io) {
    char *one_io;
    int64_t i = 0;

    oft = malloc(total_io * sizeof(int64_t));
    reqsize = malloc(total_io * sizeof(int));
    reqflag = malloc(total_io * sizeof(int));
    timestamp = malloc(total_io * sizeof(float));

    if (oft == NULL || reqsize == NULL || reqflag == NULL ||
        timestamp == NULL) {
        printf("memory allocation error (%d)!\n", __LINE__);
        exit(1);
    }

    one_io = malloc(1024);
    if (one_io == NULL) {
        fprintf(stderr, "memory allocation error (%d)!\n", __LINE__);
        exit(1);
    }
    for (i = 0; i < total_io; i++) {
        memset(one_io, 0, 1024);
        strcpy(one_io, reqs[i]);

        // 1. request arrival time in "ms"
        timestamp[i] = atof(strtok(one_io, " "));
        // 2. device number (not needed)
        strtok(NULL, " ");
        // 3. block number (offset)
        oft[i] = atoll(strtok(NULL, " "));
        oft[i] *= block_size;
        oft[i] %= DISKSZ;
        // make sure offset is 4KB aligned
        oft[i] = oft[i] / 4096 * 4096;
        assert(oft[i] >= 0);
        // 4. request size in blocks
        reqsize[i] = atoi(strtok(NULL, " ")) * block_size;
        // make sure the request size is of multiples of 4KB
        // 5. request flags: 0 for write and 1 for read
        reqflag[i] = atoi(strtok(NULL, " "));

        // printf("%.2f,%ld,%d,%d\n", timestamp[i], oft[i], reqsize[i],reqflag[i]);
        // exit(1);
    }

    free(one_io);
}

int mkdirr(const char *path, const mode_t mode, const int fail_on_exist) {
    int result = 0;
    char *dir = NULL;
    do {
        if ((dir = strrchr(path, '/'))) {
            *dir = '\0';
            result = mkdirr(path, mode, fail_on_exist);
            *dir = '/';

            if (result) {
                break;
            }
        }

        if (strlen(path)) {
            if ((result = mkdir(path, mode))) {
                char s[PATH_MAX];
                // sprintf(s, "mkdir() failed for '%s'", path);
                // perror(s);
                result = 0;
            }
        }
    } while (0);
    return result;
}

void create_file(char *logfile) {
    if (-1 == mkdirr(logfile, 0755, 0)) {
        perror("mkdirr() failed()");
        exit(1);
    }
    // remove the file that just created by mkdirr to prevent error when doing
    // fopen
    remove(logfile);

    out_file = fopen(logfile, "w");
    if (!out_file) {
        printf("Error creating out_file(%s) file!\n", logfile);
        exit(1);
    }
}

void *perform_io() {
    int64_t cur_idx;
    int mylatecount = 0;
    int myslackcount = 0;
    struct timeval t1, t2;
    useconds_t sleep_time;
    int io_limit__, size__, ret;
    int64_t offset__;
    char *req_str[2] = {"write", "read"};

    int max_len = 1, cur_len;

    while (1) {
        cur_idx = atomic_fetch_inc(&jobtracker);
        if (cur_idx >= nr_tt_ios) {
            break;
        }

        myslackcount = 0;
        mylatecount = 0;

        // respect time part
        if (respecttime == 1) {
            gettimeofday(&t1, NULL);  // get current time
            int64_t elapsedtime = t1.tv_sec * 1e6 + t1.tv_usec - starttime;
            if (elapsedtime < (int64_t)(timestamp[cur_idx] * 1000)) {
                sleep_time =
                    (useconds_t)(timestamp[cur_idx] * 1000) - elapsedtime;
                if (sleep_time > 100000) {
                    myslackcount++;
                }
                usleep(sleep_time);
            } else {  // I am late
                // DAN: High slack rate is totally fine as long as the Late rate is 0%
                mylatecount++;
            }
        }

        // do the job
        // printf("IO %lu: size: %d; offset: %lu\n", cur_idx, size__, offset__);
        gettimeofday(&t1, NULL); //reset the start time to before start doing the job
        /* the submission timestamp */
        float submission_ts = (t1.tv_sec * 1e6 + t1.tv_usec - starttime) / 1000;
        int lat, i;
        lat = 0;
        i = 0;
        gettimeofday(&t1, NULL);

        if (reqflag[cur_idx] == WRITE_IO) {
            ret = pwrite(fd, buff, reqsize[cur_idx], oft[cur_idx]);
            if (ret < 0) {
                printf("Cannot write size %d to offset %lu! ret=%d\n",
                        reqsize[cur_idx], oft[cur_idx], ret);
            }
            // printf("write\n");
        } else if (reqflag[cur_idx] == READ_IO) {
            ret = pread(fd, buff, reqsize[cur_idx], oft[cur_idx]);
            if (ret < 0) {
                printf("%ld\n", cur_idx);
                printf("Cannot read size %d to offset %" PRId64
                       ", ret=%d,"
                       "errno=%d!\n",
                       (reqsize[cur_idx]), oft[cur_idx], ret, errno);
            }
        } else {
            printf("Bad request type(%d)!\n", reqflag[cur_idx]);
            exit(1);
        }
        gettimeofday(&t2, NULL);
        /* Coperd: I/O latency in us */
        lat = (t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_usec - t1.tv_usec);
        /*
            * Coperd: keep consistent with fio latency log format:
            * 1: timestamp in ms
            * 2: latency in us
            * 3: r/w type [0 for w, 1 for r] (this is opposite of fio)
            * 4: I/O size in bytes
            * 5: offset in bytes
            */
        pthread_mutex_lock(&lock);
        // fprintf(stderr, "CHECKPOINT REACHED @  %s:%i\n", __FILE__, __LINE__);
        fprintf(out_file, "%.3f,%d,%d,%d,%ld,%.3f,%d\n", timestamp[cur_idx],
                lat, reqflag[cur_idx], reqsize[cur_idx], oft[cur_idx],
                submission_ts, ret);
        // fprintf(stderr, "CHECKPOINT REACHED @  %s:%i\n", __FILE__, __LINE__);
        pthread_mutex_unlock(&lock);
        // exit(1);

        atomic_add(&latecount, mylatecount);
        atomic_add(&slackcount, myslackcount);
        i++;
    }
    return NULL;
}

void *pr_progress() {
    int64_t progress, np;
    int64_t cur_late_cnt, cur_slack_cnt;

    while (1) {
        progress = atomic_read(&jobtracker);
        cur_late_cnt = atomic_read(&latecount);
        cur_slack_cnt = atomic_read(&slackcount);

        np = (progress > nr_tt_ios) ? nr_tt_ios : progress;
        printf(
            "Progress: %.2f%% (%lu/%lu), Late rate: %.2f%% (%lu), "
            "Slack rate: %.2f%% (%lu)\r",
            100 * (float)np / nr_tt_ios, progress, nr_tt_ios,
            100 * (float)cur_late_cnt / nr_tt_ios, cur_late_cnt,
            100 * (float)cur_slack_cnt / nr_tt_ios, cur_slack_cnt);
        fflush(stdout);

        if (progress > nr_tt_ios) {
            break;
        }

        sleep(1);
    }
    printf("\n Finished replaying!\n");

    return NULL;
}

void do_replay(void) {
    pthread_t track_thread;  // progress
    struct timeval t1, t2;
    float totaltime;
    int t;

    printf("Start doing IO replay...\n");

    // thread creation
    pthread_t *tid = malloc(nr_workers * sizeof(pthread_t));
    if (tid == NULL) {
        printf("Error malloc thread,LOC(%d)!\n", __LINE__);
        exit(1);
    }

    assert(pthread_mutex_init(&lock, NULL) == 0);

    gettimeofday(&t1, NULL);
    starttime = t1.tv_sec * 1000000 + t1.tv_usec;
    for (t = 0; t < nr_workers; t++) {
        assert(pthread_create(&tid[t], NULL, perform_io, NULL) == 0);
    }
    assert(pthread_create(&track_thread, NULL, pr_progress, NULL) == 0);

    // wait for all threads to finish
    for (t = 0; t < nr_workers; t++) {
        pthread_join(tid[t], NULL);
    }
    pthread_join(track_thread, NULL);  // progress

    gettimeofday(&t2, NULL);

    // calculate something
    totaltime = (t2.tv_sec - t1.tv_sec) * 1e3 + (t2.tv_usec - t1.tv_usec) / 1e3;
    float runtime = (totaltime/1000);
    float late_rate = 100 * (float)atomic_read(&latecount) / nr_tt_ios;
    float slack_rate = 100 * (float)atomic_read(&slackcount) / nr_tt_ios;
    printf("==============================\n");
    printf("Total run time: %.3f s\n", (totaltime/1000));

    if (respecttime == 1) {
        printf("Late rate: %.2f%%\n", late_rate);
        printf("Slack rate: %.2f%%\n", slack_rate);
    }

    fclose(out_file);
    assert(pthread_mutex_destroy(&lock) == 0);

    // run statistics
    char command[500];
    snprintf(command, sizeof(command), "%s %s %.2f %.2f %.2f %s %s.stats",
             "python statistics.py ", logfile, runtime, late_rate , slack_rate, " > ", logfile);
    system(command);
    printf("Statistics output = %s.stats\n", logfile);
}

int main(int argc, char **argv) {
    char **request;

    if (argc != 4) {
        printf("Usage: ./io_replayer /dev/nvme0n1 tracefile logfile\n");
        exit(1);
    } else {
        sprintf(device, "%s", argv[1]);
        printf("Device ==> %s\n", device);
        sprintf(tracefile, "%s", argv[2]);
        printf("Raw trace ==> %s\n", tracefile);
        sprintf(logfile, "%s", argv[3]);
        printf("Output file ==> %s\n", logfile);
    }
    // start the disk part
    fd = open(device, O_DIRECT | O_RDWR);
    if (fd < 0) {
        printf("Cannot open %s\n", device);
        exit(1);
    }
    
    // read the traces
    int total_io = read_trace(&request, tracefile);
    printf("%s\n", request[0]);

    DISKSZ = get_disksz(fd);

    // parsing io fields
    parse_io(request, total_io);
    int i = 0;
    printf("%.2f,%ld,%d,%d\n", timestamp[i], oft[i], reqsize[i], reqflag[i]);

    // create output file
    create_file(logfile);

    // Read can be anywhere: We need the disk to be full before starting the IO

    int LARGEST_REQUEST_SIZE = (8 * 1024 * 1024);  // blocks
    int MEM_ALIGN = 4096 * 8;                      // bytes
    if (posix_memalign(&buff, MEM_ALIGN, LARGEST_REQUEST_SIZE * block_size)) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }

    // do the replay here
    nr_tt_ios = total_io;
    do_replay();

    free(buff);
    return 0;
}