// flashnet_algo.h
#ifndef FLASHNET_ALGO_H
#define FLASHNET_ALGO_H


// flashnet model structure
#define LEN_INPUT 12
#define LEN_LAYER_0 12
#define LEN_LAYER_M_1 128
#define LEN_LAYER_M_2 16
#define LEN_LAYER_0_HALF 6
#define LEN_LAYER_1 1


// # of devices
#define DEVICE_NUM 2


// # of history (we append previous 3 queue_len, latency and throughput)
#define N_HIST 3


// set to 1 to be verbose
#define VERBOSE 0


// Prototypes


/* 
    Given total number of IOs, say `total_io_num`,
    init the arrays of flashnet that store the historical data, including:
        1. historical queue length
        2. historical latency
        3. historical throughtput
 */
void set_flashnet(int total_io_num);


/*
    Using machine learning model to infer whether reject this read IO or not.
    Input: 
        io_type: read/write. 1 indicates read IO, 0 indicates write IO.
        size: IO size in byte.
        device: the index of the device that this IO is originally sent to.
        cur_queue_len: the queue length when IO is submitted.
    Output:
        0 or 1 to indicate reject or accept:
            0: accept the IO to the original device.
            1: reject this IO and redirect to the secondary device.
*/
int flashnet_inference(long io_type, long size, uint32_t device, long cur_queue_len);


/*
    Called by update_thread in io_replayer.c. It will append the following IO data to historical pool:
        1. historical queue length
        2. historical latency
        3. historical throughput
*/
void update_flashnet(long io_queue_len, long io_latency, long io_throughput);


/*
    Add the current IO queue len and return.
*/
long add_fetch_cur_queue_len();


/*
    Increase the current queue length by 1.
*/
void inc_queue_len();


/*
    Decrease the current queue length by 1.
*/
void dec_queue_len();


/*
    Free the memory of historical data.
*/
void free_flashnet();


#endif
