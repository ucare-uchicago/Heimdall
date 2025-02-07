// linnos_algo.h
#ifndef LINNOS_ALGO_H
#define LINNOS_ALGO_H

// Linnos module Structure
#define LEN_INPUT 31
#define LEN_LAYER_0 256
#define LEN_LAYER_M_1 256
#define LEN_LAYER_M_2 256
#define LEN_LAYER_0_HALF 128
#define LEN_LAYER_1 2

// # of devices
#define DEVICE_NUM 2

// Prototypes
long getDigit(long number, int max_len, int index);
void set_linnos();
int linnos_inference(long size, uint32_t device);
void update_linnos(long current_io_latency, long current_io_size);



#endif