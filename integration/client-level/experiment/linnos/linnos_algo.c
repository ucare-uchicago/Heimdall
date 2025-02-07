#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>
#include "linnos_algo.h"
#include "2ssds_weights_header/w_Trace_dev_0.h"
#include "2ssds_weights_header/w_Trace_dev_1.h"


pthread_mutex_t linnos_param_lock;

long len_pending = 3;  // we use 3 integers to represent the value of pending length
long max_pending = 999;
// smaller the index, newer the IO. e.g. prev_pending_io1 is ealier than prev_pending_io3
long pending_io;
long prev_pending_io1;
long prev_pending_io2;
long prev_pending_io3;
long prev_pending_io4;

long len_latency = 4;  // we use 4 integers to represent a latency value
long max_latency = 9999;
// smaller the index, newer the IO. e.g. prev_latency_1 is earlier than prev_latency_2
long prev_latency_1;
long prev_latency_2;
long prev_latency_3;
long prev_latency_4;



long *devices_weights[][8] = {
	{weight_0_T_dev_0, weight_1_T_dev_0, bias_0_dev_0, bias_1_dev_0 ,0,0,0,0},
	{weight_0_T_dev_1, weight_1_T_dev_1, bias_0_dev_1, bias_1_dev_1 ,0,0,0,0},
};


long getDigit(long number, int max_len, int index) {
	/*
		Given an number, max_len, and index of a digit, return the digit on specific index.
		for instance, if input (123, 3, 0), the expected output is 1. 
					  If input (123, 3, 2), the expected output is 3. 
					  If input (789, 4, 0), the expected output is 0. 
	*/
	if (index >= max_len) {
		printf("[Error] Index Out of range!\n");
		printf("max_len = %d, index = %d\n", max_len, index);
		exit(1);
	}
	if (number > max_latency && number > max_pending){
		printf("Input number error: %ld!\n", number);
		exit(1);
	}

    int divider = 1;
	for (int i = 0; i < max_len - 1 - index; i++) {
		divider *= 10;
	}

    // Extract the digit at the specified index
    long digit = (number / divider) % 10;
    
    return digit;
}



void set_linnos() {
	// 1. init the global params
	pending_io = 0;
	prev_pending_io1 = 0;
	prev_pending_io2 = 0;
	prev_pending_io3 = 0;
	prev_pending_io4 = 0;

	prev_latency_1 = 0;
	prev_latency_2 = 0;
	prev_latency_3 = 0;
	prev_latency_4 = 0;

	// 2. init the lock for updating params
	pthread_mutex_init(&linnos_param_lock, NULL);
}


int linnos_inference(long size, uint32_t device) {
    /*
        features_vec (31 feats) summary:
		[	
			prev_pending_io4,  (represented by 3 ints)
			prev_pending_io3,  (represented by 3 ints)
			prev_pending_io2,  (represented by 3 ints)
			prev_pending_io1,  (represented by 3 ints)
			pending_io, 	   (represented by 3 ints)
			prev_latency_4,    (represented by 4 ints)
			prev_latency_3,    (represented by 4 ints)
			prev_latency_2,    (represented by 4 ints)
			prev_latency_1,    (represented by 4 ints)
		]
            // Input parameters
                1. size
				2. device
    */
	
	pthread_mutex_lock(&linnos_param_lock);

	// retrieve the input to linnos
	long tmp_prev_pending_io4 = prev_pending_io4;
	long tmp_prev_pending_io3 = prev_pending_io3;
	long tmp_prev_pending_io2 = prev_pending_io2;
	long tmp_prev_pending_io1 = prev_pending_io1;
	long tmp_pending_io = pending_io;
	long tmp_prev_latency_4 = prev_latency_4;
	long tmp_prev_latency_3 = prev_latency_3;
	long tmp_prev_latency_2 = prev_latency_2;
	long tmp_prev_latency_1 = prev_latency_1;

	pthread_mutex_unlock(&linnos_param_lock);

	// 1. format input features
	long input_vec_i[LEN_INPUT];
	int input_idx = 0;
	// 1.1. format prev_pending_io4
	for (int i = 0; i < len_pending; i++) {
		input_vec_i[input_idx] = getDigit(tmp_prev_pending_io4, len_pending, i);
		input_idx += 1;
	}
	// 1.2. format prev_pending_io3
	for (int i = 0; i < len_pending; i++) {
		input_vec_i[input_idx] = getDigit(tmp_prev_pending_io3, len_pending, i);
		input_idx += 1;
	}
	// 1.3. format prev_pending_io2
	for (int i = 0; i < len_pending; i++) {
		input_vec_i[input_idx] = getDigit(tmp_prev_pending_io2, len_pending, i);
		input_idx += 1;
	}
	// 1.4. format prev_pending_io1
	for (int i = 0; i < len_pending; i++) {
		input_vec_i[input_idx] = getDigit(tmp_prev_pending_io1, len_pending, i);
		input_idx += 1;
	}
	// 1.5. format pending_io
	for (int i = 0; i < len_pending; i++) {
		input_vec_i[input_idx] = getDigit(tmp_pending_io, len_pending, i);
		input_idx += 1;
	}
	// 1.6. format prev_latency_4
	for (int i = 0; i < len_latency; i++) {
		input_vec_i[input_idx] = getDigit(tmp_prev_latency_4, len_latency, i);
		input_idx += 1;
	}
	// 1.7. format prev_latency_3
	for (int i = 0; i < len_latency; i++) {
		input_vec_i[input_idx] = getDigit(tmp_prev_latency_3, len_latency, i);
		input_idx += 1;
	}
	// 1.8. format prev_latency_2
	for (int i = 0; i < len_latency; i++) {
		input_vec_i[input_idx] = getDigit(tmp_prev_latency_2, len_latency, i);
		input_idx += 1;
	}
	// 1.9. format prev_latency_1
	for (int i = 0; i < len_latency; i++) {
		input_vec_i[input_idx] = getDigit(tmp_prev_latency_1, len_latency, i);
		input_idx += 1;
	}
	if (input_idx != LEN_INPUT) {   // check whether the input is correctly formated.
		printf("[Error!] format input error!");
		exit(1);
	}

    // 3. adopt the weight of corresponding device
	long *weights[4] = {devices_weights[device][0],devices_weights[device][1], devices_weights[device][2], devices_weights[device][3]};


	long mid_res_i[LEN_LAYER_0], final_res_i[LEN_LAYER_1];
	long *weight_0_T_ent, * bias_0_ent, *weight_1_T_ent, * bias_1_ent; 
	int i, j, k, offset;
	int end;

	weight_0_T_ent = weights[0];
	weight_1_T_ent = weights[1];
	bias_0_ent = weights[2];
	bias_1_ent = weights[3];

	for (j = 0, offset=0; j < LEN_LAYER_0; j++, offset+=LEN_INPUT) {
		mid_res_i[j] = 0;
		//loop unroll
		mid_res_i[j] += (input_vec_i[0] == 0 || weight_0_T_ent[offset+0] == 0)? 0 : input_vec_i[0] * weight_0_T_ent[offset+0];
		mid_res_i[j] += (input_vec_i[1] == 0 || weight_0_T_ent[offset+1] == 0)? 0 : input_vec_i[1] * weight_0_T_ent[offset+1];
		mid_res_i[j] += (input_vec_i[2] == 0 || weight_0_T_ent[offset+2] == 0)? 0 : input_vec_i[2] * weight_0_T_ent[offset+2];
		mid_res_i[j] += (input_vec_i[3] == 0 || weight_0_T_ent[offset+3] == 0)? 0 : input_vec_i[3] * weight_0_T_ent[offset+3];
		mid_res_i[j] += (input_vec_i[4] == 0 || weight_0_T_ent[offset+4] == 0)? 0 : input_vec_i[4] * weight_0_T_ent[offset+4];
		mid_res_i[j] += (input_vec_i[5] == 0 || weight_0_T_ent[offset+5] == 0)? 0 : input_vec_i[5] * weight_0_T_ent[offset+5];
		mid_res_i[j] += (input_vec_i[6] == 0 || weight_0_T_ent[offset+6] == 0)? 0 : input_vec_i[6] * weight_0_T_ent[offset+6];
		mid_res_i[j] += (input_vec_i[7] == 0 || weight_0_T_ent[offset+7] == 0)? 0 : input_vec_i[7] * weight_0_T_ent[offset+7];
		mid_res_i[j] += (input_vec_i[8] == 0 || weight_0_T_ent[offset+8] == 0)? 0 : input_vec_i[8] * weight_0_T_ent[offset+8];
		mid_res_i[j] += (input_vec_i[9] == 0 || weight_0_T_ent[offset+9] == 0)? 0 : input_vec_i[9] * weight_0_T_ent[offset+9];
		mid_res_i[j] += (input_vec_i[10] == 0 || weight_0_T_ent[offset+10] == 0)? 0 : input_vec_i[10] * weight_0_T_ent[offset+10];
		mid_res_i[j] += (input_vec_i[11] == 0 || weight_0_T_ent[offset+11] == 0)? 0 : input_vec_i[11] * weight_0_T_ent[offset+11];
		mid_res_i[j] += (input_vec_i[12] == 0 || weight_0_T_ent[offset+12] == 0)? 0 : input_vec_i[12] * weight_0_T_ent[offset+12];
		mid_res_i[j] += (input_vec_i[13] == 0 || weight_0_T_ent[offset+13] == 0)? 0 : input_vec_i[13] * weight_0_T_ent[offset+13];
		mid_res_i[j] += (input_vec_i[14] == 0 || weight_0_T_ent[offset+14] == 0)? 0 : input_vec_i[14] * weight_0_T_ent[offset+14];
		mid_res_i[j] += (input_vec_i[15] == 0 || weight_0_T_ent[offset+15] == 0)? 0 : input_vec_i[15] * weight_0_T_ent[offset+15];
		mid_res_i[j] += (input_vec_i[16] == 0 || weight_0_T_ent[offset+16] == 0)? 0 : input_vec_i[16] * weight_0_T_ent[offset+16];
		mid_res_i[j] += (input_vec_i[17] == 0 || weight_0_T_ent[offset+17] == 0)? 0 : input_vec_i[17] * weight_0_T_ent[offset+17];
		mid_res_i[j] += (input_vec_i[18] == 0 || weight_0_T_ent[offset+18] == 0)? 0 : input_vec_i[18] * weight_0_T_ent[offset+18];
		mid_res_i[j] += (input_vec_i[19] == 0 || weight_0_T_ent[offset+19] == 0)? 0 : input_vec_i[19] * weight_0_T_ent[offset+19];
		mid_res_i[j] += (input_vec_i[20] == 0 || weight_0_T_ent[offset+20] == 0)? 0 : input_vec_i[20] * weight_0_T_ent[offset+20];
		mid_res_i[j] += (input_vec_i[21] == 0 || weight_0_T_ent[offset+21] == 0)? 0 : input_vec_i[21] * weight_0_T_ent[offset+21];
		mid_res_i[j] += (input_vec_i[22] == 0 || weight_0_T_ent[offset+22] == 0)? 0 : input_vec_i[22] * weight_0_T_ent[offset+22];
		mid_res_i[j] += (input_vec_i[23] == 0 || weight_0_T_ent[offset+23] == 0)? 0 : input_vec_i[23] * weight_0_T_ent[offset+23];
		mid_res_i[j] += (input_vec_i[24] == 0 || weight_0_T_ent[offset+24] == 0)? 0 : input_vec_i[24] * weight_0_T_ent[offset+24];
		mid_res_i[j] += (input_vec_i[25] == 0 || weight_0_T_ent[offset+25] == 0)? 0 : input_vec_i[25] * weight_0_T_ent[offset+25];
		mid_res_i[j] += (input_vec_i[26] == 0 || weight_0_T_ent[offset+26] == 0)? 0 : input_vec_i[26] * weight_0_T_ent[offset+26];
		mid_res_i[j] += (input_vec_i[27] == 0 || weight_0_T_ent[offset+27] == 0)? 0 : input_vec_i[27] * weight_0_T_ent[offset+27];
		mid_res_i[j] += (input_vec_i[28] == 0 || weight_0_T_ent[offset+28] == 0)? 0 : input_vec_i[28] * weight_0_T_ent[offset+28];
		mid_res_i[j] += (input_vec_i[29] == 0 || weight_0_T_ent[offset+29] == 0)? 0 : input_vec_i[29] * weight_0_T_ent[offset+29];
		mid_res_i[j] += (input_vec_i[30] == 0 || weight_0_T_ent[offset+30] == 0)? 0 : input_vec_i[30] * weight_0_T_ent[offset+30];

		// apply bias
		mid_res_i[j] += bias_0_ent[j];
		// relu
		if (mid_res_i[j] < 0) {
			mid_res_i[j] = 0;
		}
	}
	final_res_i[0] = 0;
	for(k=0; k<LEN_LAYER_0; k += 8) {
		final_res_i[0] += (mid_res_i[k] == 0 || weight_1_T_ent[k] == 0)? 0 : mid_res_i[k] * weight_1_T_ent[k];
		final_res_i[0] += (mid_res_i[k+1] == 0 || weight_1_T_ent[k+1] == 0)? 0 : mid_res_i[k+1] * weight_1_T_ent[k+1];
		final_res_i[0] += (mid_res_i[k+2] == 0 || weight_1_T_ent[k+2] == 0)? 0 : mid_res_i[k+2] * weight_1_T_ent[k+2];
		final_res_i[0] += (mid_res_i[k+3] == 0 || weight_1_T_ent[k+3] == 0)? 0 : mid_res_i[k+3] * weight_1_T_ent[k+3];
		final_res_i[0] += (mid_res_i[k+4] == 0 || weight_1_T_ent[k+4] == 0)? 0 : mid_res_i[k+4] * weight_1_T_ent[k+4];
		final_res_i[0] += (mid_res_i[k+5] == 0 || weight_1_T_ent[k+5] == 0)? 0 : mid_res_i[k+5] * weight_1_T_ent[k+5];
		final_res_i[0] += (mid_res_i[k+6] == 0 || weight_1_T_ent[k+6] == 0)? 0 : mid_res_i[k+6] * weight_1_T_ent[k+6];
		final_res_i[0] += (mid_res_i[k+7] == 0 || weight_1_T_ent[k+7] == 0)? 0 : mid_res_i[k+7] * weight_1_T_ent[k+7];
	}
	// apply bias
	final_res_i[0] += bias_1_ent[0];

	final_res_i[1] = 0;
	for(k=0; k<LEN_LAYER_0; k += 8) {
		final_res_i[1] += (mid_res_i[k] == 0 || weight_1_T_ent[k+256] == 0)? 0 : mid_res_i[k] * weight_1_T_ent[k+256];
		final_res_i[1] += (mid_res_i[k+1] == 0 || weight_1_T_ent[k+257] == 0)? 0 : mid_res_i[k+1] * weight_1_T_ent[k+257];
		final_res_i[1] += (mid_res_i[k+2] == 0 || weight_1_T_ent[k+258] == 0)? 0 : mid_res_i[k+2] * weight_1_T_ent[k+258];
		final_res_i[1] += (mid_res_i[k+3] == 0 || weight_1_T_ent[k+259] == 0)? 0 : mid_res_i[k+3] * weight_1_T_ent[k+259];
		final_res_i[1] += (mid_res_i[k+4] == 0 || weight_1_T_ent[k+260] == 0)? 0 : mid_res_i[k+4] * weight_1_T_ent[k+260];
		final_res_i[1] += (mid_res_i[k+5] == 0 || weight_1_T_ent[k+261] == 0)? 0 : mid_res_i[k+5] * weight_1_T_ent[k+261];
		final_res_i[1] += (mid_res_i[k+6] == 0 || weight_1_T_ent[k+262] == 0)? 0 : mid_res_i[k+6] * weight_1_T_ent[k+262];
		final_res_i[1] += (mid_res_i[k+7] == 0 || weight_1_T_ent[k+263] == 0)? 0 : mid_res_i[k+7] * weight_1_T_ent[k+263];
	}
	// apply bias
	final_res_i[1] += bias_1_ent[1];
    end = (final_res_i[0]>=final_res_i[1])? 0: 1;

	if (end == 0) {   // accept this IO to original device
		pthread_mutex_lock(&linnos_param_lock);
		// 1. Update pending io
		pending_io = pending_io + size;
		if (pending_io > max_pending) {
			pending_io = max_pending;
		}
		pthread_mutex_unlock(&linnos_param_lock);
	}

	return end;
}


void update_linnos(long current_io_latency, long current_io_size){
    /*
        Update the following pamemters:
			prev_pending_io4,
			prev_pending_io3,
			prev_pending_io2,
			prev_pending_io1,
			pending_io,

            prev_latency_1,
            prev_latency_2,
            prev_latency_3,
			prev_latency_4,
    */

	pthread_mutex_lock(&linnos_param_lock);
	// 1. update queue length
	prev_pending_io4 = prev_pending_io3;
	prev_pending_io3 = prev_pending_io2;
	prev_pending_io2 = prev_pending_io1;
	prev_pending_io1 = pending_io;
	pending_io = pending_io - current_io_size;
	if (pending_io < 0) {  // a special situation that io size is greater than max_pending
		pending_io = 0;
	}

	// 2. update latency
	prev_latency_4 = prev_latency_3;
	prev_latency_3 = prev_latency_2;
	prev_latency_2 = prev_latency_1;
	prev_latency_1 = current_io_latency;
	if (prev_latency_1 > max_latency) {
		prev_latency_1 = max_latency;
	}

	pthread_mutex_unlock(&linnos_param_lock);
}