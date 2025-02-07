#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <pthread.h>
#include <stdlib.h>
#include "2ssds_weights_header/w_Trace_dev_0.h"
#include "2ssds_weights_header/w_Trace_dev_1.h"
#include "atomic.h"
#include "flashnet_algo.h"


long queue_len;   // queue len is updated by multiple thread and should be protected by atomic operation.

/* The following four variables are not updated by multiple threads any more.  */
long hist_index;
long * prev_queue_len;     // An array<long> to store historical queue len
long * prev_latency;       // An array<long> to store historical latency 
long * prev_throughput;  // An array<long> to store historical throughput


long *devices_weights[][8] = {
	{weight_0_T_dev_0, weight_3_T_dev_0, bias_0_dev_0, bias_3_dev_0, weight_1_T_dev_0, bias_1_dev_0 ,weight_2_T_dev_0, bias_2_dev_0},
	{weight_0_T_dev_1, weight_3_T_dev_1, bias_0_dev_1, bias_3_dev_1, weight_1_T_dev_1, bias_1_dev_1 ,weight_2_T_dev_1, bias_2_dev_1},
};


long add_fetch_cur_queue_len() {
	/*
		Return the current IO queue len.
	*/
	return atomic_inc_fetch(&queue_len);   // the queue len will also count the current IO itself.
}


void inc_queue_len() {
	/*
		Increase the current queue length by 1.
	*/
	atomic_inc(&queue_len);
}


void dec_queue_len() {
	/*
		Decrease the current queue length by 1.
	*/
	atomic_dec(&queue_len);
}

void set_flashnet(int total_io_num) {
	// 1. init the global params
	queue_len = 0;
	hist_index = 0;
	
	// 2. init previous queue len
	prev_queue_len = malloc(total_io_num * sizeof(long));
	for (long i = 0; i < total_io_num; i++) {
		prev_queue_len[i] = -1;    // init to `-1` to indicate invalid
	}

	// 3. init previous latency
	prev_latency = malloc(total_io_num * sizeof(long));
	for (long i = 0; i < total_io_num; i++) {
		prev_latency[i] = -1;
	}

	// 4. init previous throughput
	prev_throughput = malloc(total_io_num * sizeof(long));
	for (long i = 0; i < total_io_num; i++) {
		prev_throughput[i] = -1;
	}
}


int flashnet_inference(long io_type, long size, uint32_t device, long cur_queue_len) {
    /*
        features_vec:
            // Input parameters
                0. io_type
                1. size
                2. device
				3. cur_queue_len
            // Retrieve the bellowing values from global historical data.
                4. prev_queue_len_1,
                5. prev_queue_len_2,
                6. prev_queue_len_3,
                7. prev_latency_1,
                8. prev_latency_2,
                9. prev_latency_3,
                10. prev_throughput_1,
                11. prev_throughput_2,
                12. prev_throughput_3
    */
	
	// 1. format input features, we expect the following features ordering:
	// 	{io_type, size, cur_queue_len, prev_queue_len_1, prev_queue_len_2, prev_queue_len_3, prev_latency_1, prev_latency_2, prev_latency_3, prev_throughput_1, prev_throughput_2, prev_throughput_3}
	long input_vec_i[LEN_INPUT];
	for (int i = 0; i < LEN_INPUT; i++) {   // init to 0
		input_vec_i[i] = 0;
	}
	// 1.1. format io_type, io_size, and cur_queue_len
	input_vec_i[0] = io_type;
	input_vec_i[1] = size;
	input_vec_i[2] = cur_queue_len;
	// 1.2. format previous queue len, latency, and throughput
	long cur_hist_index = hist_index;
	for (int i = 1; i <= N_HIST; i ++) {
		if (cur_hist_index - i >= 0) {   // check whether we have enough historical data.
			input_vec_i[2 + i] = prev_queue_len[cur_hist_index - i];
			input_vec_i[5 + i] = prev_latency[cur_hist_index - i];
			input_vec_i[8 + i] = prev_throughput[cur_hist_index - i];

			if (prev_queue_len[cur_hist_index - i] == -1 || prev_latency[cur_hist_index - i] == -1 || prev_throughput[cur_hist_index - i] == -1) {
				printf("[Error] The historical data is not valid!");
				exit(1);
			}
		}
	}


    // 2. adopt the weight of corresponding device
	long *weights[8] = {devices_weights[device][0],devices_weights[device][1], devices_weights[device][2], devices_weights[device][3], devices_weights[device][4], devices_weights[device][5], devices_weights[device][6], devices_weights[device][7]};

	long mid_res_i[LEN_LAYER_0], mid_res_m_1[LEN_LAYER_M_1], mid_res_m_2[LEN_LAYER_M_2], final_res_i[LEN_LAYER_1];
	long *weight_0_T_ent, * bias_0_ent, *weight_1_T_ent, * bias_1_ent, *weight_M_1, *bias_M_1, *weight_M_2, *bias_M_2; 
	int i, j, k, offset;
	int end;
	
	weight_0_T_ent = weights[0];
	weight_1_T_ent = weights[1];
	bias_0_ent = weights[2];
	bias_1_ent = weights[3];

	weight_M_1 = weights[4];
	bias_M_1 = weights[5];

	weight_M_2 = weights[6];
	bias_M_2 = weights[7];

	// Normalizer
	// Formula of MinMax Scaler from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
	// x_scaled = (x - min)/range
	for (j = 0, offset=0; j < LEN_LAYER_0; j++) {
		mid_res_i[j] = input_vec_i[j]-weight_0_T_ent[j]; // in weights was scale up to 1000 times
		mid_res_i[j] = mid_res_i[j] * bias_0_ent[j];   // scale of normalizer: multiply 1000000 to avoid round to 0. will divide 1000 later.
	}

	for (j = 0; j < LEN_LAYER_M_1; j++) {
		mid_res_m_1[j] = 0;
		for(int inputIndex = 0; inputIndex < LEN_LAYER_0; inputIndex++) 
			mid_res_m_1[j] += mid_res_i[inputIndex] * weight_M_1[j * LEN_LAYER_0 + inputIndex] >> 30;   // divide the scale of normalizer.

		// apply bias
		mid_res_m_1[j] += bias_M_1[j];
		// relu
		if (mid_res_m_1[j] < 0) {
			mid_res_m_1[j] = 0;
		}
	}
	
	for (j = 0; j < LEN_LAYER_M_2; j++) {
		mid_res_m_2[j] = 0;
		for(int inputIndex = 0; inputIndex < LEN_LAYER_M_1; inputIndex++) 
			mid_res_m_2[j] += mid_res_m_1[inputIndex]*weight_M_2[j * LEN_LAYER_M_1 + inputIndex];

		// apply bias
		mid_res_m_2[j] += bias_M_2[j];
		// relu
		if (mid_res_m_2[j] < 0) {
			mid_res_m_2[j] = 0;
		}
	}

	for (j = 0; j < LEN_LAYER_1; j++) {
		final_res_i[j] = 0;
		for(int inputIndex = 0; inputIndex < LEN_LAYER_M_2; inputIndex++) 
			final_res_i[j] += mid_res_m_2[inputIndex]*weight_1_T_ent[j * LEN_LAYER_M_2 + inputIndex];

		// apply bias
		final_res_i[j] += bias_1_ent[j];
	 }
    end = (final_res_i[0] >= 0) ? 1 : 0;   // 1 means "reject" and 0 means "not reject"


	if (VERBOSE) {
		if (end == 0) {
			printf("[dev %d] - [cur_hist_index=%ld] [accept] input_vec_i = {%ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld}\n", device, cur_hist_index, input_vec_i[0], input_vec_i[1], input_vec_i[2], input_vec_i[3], input_vec_i[4], input_vec_i[5], input_vec_i[6], input_vec_i[7], input_vec_i[8], input_vec_i[9], input_vec_i[10], input_vec_i[11]);
		} else {
			printf("[dev %d] - [cur_hist_index=%ld] [rej] input_vec_i = {%ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld}\n", device, cur_hist_index, input_vec_i[0], input_vec_i[1], input_vec_i[2], input_vec_i[3], input_vec_i[4], input_vec_i[5], input_vec_i[6], input_vec_i[7], input_vec_i[8], input_vec_i[9], input_vec_i[10], input_vec_i[11]);
		}
	}

    return end;
}


void update_flashnet(long io_queue_len, long io_latency, long io_throughput){
    /*
		Called by single update_thread in io_replayer.c, to insert latest completed IO's queue len
		when submitted, IO latency, and IO throughput into the historical pool.
        The historical pools to be Updated:
            prev_queue_len,
            prev_latency,
            prev_throughput,
    */

	// update previous queue len
	prev_queue_len[hist_index] = io_queue_len;

	// update previous latency
	prev_latency[hist_index] = io_latency;

	// update previous throughout
	prev_throughput[hist_index] = io_throughput;

	hist_index += 1;
}


void free_flashnet() {
	/*
		Free the memory that flashnet allocated
	*/
	
	free(prev_queue_len);
	free(prev_latency);
	free(prev_throughput);
}