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

#include <linux/sched/signal.h>
#include <linux/slab.h>
#include <linux/time.h>
#include <linux/ktime.h>
#include <linux/delay.h>
#include <linux/blkdev.h>
#include <linux/string.h>
#include <linux/completion.h>
#include <linux/vmalloc.h>
#include "predictors.h"
#include "lake_shm.h"
#include "queue_depth.h"
#include "helpers.h"

#define SET_SYSCTL_DEBUG 0

extern unsigned long sysctl_lake_enable_linnos;
extern unsigned long sysctl_lake_linnos_debug;

static char *predictor_str = "fake";
module_param(predictor_str, charp, 0444);
MODULE_PARM_DESC(predictor_str, "What predictor to use: fake, cpu, gpu, batchtest, queudepth");

static char *cubin_path = "linnos.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to linnos.cubin in case you're using gpu predictor");

int model_size = 0;
module_param(model_size, int, 0444);
MODULE_PARM_DESC(model_size, "what model to use, 0 default, 1 +1, 2 +2");

//adding a model to a device requires:
// 1. include the header with the weights
// 2. put device name in devices
// 3. set the pointers into a new array in weights (dont mess with the ending 0)

#include "sde.h"


#include "weights_header/w_Trace_dev0.h"
#include "weights_header/w_Trace_dev1.h"

long *weights[][8] = {
	//NN+2
	{weight_0_T_dev0, weight_3_T_dev0, bias_0_dev0, bias_3_dev0, weight_1_T_dev0, bias_1_dev0 ,weight_2_T_dev0, bias_2_dev0},
	{weight_0_T_dev1, weight_3_T_dev1, bias_0_dev1, bias_3_dev1, weight_1_T_dev1, bias_1_dev1 ,weight_2_T_dev1, bias_2_dev1},
};

static const char *devices[] = {
	"/dev/nvme0n1",
	"/dev/nvme2n1",
	0
};

//the predictor function to use
bool (*fptr)(char*,int,long**);

bool is_qdepth = false;
bool is_batch_test = false;
bool is_gpu_inf = false;

/*
 *  Helpers for Batch test
 */
static void batch_test_attach(void) {
	int i;
	fptr = batch_test;
	window_size_hist = vmalloc(512);
	for (i=0;i<512;i++) window_size_hist[i] = 0;
}
static void batch_test_detach(void) {
	int i;
	for (i=0;i<512;i++)
		if (window_size_hist[i] != 0)
			pr_warn("%d:\t%u\n", i, window_size_hist[i]);
	vfree(window_size_hist);
}

/*
 *  Helpers for queue depth stats
 */
static int qdepth_attach(void) {
	int err;
	err = qd_init(); //this sets ptr
	if (err != 0) return err;
	usleep_range(5,10); //lets chill, why not
	sysctl_lake_linnos_debug = 3; //this enables storing batches
	return 0;
}
static void qdepth_detach(void) {
	qd_writeout();
}

/*
 *  Actual hook code
 */
static int parse_arg(void) {
	if (!strcmp("fake", predictor_str)) {
		fptr = fake_prediction_model;
	} else if (!strcmp("cpu", predictor_str)) {
		if (model_size == 0) {
			fptr = cpu_prediction_model;
			no_reject = false;
		}
		else if (model_size == 1) {
			fptr = cpu_prediction_model_plus_1;
			no_reject = true;
		}
		else if(model_size == 3){
			fptr = cpu_prediction_model_linear;
			no_reject = false;
		}
		else {
			fptr = cpu_prediction_model_plus_2;
			no_reject = false;
		}
		pr_warn("Inserting CPU prediction with %d extra layers\n", model_size);
	}else if (!strcmp("gpu", predictor_str)) {
		is_gpu_inf = true;
	} else if (!strcmp("batchtest", predictor_str)) {
		pr_warn("Inserting batch test prediction\n");
		is_batch_test = true;
	} else if (!strcmp("queue_depth", predictor_str)) {
		pr_warn("Inserting queue_depth\n");
		//set fake so we go through everything
		is_qdepth = true;
		fptr = fake_prediction_model;
	} else {	
		pr_warn("Invalid predictor argument\n");
		return -2;
	}
	return 0;
}


/*
 *  Helpers for GPU inference
 */
static int gpu_attach(void) {
	int i, ndev=0;
	const char *devs;
	
	fptr = gpu_batch_entry;
	for(devs = devices[0], i=0 ; devs != 0 ; devs = devices[++i]) 
		ndev++;
	pr_warn("initing for %d devices\n", ndev);
	multi_initialize_gpu(cubin_path, 512, ndev);
	window_size_hist = vmalloc(256);
	for (i=0;i<256;i++) 
		window_size_hist[i] = 0;
	if(model_size==0) {
	 	cpu_gpu_threshold = 8;
		max_batch_size = 10;
	 	window_size_ns = 5*_us;
		no_reject = false;
	} else if (model_size == 1) {
		window_size_ns = 40*_us;
	 	cpu_gpu_threshold = 4;
		max_batch_size = 8;
		no_reject = true;
	} else if (model_size == 2) {
	 	cpu_gpu_threshold = 4;
	 	window_size_ns = 40*_us;
	 	max_batch_size = 6;
		no_reject = false;
	}
	predictors_mgpu_init();
	return 0;
}

static void gpu_detach(void) {
	const char *devs;
	int i;
	for(devs = devices[0], i=0 ; devs != 0 ; devs = devices[++i]) {
		multi_gpu_cuda_cleanup_dev(&gpu_weights[i], i);
	}
	
	for (i=0;i<128;i++)
		if (window_size_hist[i] != 0)
			pr_warn("%d:\t%u\n", i, window_size_hist[i]);

	pr_warn("GPU was used %u times\n", n_used_gpu);
	pr_warn("Batch skipped %u times\n", n_skipped);
	// for (i=0;i<NUMBER_DEVICES;i++) {
	// 	pr_warn("IOs on device %d: %u\n", i, ios_on_device[i]);
	// }
	cuCtxDestroy(cuctx);
}
static void gpu_copy_weight(int idx) {
	long **wts = weights[idx];
	pr_warn("Copying weights for idx %d\n", idx);
	copy_weights(wts, &gpu_weights[idx]);

	first_weight_ptr_to_dev[idx] = wts[0];
}

static int attach_to_queue(int idx) {
	struct block_device *dev;
	struct request_queue *q;
	long **wts = weights[idx];

	pr_warn("Attaching to queue on %s\n", devices[idx]);
	dev = blkdev_get_by_path(devices[idx], FMODE_READ|FMODE_WRITE, THIS_MODULE);
	if(IS_ERR(dev)) {
		pr_warn("Error getting dev by path (%s): %ld\n", devices[idx], PTR_ERR(dev));
		return -2;
	}
	q = bdev_get_queue(dev);

	//more spaggheti, nice
	if (is_gpu_inf) 
		gpu_copy_weight(idx);

	q->weight_0_T = wts[0];
	q->weight_1_T = wts[1];
	q->bias_0 = wts[2];
	q->bias_1 = wts[3];

	q->weight_2_T = wts[4];
	q->bias_2 = wts[5];
	q->weight_3_T = wts[6];
	q->bias_3 = wts[7];

	q->predictor = fptr;
	q->ml_enabled = true;
	sysctl_lake_enable_linnos = true;
	pr_warn("Attached!\n");
	return 0;
}

static int gpu_detach_queue(int idx) {
	struct block_device *dev;
	struct request_queue *q;

	pr_warn("Dettaching queue on %s\n", devices[idx]);
	dev = blkdev_get_by_path(devices[idx], FMODE_READ|FMODE_WRITE, THIS_MODULE);
	if(IS_ERR(dev)) {
		pr_warn("Error getting dev by path (%s): %ld\n", devices[idx], PTR_ERR(dev));
		return -1;
	}
	q = bdev_get_queue(dev);

	q->ml_enabled = false;
	sysctl_lake_enable_linnos = false;
	usleep_range(100,200);
	q->predictor = 0;
	q->weight_0_T = 0;
	q->weight_1_T = 0;
	q->bias_0 = 0;
	q->bias_1 = 0;

	q->weight_2_T = 0;
	q->bias_2 = 0;
	q->weight_3_T = 0;
	q->bias_3 = 0;

	pr_warn("Dettached!\n");
	return 0;
}

/**
 * Program main
 */
static int __init hook_init(void)
{
	const char *devs;
	int i, err;

	sysctl_lake_linnos_debug = SET_SYSCTL_DEBUG;
	err = parse_arg();
	if(err < 0) return -2;

	//special handling
	if(is_batch_test) batch_test_attach();
	if(is_qdepth) 
		if(qdepth_attach() != 0)
			return -2;
	if(is_gpu_inf) gpu_attach();

	for(devs = devices[0], i=0 ; devs != 0 ; devs = devices[++i]) {
		err = attach_to_queue(i);
		if (err) return err;
	}

	return 0;
}

static void __exit hook_fini(void)
{
	const char *devs;
	int i, err;

	sysctl_lake_linnos_debug = 0;
	for(devs = devices[0], i=0 ; devs != 0 ; devs = devices[++i]){
		err = gpu_detach_queue(i);
		if (err) return;
	}

	if(is_qdepth) qdepth_detach();
	if(is_batch_test) batch_test_detach();
	if(is_gpu_inf) gpu_detach();
}

module_init(hook_init);
module_exit(hook_fini);

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("Kernel predictor hooks for LAKE-linnos");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
