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

#ifndef __LINNOS_PREDICTORS_H
#define __LINNOS_PREDICTORS_H

#define LEN_INPUT 12
#define LEN_LAYER_0 12
#define LEN_LAYER_M_1 128
#define LEN_LAYER_M_2 16
#define LEN_LAYER_0_HALF 6
#define LEN_LAYER_1 1


#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/completion.h>
#else
#include <stdbool.h>
#endif

#include "variables.h"


#ifdef __KERNEL__
//these externs are for batching
extern bool* gpu_results;
extern u32* window_size_hist;
extern u32 n_used_gpu;
extern u32 n_skipped;
extern u32 ios_on_device[NUMBER_DEVICES];
bool batch_test(char *feat_vec, int n_vecs, long **weights);

extern struct GPU_weights gpu_weights[NUMBER_DEVICES];
#endif

bool fake_prediction_model(char *feat_vec, int n_vecs, long **weights);
bool gpu_batch_entry(char *feat_vec, int n_vecs, long **weights);
bool cpu_prediction_model(char *feat_vec, int n_vecs, long **weights);
bool cpu_prediction_model_plus_1(char *feat_vec, int n_vecs, long **weights);
bool cpu_prediction_model_plus_2(char *feat_vec, int n_vecs, long **weights);
bool cpu_prediction_model_linear(char *feat_vec, int n_vecs, long **weights);
void gpu_predict_batch(char *__feat_vec, int n_vecs, long **weights);
void gpu_predict_batch_plus_1(char *__feat_vec, int n_vecs, long **weights);
void gpu_predict_batch_plus_2(char *__feat_vec, int n_vecs, long **weights);

void multi_gpu_predict_batch(char *__feat_vec, int n_vecs, long **weights, int dev, int batch);
void multi_gpu_predict_batch_plus_1(char *__feat_vec, int n_vecs, long **weights, int dev, int batch);
void multi_gpu_predict_batch_plus_2(char *__feat_vec, int n_vecs, long **weights, int dev, int batch);


void predictors_mgpu_init(void);
int gpu_get_prediction(int dev, int batch, int id);
extern int PREDICT_GPU_SYNC;

#ifdef INFPOINT
#define DEADFLAG "O3"
#else
#define DEADFLAG "O0"
#endif

#endif