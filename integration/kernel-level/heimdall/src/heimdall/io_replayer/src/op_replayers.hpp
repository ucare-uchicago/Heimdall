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


#ifndef __OP_REPLAYER_H__
#define __OP_REPLAYER_H__

#include <pthread.h>
#include <unistd.h>
#include <thread>

#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

#include "replayer.hpp"


#define N_THREADS 8

struct Thread_arg {
    Trace *trace;
    uint32_t device;
    pthread_barrier_t *sync_barrier;
    uint64_t start_ts;
    std::string type; 
    int is_warmup;

    void (*executor)(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf);
};


void baseline_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf);
void heimdall_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf);
void random_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf);

void* replayer_fn(void* arg);


#endif