#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <pthread.h>
#include <stdbool.h>

typedef struct ThreadPool ThreadPool;

typedef void (*Task)(int64_t);

ThreadPool* createThreadPool(size_t numThreads);
void destroyThreadPool(ThreadPool* pool);
void submitTask(ThreadPool* pool, Task task, int64_t arg);

#endif