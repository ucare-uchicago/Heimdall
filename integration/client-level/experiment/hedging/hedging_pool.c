#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include "hedging_pool.h"

typedef struct ThreadPool {
    pthread_t* threads;
    size_t numThreads;
    void (*func)(int64_t);
    int64_t args;
    pthread_mutex_t mutex;
    pthread_cond_t task_coming;
    pthread_cond_t func_null;
    bool stop;
} ThreadPool;

typedef void (*Task)(int64_t);


void* thread_func(void* data) {
    ThreadPool* pool = (ThreadPool*)data;

    while (1) {
        pthread_mutex_lock(&pool->mutex);

        while (!pool->stop && pool->func == NULL) {
            pthread_cond_wait(&pool->task_coming, &pool->mutex);     // the worker is waiting for a task to come
        }

        if (pool->stop) {
            pthread_mutex_unlock(&pool->mutex);
            pthread_exit(NULL);
        }

        Task task = pool->func;
        int64_t arg = pool->args;

        pool->func = NULL;
        pool->args = -1;
        pthread_cond_signal(&pool->func_null);

        pthread_mutex_unlock(&pool->mutex);

        task(arg);
    }

    return NULL;
}

ThreadPool* createThreadPool(size_t numThreads) {
    ThreadPool* pool = (ThreadPool*)malloc(sizeof(ThreadPool));
    if (pool == NULL) {
        perror("Unable to allocate memory for ThreadPool");
        exit(EXIT_FAILURE);
    }

    pool->threads = (pthread_t*)malloc(numThreads * sizeof(pthread_t));
    if (pool->threads == NULL) {
        perror("Unable to allocate memory for threads");
        exit(EXIT_FAILURE);
    }

    pool->numThreads = numThreads;
    pool->func = NULL;
    pool->args = -1;
    pool->stop = false;

    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        perror("Mutex initialization failed");
        exit(EXIT_FAILURE);
    }

    if (pthread_cond_init(&pool->task_coming, NULL) != 0) {
        perror("Condition variable initialization failed");
        exit(EXIT_FAILURE);
    }

    if (pthread_cond_init(&pool->func_null, NULL) != 0) {
        perror("Condition variable initialization failed");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < numThreads; ++i) {
        if (pthread_create(&pool->threads[i], NULL, thread_func, pool) != 0) {
            perror("Thread creation failed");
            exit(EXIT_FAILURE);
        }
    }

    return pool;
}

void destroyThreadPool(ThreadPool* pool) {
    if (pool == NULL)
        return;

    pool->stop = true;
    pthread_cond_broadcast(&pool->task_coming);
    pthread_cond_broadcast(&pool->func_null);

    for (size_t i = 0; i < pool->numThreads; ++i) {
        if (pthread_join(pool->threads[i], NULL) != 0) {
            perror("Thread join failed");
        }
    }

    free(pool->threads);
    free(pool);
}

void submitTask(ThreadPool* pool, Task task, int64_t arg) {
    pthread_mutex_lock(&pool->mutex);

    while (pool->func != NULL) {
        pthread_cond_wait(&pool->func_null, &pool->mutex);
    }

    pool->func = task;
    pool->args = arg;
    pthread_cond_signal(&pool->task_coming);

    pthread_mutex_unlock(&pool->mutex);
}
