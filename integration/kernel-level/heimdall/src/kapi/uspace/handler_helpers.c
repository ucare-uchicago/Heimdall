#include "handler_helpers.h"
#include <nvml.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

static bool nvml_is_setup = false;
static nvmlDevice_t dev;
static void nvml_setup(int devidx) {
    if (nvml_is_setup) return;

    nvmlReturn_t result;
    unsigned int device_count;

    result = nvmlInit();
    if (result != NVML_SUCCESS)
        exit(1);
    
    result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS)
        exit(1);

    result = nvmlDeviceGetHandleByIndex(devidx, &dev);
    if (result != NVML_SUCCESS)
        exit(1);
}

int nvml_get_procs_running(void) {
    nvml_setup(0);
    nvmlReturn_t result;
    unsigned int np=10;
    nvmlProcessInfo_t pis[10];
    result=nvmlDeviceGetComputeRunningProcesses(dev , &np, pis);
    if (result != NVML_SUCCESS)
        exit(1);
    //printf("procs: %d\n", np);
    return np;
}

int nvml_get_util_rate(void) {
    nvml_setup(0);
    nvmlReturn_t result;
    struct nvmlUtilization_st device_utilization;

    result = nvmlDeviceGetUtilizationRates(dev, &device_utilization);
    if (result != NVML_SUCCESS) {
        printf("error nvmlDeviceGetUtilizationRates %d\n", result);
        exit(1);
    }
    return device_utilization.gpu; 
}


