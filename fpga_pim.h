#pragma once
#include <fcntl.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <stdbool.h>

// #include "../xdma/cdev_sgdma.h"
#ifdef __cplusplus
extern "C"
{
    long pimExecution(uint32_t addr, void *data, int iswrite);
}
#endif
