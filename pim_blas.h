#ifndef __PIM_BLAS_H_
#define __PIM_BLAS_H_

#include "pim_runtime.h"
#include "pim_config.h"

bool pim_add(uint8_t* pim_mem, int n, uint8_t *x, uint8_t *y, uint8_t *z);
bool pim_mul(uint8_t* pim_mem, int n, uint8_t *x, uint8_t *y, uint8_t *z);
bool pim_bn(uint8_t* pim_mem, int l, int f, uint8_t *x, uint8_t *y, uint8_t *z);
bool pim_gemv(uint8_t* pim_mem, int m, int n, uint8_t *x, uint8_t *y, uint8_t *z);
bool pim_lstm(uint8_t* pim_mem, int m, int n, uint8_t *x, uint8_t *y, uint8_t *z);

#endif  // __PIM_BLAS_H_
