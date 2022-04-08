#ifndef __PIM_RUNTIME_H_
#define __PIM_RUNTIME_H_

#include <iostream>
#include "pim_config.h"

// PIM Preprocessor
bool isSuitableOps(PIM_OP op);
uint8_t* MapMemory(uint8_t *pmemAddr_, uint8_t *data, size_t len);
PIMKernel GetMicrokernelCode(PIM_OP op, PIM_OP_ATTRS op_attrs);

// PIM Memory Manager
bool AllocMem(uint8_t* pim_mem);
void FreeMem(uint8_t* pim_mem);
size_t ReadMem(uint8_t* pim_mem, uint8_t *data, size_t len);
size_t WriteMem(uint8_t* pim_mem, uint8_t *data, size_t len);

// PIM Kernel Executor
bool ExecuteKernel(PIMKernel pim_kernel);

#endif  // __PIM_RUNTIME_H_
