#ifndef __PIM_RUNTIME_H_
#define __PIM_RUNTIME_H_

#include <iostream>
#include <pthread.h>
#include "pim_config.h"
#include "pim_func_sim/pim_func_sim.h"

void runtime_init(uint64_t num);

// PIM Preprocessor
bool isSuitableOps(PIM_OP op, int len0, int len1);
uint8_t *MapMemory(uint8_t *data, size_t len);
PIMKernel GetMicrokernelCode(PIM_OP op, PIM_OP_ATTRS op_attrs);

// PIM Memory Manager
uint8_t *AllocMem(uint8_t *data, size_t len);
void FreeMem(uint8_t *pim_addr);
size_t ReadMem(uint8_t *pim_addr, uint8_t *data, size_t len);
size_t WriteMem(uint8_t *pim_addr, uint8_t *data, size_t len);
size_t ReadReg(PIM_REG pim_reg, uint8_t *data, size_t len);
size_t WriteReg(PIM_REG pim_reg, uint8_t *data, size_t len);

// PIM Kernel Executor
void ExecuteKernel_1COL(uint8_t *pim_target, bool is_write, int bank);
void ExecuteKernel_8COL(uint8_t *pim_target, bool is_write, int bank);
bool ExecuteKernel(uint8_t *pim_x, uint8_t *pim_y, uint8_t *pim_z, PIM_CMD pim_cmd, int bank);

// Some tools
uint64_t Ceiling(uint64_t num, uint64_t stride);
void TryAddTransaction(uint8_t *pim_mem, uint8_t *data, bool is_write);
uint64_t GetAddress(int channel, int rank, int bankgroup, int bank, int row, int column);
bool DebugMode();
bool FpgaMode();
bool ComputeMode();
void pimExecution(uint8_t *pim_addr, uint8_t *data, bool is_write);

#endif // __PIM_RUNTIME_H_
