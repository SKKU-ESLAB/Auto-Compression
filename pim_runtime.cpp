#include "pim_runtime.h"

// PIM Preprocessor
bool isSuitableOps(PIM_OP op) {
	std::cout << "isSuitableOps!\n";
	return 1;
}

uint8_t* MapMemory(uint8_t *pmemAddr_, uint8_t *data, size_t len) {
	std::cout << "MapMemory!\n";
	uint64_t addr = 0;
	return (uint8_t*)(pmemAddr_+addr);
}

PIMKernel GetMicrokernelCode(PIM_OP op, PIM_OP_ATTRS op_attrs) {
	std::cout << "GetMicrokernelCode!\n";
	PIMKernel new_kernel;
	new_kernel.SetMicrokernelCode(op);
	return new_kernel;
}

// PIM Memory Manager
bool AllocMem(uint8_t* pim_mem) {
	std::cout << "AllocMem!\n";
	return 1;
}

void FreeMem(uint8_t* pim_mem) {
	std::cout << "FreeMem!\n";
}

size_t ReadMem(uint8_t* pim_mem, uint8_t *data, size_t len) {
	std::cout << "ReadMem!\n";
}

size_t WriteMem(uint8_t* pim_mem, uint8_t *data, size_t len) {
	std::cout << "WriteMem!\n";
}

// PIM Kernel Executor
bool ExecuteKernel(PIMKernel pim_kernel) {
	std::cout << "ExecuteKernel!\n";
	return 1;
}

