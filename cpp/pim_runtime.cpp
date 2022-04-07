#include <torch/extension.h>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>

// PIM Preprocessor
// bool isSuitableOps(PIM_OP op) {
void isSuitableOps(PIM_OP op) {
	std::cout << "isSuitableOps!\n";
}

//PIMMem MapMemory(uint64_t addr, size_t len) {
PIMMem MapMemory(uint64_t addr, size_t len) {
	std::cout << "MapMemory!\n";

}

PIMKernel GetMicrokernelCode(PIM_OP op, PIM_OP_ATTRS op_attrs) {
PIMKernel GetMicrokernelCode(PIM_OP op, PIM_OP_ATTRS op_attrs) {
	std::cout << "GetMicrokernelCode!\n";
}

// PIM Memory Manager
bool AllocMem(PIMMem pim_mem) {
bool AllocMem(PIMMem pim_mem) {
	std::cout << "AllocMem!\n";
}

void FreeMem(PIMMem pim_mem) {
void FreeMem(PIMMem pim_mem) {
	std::cout << "FreeMem!\n";

}

size_t ReadMem(PIMMem pim_mem, uint8_t *data, size_t len) {
size_t ReadMem(PIMMem pim_mem, uint8_t *data, size_t len) {
	std::cout << "ReadMem!\n";

}

size_t WriteMem(PIMMem pim_mem, uint8_t *data, size_t len) {
size_t WriteMem(PIMMem pim_mem, uint8_t *data, size_t len) {
	std::cout << "WriteMem!\n";

}

// PIM Kernel Executor
bool ExecuteKernel(PIMKernel pim_kernel) {
bool ExecuteKernel(PIMKernel pim_kernel) {
	std::cout << "ExecuteKernel!\n";

}

