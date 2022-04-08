#include "pim_runtime.h"

uint64_t next_addr = 0;
unsigned int burstSize_ = 32;

int ch_pos_ = 0;
int ba_pos_ = ch_pos_ + 4;
int bg_pos_ = ba_pos_ + 2;
int co_pos_ = bg_pos_ + 2;
int ra_pos_ = co_pos_ + 5;
int ro_pos_ = ra_pos_ + 0;
int shift_bits_ = 5;

// PIM Preprocessor
bool isSuitableOps(PIM_OP op) {
	std::cout << "  PIM_RUNTIME\t isSuitableOps!\n";
	
	// For now, just return True
	return true;
}

uint8_t* MapMemory(uint8_t *pim_mem, uint8_t *data, size_t len) {
	std::cout << "  PIM_RUNTIME\t MapMemory!\n";
	uint64_t addr = next_addr;
	next_addr += Ceiling(len * UNIT_SIZE, SIZE_ROW * NUM_BANK);
	return (uint8_t*)(pim_mem+addr);
}

PIMKernel GetMicrokernelCode(PIM_OP op, PIM_OP_ATTRS op_attrs) {
	std::cout << "  PIM_RUNTIME\t GetMicrokernelCode!\n";
	PIMKernel new_kernel;

	// For now, Make same ukernel without considering op_attrs
	new_kernel.SetMicrokernelCode(op);
	return new_kernel;
}

// PIM Memory Manager
bool AllocMem(uint8_t* pim_mem) {
	std::cout << "  PIM_RUNTIME\t AllocMem!\n";
	
	// For now, I didn't get it
	// So, just return True
	return true;
}

void FreeMem(uint8_t* pim_mem) {
	std::cout << "  PIM_RUNTIME\t FreeMem!\n";
}

size_t ReadMem(uint8_t* pim_mem, uint8_t *data, size_t len) {
	std::cout << "  PIM_RUNTIME\t ReadMem!\n";
	uint64_t strided_size = Ceiling(len * UNIT_SIZE, SIZE_WORD * NUM_BANK);
	for (int offset = 0; offset < strided_size; offset += SIZE_WORD)
		TryAddTransaction(pim_mem + offset, data + offset, false); 
}

size_t WriteMem(uint8_t* pim_mem, uint8_t *data, size_t len) {
	std::cout << "  PIM_RUNTIME\t WriteMem!\n";
	uint64_t strided_size = Ceiling(len * UNIT_SIZE, SIZE_WORD * NUM_BANK);

	for (int offset = 0; offset < strided_size; offset += SIZE_WORD)
		TryAddTransaction(pim_mem + offset, data + offset, true); 
}

// PIM Kernel Executor
bool ExecuteKernel(PIMKernel pim_kernel) {
	std::cout << "  PIM_RUNTIME\t ExecuteKernel!\n";

	return 1;
}

// Some tool
uint64_t Ceiling(uint64_t num, uint64_t stride) {
	return ((num + stride - 1) / stride) * stride;
}

void TryAddTransaction(uint8_t* pim_mem, uint8_t* data, bool is_write) {
	if (is_write)
		std::memcpy(pim_mem, data, burstSize_);
	else
		std::memcpy(data, pim_mem, burstSize_);
}

uint64_t GetAddress(int channel, int rank, int bankgroup, int bank, int row, int column) {
	uint64_t hex_addr = 0;
	hex_addr += ((uint64_t)channel) << ch_pos_;
	hex_addr += ((uint64_t)rank) << ra_pos_;
	hex_addr += ((uint64_t)bankgroup) << bg_pos_;
	hex_addr += ((uint64_t)bank) << ba_pos_;
	hex_addr += ((uint64_t)row) << ro_pos_;
	hex_addr += ((uint64_t)column) << co_pos_;
	return hex_addr << shift_bits_;
}

