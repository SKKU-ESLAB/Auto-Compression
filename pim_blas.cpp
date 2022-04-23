#include "pim_blas.h"

bool pim_add(uint8_t* pim_mem, int len, uint8_t *x, uint8_t *y, uint8_t *z) {
	std::cout << " PIM_BLAS\t pim_add!\n";

	if (!isSuitableOps(PIM_OP::ADD))
		return false;
	
	uint8_t* pim_x = MapMemory(pim_mem, x, len);
	uint8_t* pim_y = MapMemory(pim_mem, y, len);
	uint8_t* pim_z = MapMemory(pim_mem, z, len);
	std::cout << "- Test1_ans: interval " << SIZE_ROW * NUM_BANK << std::endl;
	std::cout << "- Test1_x: " << pim_x - pim_mem << std::endl;
	std::cout << "- Test1_y: " << pim_y - pim_mem << std::endl;
	std::cout << "- Test1_z: " << pim_z - pim_mem << std::endl;

	PIM_OP_ATTRS add_attrs;
	add_attrs.ADD(pim_x, pim_y, pim_z, len);
	PIMKernel add_kernel = GetMicrokernelCode(PIM_OP::ADD, add_attrs);

	// Check if add_kernel is correctly created
	std::cout << "- Test2_ans: " << 0b01000010000000001000000000000000 << std::endl;
	std::cout << "- Test2_out: " << add_kernel.ukernel[0] << std::endl;

	AllocMem(pim_x);
	AllocMem(pim_y);
	AllocMem(pim_z);
	std::cout << "- Test3_ans: Pass AllocMem for now ← I didn't get it\n"; 

	WriteMem(pim_x, x, len);
	WriteMem(pim_y, y, len);
	std::cout << "- Test4_ans: " << (int)y[127] << std::endl;
	std::cout << "- Test4_out: " << (int)pim_y[127] << std::endl;

	bool ret = ExecuteKernel(pim_mem, pim_x, pim_y, pim_z, add_kernel);
	
	ReadMem(pim_z, z, len);

	FreeMem(pim_x);
	FreeMem(pim_y);
	FreeMem(pim_z);

	std::cout << " PIM_BLAS\t pim_add done!\n";

	return ret;
}

bool pim_mul(uint8_t* pim_mem, int len, uint8_t *x, uint8_t *y, uint8_t *z) {
	std::cout << " PIM_BLAS\t pim_mul!\n";

	if (!isSuitableOps(PIM_OP::MUL))
		return false;
	
	uint8_t* pim_x = MapMemory(pim_mem, x, len);
	uint8_t* pim_y = MapMemory(pim_mem, y, len);
	uint8_t* pim_z = MapMemory(pim_mem, z, len);
	std::cout << "- Test1_ans: interval " << SIZE_ROW * NUM_BANK << std::endl;
	std::cout << "- Test1_x: " << pim_x - pim_mem << std::endl;
	std::cout << "- Test1_y: " << pim_y - pim_mem << std::endl;
	std::cout << "- Test1_z: " << pim_z - pim_mem << std::endl;

	PIM_OP_ATTRS mul_attrs;
	mul_attrs.MUL(pim_x, pim_y, pim_z, len);
	PIMKernel mul_kernel = GetMicrokernelCode(PIM_OP::MUL, mul_attrs);

	// Check if mul_kernel is correctly created
	std::cout << "- Test2_ans: " << 0b01000010000000001000000000000000 << std::endl;
	std::cout << "- Test2_out: " << mul_kernel.ukernel[0] << std::endl;

	AllocMem(pim_x);
	AllocMem(pim_y);
	AllocMem(pim_z);
	std::cout << "- Test3_ans: Pass AllocMem for now ← I didn't get it\n"; 

	WriteMem(pim_x, x, len);
	WriteMem(pim_y, y, len);
	std::cout << "- Test4_ans: " << (int)y[127] << std::endl;
	std::cout << "- Test4_out: " << (int)pim_y[127] << std::endl;

	bool ret = ExecuteKernel(pim_mem, pim_x, pim_y, pim_z, mul_kernel);
	
	ReadMem(pim_z, z, len);

	FreeMem(pim_x);
	FreeMem(pim_y);
	FreeMem(pim_z);

	std::cout << " PIM_BLAS\t pim_mul done!\n";

	return ret;
}

bool pim_bn(uint8_t* pim_mem, int len_in0, int len_in1, uint8_t *x, uint8_t *y, uint8_t *z) {
	std::cout << " PIM_BLAS\t pim_bn!\n";
	return true;
}

bool pim_gemv(uint8_t* pim_mem, int len_in, int len_out, uint8_t *x, uint8_t *y, uint8_t *z) {
	std::cout << " PIM_BLAS\t pim_gemv!\n";

	if (!isSuitableOps(PIM_OP::GEMV))
		return false;
	uint8_t* pim_x;	
	uint8_t* pim_y = MapMemory(pim_mem, y, len_in * len_out);
	uint8_t* pim_z = MapMemory(pim_mem, z, len_out);
	std::cout << "- Test1_ans: interval " << SIZE_ROW * NUM_BANK << std::endl;
	std::cout << "- Test1_y: " << pim_y - pim_mem << std::endl;

	PIM_OP_ATTRS gemv_attrs;
	gemv_attrs.GEMV(pim_y, pim_z, len_in, len_out);
	PIMKernel gemv_kernel = GetMicrokernelCode(PIM_OP::GEMV, gemv_attrs);

	// Check if gemv_kernel is correctly created
	std::cout << "- Test2_ans: " << 0b01000010000000001000000000000000 << std::endl;
	std::cout << "- Test2_out: " << gemv_kernel.ukernel[0] << std::endl;

	AllocMem(pim_y);
	AllocMem(pim_z);
	std::cout << "- Test3_ans: Pass AllocMem for now ← I didn't get it\n"; 

	WriteMem(pim_y, y, len_in * len_out);
	std::cout << "- Test4_ans: " << (int)y[127] << std::endl;
	std::cout << "- Test4_out: " << (int)pim_y[127] << std::endl;

	bool ret = ExecuteKernel(pim_mem, pim_x, pim_y, pim_z, gemv_kernel);

	std::cout << "read result\n";	
	ReadMem(pim_z, z, len_out);

	FreeMem(pim_x);
	FreeMem(pim_y);
	FreeMem(pim_z);

	std::cout << " PIM_BLAS\t pim_gemv done!\n";

	return ret;
}
