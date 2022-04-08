#include "pim_blas.h"

bool pim_add(uint8_t* pmemAddr_, int len, uint8_t *x, uint8_t *y, uint8_t *z) {
	if (!isSuitableOps(PIM_OP::ADD))
		return false;
	
	uint8_t* pim_x = MapMemory(pmemAddr_, x, len);
	uint8_t* pim_y = MapMemory(pmemAddr_, y, len);
	uint8_t* pim_z = MapMemory(pmemAddr_, z, len);

	PIM_OP_ATTRS add_attrs;
	add_attrs.ADD(pim_x, pim_y, pim_z, len);
	PIMKernel add_kernel = GetMicrokernelCode(PIM_OP::ADD, add_attrs);

	AllocMem(pim_x);
	AllocMem(pim_y);
	AllocMem(pim_z);

	WriteMem(pim_x, x, len);
	WriteMem(pim_y, y, len);
	
	bool ret = ExecuteKernel(add_kernel);

	ReadMem(pim_z, z, len);

	FreeMem(pim_x);
	FreeMem(pim_y);
	FreeMem(pim_z);

	return ret;
}


