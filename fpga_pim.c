#include "fpga_pim.h"

uint64_t pimExecution(uint32_t addr, void *data, int iswrite) {
	printf("   fpga: %x\n", addr);
	return 1;
}



