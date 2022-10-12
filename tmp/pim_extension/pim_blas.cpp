#include "pim_blas.h"

uint8_t *null_ptr = (uint8_t *)malloc(sizeof(uint8_t) * WORD_SIZE);

void blas_init(uint64_t num)
{
	runtime_init(0);
}

/*
bool pim_add(uint8_t* pim_mem, int len, uint8_t *x, uint8_t *y, uint8_t *z) {
	std::cout << " PIM_BLAS\t pim_add!\n";
	return true;
}

bool pim_mul(uint8_t* pim_mem, int len, uint8_t *x, uint8_t *y, uint8_t *z) {
	std::cout << " PIM_BLAS\t pim_mul!\n";
	return true;
}

bool pim_bn(uint8_t* pim_mem, int len_in0, int len_in1, uint8_t *x, uint8_t *y, uint8_t *z) {
	std::cout << " PIM_BLAS\t pim_bn!\n";
	return true;
}
*/

bool pim_gemv(PIMKernel micro_kernel, int len_in, int len_out, uint8_t *in, uint8_t *weight, uint8_t *out)
{
	std::cout << " PIM_BLAS\t pim_gemv!\n";

	// will be deleted later (Executed outside blas library)
	uint8_t *pim_w = MapMemory(weight, len_in * len_out);
	uint8_t *pim_out = AllocMem(out, len_out);

	PIM_OP_ATTRS gemv_attrs = micro_kernel.pim_op_attrs;
	ReadReg(PIM_REG::ABMR, null_ptr, WORD_SIZE);

	int in_idx = 0, w_idx = 0, out_idx = 0;
	for (int i = 0; i < gemv_attrs.code_iter; i++)
	{
		WriteReg(PIM_REG::CRF, (uint8_t *)micro_kernel.code0, WORD_SIZE);
		for (int j = 0; j < gemv_attrs.code0_iter; j++)
		{
			WriteReg(PIM_REG::SRF, in + in_idx, WORD_SIZE);
			WriteReg(PIM_REG::PIM_OP_MODE, null_ptr, WORD_SIZE);
			for (int k = 0; k < micro_kernel.code0_num_cmds; k++)
			{
				bool ret = ExecuteKernel(in + in_idx, pim_w + w_idx, pim_out + out_idx, micro_kernel.code0_cmd[k]);
			}
			w_idx += WORD_SIZE * 8 * NUM_BANK;
			in_idx += UNIT_SIZE * 8;
		}
		std::cout << "> Code0 Finished!\n";
		WriteReg(PIM_REG::CRF, (uint8_t *)micro_kernel.code1, WORD_SIZE);
		for (int j = 0; j < gemv_attrs.code1_iter; j++)
		{
			WriteReg(PIM_REG::PIM_OP_MODE, null_ptr, WORD_SIZE);
			for (int k = 0; k < micro_kernel.code1_num_cmds; k++)
			{
				bool ret = ExecuteKernel(in + in_idx, pim_w + w_idx, pim_out + out_idx, micro_kernel.code1_cmd[k]);
			}
		}
		in_idx = 0;
		out_idx += WORD_SIZE * NUM_BANK;
	}
	ReadReg(PIM_REG::SBMR, null_ptr, WORD_SIZE);

	for (int i = 0; i < int((len_out + WORD_SIZE - 1) / WORD_SIZE); i++)
		ReadMem(pim_out + i * WORD_SIZE, out + i * WORD_SIZE, WORD_SIZE);

	return 1;
}
