#include "pim_blas.h"

uint8_t *null_ptr = (uint8_t *)malloc(sizeof(uint8_t) * WORD_SIZE);

void blas_init(uint64_t num)
{
	runtime_init(0);
}

bool pim_add(PIMKernel micro_kernel, int len, uint8_t *in0, uint8_t *in1, uint8_t *out)
{
	InitFpgaTime();
	std::cout << " PIM_BLAS\t pim_add!\n";
	uint8_t *pim_out = AllocMem(out, len);

	PIM_OP_ATTRS add_attrs = micro_kernel.pim_op_attrs;
	ReadReg(PIM_REG::ABMR, null_ptr, WORD_SIZE);

	int idx = 0;
	int bank = 0;
	WriteReg(PIM_REG::CRF, (uint8_t *)(micro_kernel.code0), 4 * WORD_SIZE);
	for (int i = 0; i < add_attrs.code_iter; i++)
	{
		std::cout << " PIM_BLAS\t Code Start!\n";
		for (int j = 0; j < add_attrs.code0_iter; j++)
		{
			std::cout << " PIM_BLAS\t Code0 Start!\n";
#ifdef fpga_mode
			for (int k = 0; k < micro_kernel.code0_num_cmds; k++)
				bool ret = GetFpgaAddr(in0 + idx, in1 + idx, pim_out + idx, micro_kernel.code0_cmd[k], bank);
			SetFpgaAddr();
			WriteReg(PIM_REG::PIM_OP_MODE, null_ptr, WORD_SIZE);
			ExecuteKernel(in0, in1, pim_out, micro_kernel.code0_cmd[0], bank);
#else
			WriteReg(PIM_REG::PIM_OP_MODE, null_ptr, WORD_SIZE);
			for (int k = 0; k < micro_kernel.code0_num_cmds; k++)
				bool ret = ExecuteKernel(in0 + idx, in1 + idx, pim_out + idx, micro_kernel.code0_cmd[k], bank);
#endif
			idx += WORD_SIZE * 8 * NUM_BANK;
			std::cout << " PIM_BLAS\t Code0 Finished!\n";
		}
		idx = 0;
		bank = 1 - bank;
		std::cout << " PIM_BLAS\t Code Finished!\n";
	}
	ReadReg(PIM_REG::SBMR, null_ptr, 1);

	ReadMem(pim_out, out, len * UNIT_SIZE);

	PrintFpgaTime();

	return 1;
}

bool pim_mul(PIMKernel micro_kernel, int len, uint8_t *in0, uint8_t *in1, uint8_t *out)
{
	InitFpgaTime();
	std::cout << " PIM_BLAS\t pim_mul!\n";
	uint8_t *pim_out = AllocMem(out, len);

	PIM_OP_ATTRS mul_attrs = micro_kernel.pim_op_attrs;
	ReadReg(PIM_REG::ABMR, null_ptr, WORD_SIZE);

	int idx = 0;
	int bank = 0;
	WriteReg(PIM_REG::CRF, (uint8_t *)(micro_kernel.code0), 4 * WORD_SIZE);
	std::cout << "code  iter 1: " << mul_attrs.code_iter << std::endl;
	std::cout << "code0 iter 2: " << mul_attrs.code0_iter << std::endl;
	std::cout << "step : " << WORD_SIZE * 8 * NUM_BANK << std::endl;
	for (int i = 0; i < mul_attrs.code_iter; i++)
	{
		std::cout << " PIM_BLAS\t Code Start!\n";
		for (int j = 0; j < mul_attrs.code0_iter; j++)
		{
			std::cout << " PIM_BLAS\t Code0 Start!\n";
			WriteReg(PIM_REG::PIM_OP_MODE, null_ptr, WORD_SIZE);
			for (int k = 0; k < micro_kernel.code0_num_cmds; k++)
			{
				bool ret = ExecuteKernel(in0 + idx, in1 + idx, pim_out + idx, micro_kernel.code0_cmd[k], bank);
			}
			idx += WORD_SIZE * 8 * NUM_BANK;
			std::cout << " PIM_BLAS\t Code0 Finished!\n";
		}
		idx = 0;
		bank = 1 - bank;
		std::cout << " PIM_BLAS\t Code Finished!\n";
	}
	ReadReg(PIM_REG::SBMR, null_ptr, 1);

	ReadMem(pim_out, out, len * UNIT_SIZE);

	PrintFpgaTime();

	return 1;
}

/*
bool pim_bn(uint8_t *pim_mem, int len_in0, int len_in1, uint8_t *x, uint8_t *y, uint8_t *z)
{
	std::cout << " PIM_BLAS\t pim_bn!\n";
	return true;
}
*/

bool pim_gemv(PIMKernel micro_kernel, int len_in, int len_out, uint8_t *in, uint8_t *weight, uint8_t *out)
{
	InitFpgaTime();
	std::cout << " PIM_BLAS\t pim_gemv!\n";

	uint8_t *pim_w = weight;
	uint8_t *pim_out = AllocMem(out, len_out);

	PIM_OP_ATTRS gemv_attrs = micro_kernel.pim_op_attrs;
	ReadReg(PIM_REG::ABMR, null_ptr, WORD_SIZE);

	int in_idx = 0, w_idx = 0, out_idx = 0;
	int bank = 0;
	uint8_t *zeros = (uint8_t *)calloc(8 * NUM_UNIT_PER_WORD, UNIT_SIZE);
	for (int i = 0; i < gemv_attrs.code_iter; i++)
	{
		std::cout << " PIM_BLAS\t Code Start!\n";
		WriteReg(PIM_REG::CRF, (uint8_t *)micro_kernel.code0, WORD_SIZE);

		for (int j = 0; j < gemv_attrs.code0_iter; j++)
		{
			std::cout << " PIM_BLAS\t Code0 Start!\n";
			WriteReg(PIM_REG::SRF_M, in + in_idx, WORD_SIZE);

#ifdef fpga_mode
			for (int k = 0; k < micro_kernel.code0_num_cmds; k++)
				bool ret = GetFpgaAddr(in + in_idx, pim_w + w_idx, pim_out + out_idx, micro_kernel.code0_cmd[k], bank);
			SetFpgaAddr();
			WriteReg(PIM_REG::PIM_OP_MODE, null_ptr, WORD_SIZE);
			ExecuteKernel(in, pim_w, pim_out, micro_kernel.code0_cmd[0], bank); // 1 Trash Mem CMD
#else
			WriteReg(PIM_REG::PIM_OP_MODE, null_ptr, WORD_SIZE);
			BM_cnt = BM_cnt + 1;

			for (int k = 0; k < micro_kernel.code0_num_cmds; k++)
				bool ret = ExecuteKernel(in + in_idx, pim_w + w_idx, pim_out + out_idx, micro_kernel.code0_cmd[k], bank);
#endif
			w_idx += WORD_SIZE * 8 * NUM_BANK;
			in_idx += UNIT_SIZE * 8;
			std::cout << " PIM_BLAS\t Code0 Finished!\n";
		}
		WriteReg(PIM_REG::CRF, (uint8_t *)micro_kernel.code1, WORD_SIZE);
		for (int j = 0; j < gemv_attrs.code1_iter; j++)
		{
			std::cout << " PIM_BLAS\t Code1 Start!\n";
#ifdef fpga_mode
			for (int k = 0; k < micro_kernel.code0_num_cmds; k++)
				bool ret = GetFpgaAddr(in + in_idx, pim_w + w_idx, pim_out + out_idx, micro_kernel.code1_cmd[k], bank);
			SetFpgaAddr();
			WriteReg(PIM_REG::PIM_OP_MODE, null_ptr, WORD_SIZE);
			ExecuteKernel(in, pim_w, pim_out, micro_kernel.code0_cmd[0], bank); // 1 Trash Mem CMD
#else
			WriteReg(PIM_REG::PIM_OP_MODE, null_ptr, WORD_SIZE);
			for (int k = 0; k < micro_kernel.code1_num_cmds; k++)
				bool ret = ExecuteKernel(in + in_idx, pim_w + w_idx, pim_out + out_idx, micro_kernel.code1_cmd[k], bank);
			std::cout << " PIM_BLAS\t Code1 Finished!\n";
#endif
		}
		WriteReg(PIM_REG::GRF_B, zeros, WORD_SIZE * 8);
		in_idx = 0;
		w_idx = 0;
		// out_idx += WORD_SIZE * NUM_BANK;
		out_idx = 0;
		bank = 1 - bank;
		std::cout << " PIM_BLAS\t Code Finished!\n";
	}
	ReadReg(PIM_REG::SBMR, null_ptr, WORD_SIZE);

	ReadMem(pim_out, out, len_out * UNIT_SIZE);

	PrintFpgaTime();

	return 1;
}
