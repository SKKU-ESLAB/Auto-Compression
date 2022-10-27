#include "pim_blas.h"

uint8_t *null_ptr = (uint8_t *)malloc(sizeof(uint8_t) * WORD_SIZE);

void blas_init(uint64_t num) {
	runtime_init(0);
}

bool C_pimblasAddPreprocess(int len, uint8_t **in0, uint8_t **in1) {
	*in0 = MapMemory(*in0, len * UNIT_SIZE);
	*in1 = MapMemory(*in1, len * UNIT_SIZE);
	return true;
}

bool C_pim_add(int len, uint8_t *in0, uint8_t *in1, uint8_t *out) {
	PIM_OP pim_op = PIM_OP::ADD;
	PIM_OP_ATTRS add_attrs = PIM_OP_ATTRS();
	add_attrs.ADD(len);
	PIMKernel micro_kernel = PIMKernel();
	micro_kernel = GetMicrokernelCode(pim_op, add_attrs);

	return pim_add(micro_kernel, len, in0, in1, out);	
}

bool pimblasAddPreprocess(PIMKernel *micro_kernel, int len, uint8_t **in0, uint8_t **in1) {
	PIM_OP pim_op = PIM_OP::ADD;
	PIM_OP_ATTRS add_attrs = PIM_OP_ATTRS();
	add_attrs.ADD(len);
	*micro_kernel = GetMicrokernelCode(pim_op, add_attrs);
	*in0 = MapMemory(*in0, len * UNIT_SIZE);
	*in1 = MapMemory(*in1, len * UNIT_SIZE);
	return true;
}

bool pim_add(PIMKernel micro_kernel, int len, uint8_t *in0, uint8_t *in1, uint8_t *out) {
	InitFpgaTime();
	InitFpgaData(1);
	if (DebugMode())
		std::cout << " PIM_BLAS\t pim_add!\n";
	uint8_t *pim_out = AllocMem(out, len * UNIT_SIZE);

	PIM_OP_ATTRS add_attrs = micro_kernel.pim_op_attrs;
	ReadReg(PIM_REG::ABMR, null_ptr, WORD_SIZE);

	int idx = 0;
	int bank = 0;
	WriteReg(PIM_REG::CRF, (uint8_t *)(micro_kernel.code0), 4 * WORD_SIZE);
	for (int i = 0; i < add_attrs.code_iter; i++) {
		if (DebugMode())
			std::cout << " PIM_BLAS\t Code Start!\n";
		for (int j = 0; j < add_attrs.code0_iter; j++) {
			if (DebugMode())
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
			if (DebugMode())
				std::cout << " PIM_BLAS\t Code0 Finished!\n";
		}
		idx = 0;
		bank = 1 - bank;
		if (DebugMode())
			std::cout << " PIM_BLAS\t Code Finished!\n";
	}
#ifdef fpga_mode
	SetFpgaData();
#endif
	ReadReg(PIM_REG::SBMR, null_ptr, 1);

	ReadMem(pim_out, out, len * UNIT_SIZE);

	if (DebugMode() && FpgaMode())
		PrintFpgaTime();

	return 1;
}

bool C_pimblasMulPreprocess(int len, uint8_t **in0, uint8_t **in1) {
	*in0 = MapMemory(*in0, len * UNIT_SIZE);
	*in1 = MapMemory(*in1, len * UNIT_SIZE);
	return true;
}

bool C_pim_mul(int len, uint8_t *in0, uint8_t *in1, uint8_t *out) {
	PIM_OP pim_op = PIM_OP::MUL;
	PIM_OP_ATTRS mul_attrs = PIM_OP_ATTRS();
	mul_attrs.MUL(len);
	PIMKernel micro_kernel = PIMKernel();
	micro_kernel = GetMicrokernelCode(pim_op, mul_attrs);

	return pim_mul(micro_kernel, len, in0, in1, out);	
}

bool pimblasMulPreprocess(PIMKernel *micro_kernel, int len, uint8_t **in0, uint8_t **in1) {
	PIM_OP pim_op = PIM_OP::MUL;
	PIM_OP_ATTRS mul_attrs = PIM_OP_ATTRS();
	mul_attrs.MUL(len);
	*micro_kernel = GetMicrokernelCode(pim_op, mul_attrs);
	*in0 = MapMemory(*in0, len * UNIT_SIZE);
	*in1 = MapMemory(*in1, len * UNIT_SIZE);
	return true;
}

bool C_pimblasGemvPreprocess(int len_in, int len_out, uint8_t **w) {
	int len_in_ = Ceiling(len_in, 8);
	int len_out_ = Ceiling(len_out, 4096);
	// w [8150, 1020]
	*w = GemvReshape(*w, len_in, len_out);
	// w [8192, 1024]
	*w = Transpose(*w, len_in, len_out);
	// w [2048, 4096]

	// w [4096, 8], [4096, 8]

	*w = MapMemory(*w, len_in_ * len_out_ * UNIT_SIZE);
	return true;
}

bool C_pim_gemv(int len_in, int len_out, uint8_t *in, uint8_t *w, uint8_t *out) {
	PIM_OP pim_op = PIM_OP::GEMV;
	PIM_OP_ATTRS gemv_attrs = PIM_OP_ATTRS();
	gemv_attrs.GEMV(len_in, len_out);
	PIMKernel micro_kernel = PIMKernel();
	micro_kernel = GetMicrokernelCode(pim_op, gemv_attrs);
	
	return pim_gemv(micro_kernel, len_in, len_out, in, w, out);
}

bool pimblasGemvPreprocess(PIMKernel *micro_kernel, int len_in, int len_out, uint8_t **w) {
	PIM_OP pim_op = PIM_OP::GEMV;
	PIM_OP_ATTRS gemv_attrs = PIM_OP_ATTRS();
	gemv_attrs.GEMV(len_in, len_out);
	*micro_kernel = GetMicrokernelCode(pim_op, gemv_attrs);

	int len_in_ = Ceiling(len_in, 8);
	int len_out_ = Ceiling(len_out, 4096);
	// w [8150, 1020]
	*w = GemvReshape(*w, len_in, len_out);
	// w [8192, 1024]
	if (micro_kernel->layout == 1) {
		*w = Transpose(*w, len_in, len_out);
	}
	// w [2048, 4096]

	// w [4096, 8], [4096, 8]

	*w = MapMemory(*w, len_in_ * len_out_ * UNIT_SIZE);
	return true;
}

bool pim_mul(PIMKernel micro_kernel, int len, uint8_t *in0, uint8_t *in1, uint8_t *out)
{
	InitFpgaTime();
	if (DebugMode())
		std::cout << " PIM_BLAS\t pim_mul!\n";
	uint8_t *pim_out = AllocMem(out, len * UNIT_SIZE);

	PIM_OP_ATTRS mul_attrs = micro_kernel.pim_op_attrs;
	ReadReg(PIM_REG::ABMR, null_ptr, WORD_SIZE);

	int idx = 0;
	int bank = 0;
	WriteReg(PIM_REG::CRF, (uint8_t *)(micro_kernel.code0), 4 * WORD_SIZE);
	if (DebugMode()) {
		std::cout << "code  iter 1: " << mul_attrs.code_iter << std::endl;
		std::cout << "code0 iter 2: " << mul_attrs.code0_iter << std::endl;
		std::cout << "step : " << WORD_SIZE * 8 * NUM_BANK << std::endl;
	}
	for (int i = 0; i < mul_attrs.code_iter; i++)
	{
		if (DebugMode())
			std::cout << " PIM_BLAS\t Code Start!\n";
		for (int j = 0; j < mul_attrs.code0_iter; j++)
		{
			if (DebugMode())
				std::cout << " PIM_BLAS\t Code0 Start!\n";
			WriteReg(PIM_REG::PIM_OP_MODE, null_ptr, WORD_SIZE);
			for (int k = 0; k < micro_kernel.code0_num_cmds; k++)
			{
				bool ret = ExecuteKernel(in0 + idx, in1 + idx, pim_out + idx, micro_kernel.code0_cmd[k], bank);
			}
			idx += WORD_SIZE * 8 * NUM_BANK;
			if (DebugMode())
				std::cout << " PIM_BLAS\t Code0 Finished!\n";
		}
		idx = 0;
		bank = 1 - bank;
		if (DebugMode())
			std::cout << " PIM_BLAS\t Code Finished!\n";
	}
#ifdef fpga_mode
	SetFpgaData();
#endif
	ReadReg(PIM_REG::SBMR, null_ptr, 1);

	ReadMem(pim_out, out, len * UNIT_SIZE);

	if (DebugMode() && FpgaMode())
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
	InitFpgaData(2);
	if (DebugMode())
		std::cout << " PIM_BLAS\t pim_gemv!\n";

	uint8_t *pim_w = weight;
	uint8_t *pim_out = AllocMem(out, Ceiling(len_out, 4096) * UNIT_SIZE);

	PIM_OP_ATTRS gemv_attrs = micro_kernel.pim_op_attrs;
	ReadReg(PIM_REG::ABMR, null_ptr, WORD_SIZE);

	int in_idx = 0, w_idx = 0, out_idx = 0;
	int bank = 0;
	uint8_t *zeros = (uint8_t *)calloc(8 * NUM_UNIT_PER_WORD, UNIT_SIZE);
	for (int i = 0; i < gemv_attrs.code_iter; i++)
	{
		if (DebugMode())
			std::cout << " PIM_BLAS\t Code Start!\n";
		WriteReg(PIM_REG::CRF, (uint8_t *)micro_kernel.code0, WORD_SIZE);

		for (int j = 0; j < gemv_attrs.code0_iter; j++)
		{
			if (DebugMode())
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
			// BM_cnt = BM_cnt + 1;  // >> KKM << 22.10.25 Just turn off for now. Needed for build

			for (int k = 0; k < micro_kernel.code0_num_cmds; k++)
				bool ret = ExecuteKernel(in + in_idx, pim_w + w_idx, pim_out + out_idx, micro_kernel.code0_cmd[k], bank);
#endif
			w_idx += WORD_SIZE * 8 * NUM_BANK;
			in_idx += UNIT_SIZE * 8;
			if (DebugMode())
				std::cout << " PIM_BLAS\t Code0 Finished!\n";
		}
		WriteReg(PIM_REG::CRF, (uint8_t *)micro_kernel.code1, WORD_SIZE);
		for (int j = 0; j < gemv_attrs.code1_iter; j++)
		{
			if (DebugMode())
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
			if (DebugMode())
				std::cout << " PIM_BLAS\t Code1 Finished!\n";
#endif
		}
		WriteReg(PIM_REG::GRF_B, zeros, WORD_SIZE * 8);
		in_idx = 0;
		w_idx = 0;
		// out_idx += WORD_SIZE * NUM_BANK;
		out_idx = 0;
		bank = 1 - bank;
		if (DebugMode())
			std::cout << " PIM_BLAS\t Code Finished!\n";
	}

#ifdef fpga_mode
	SetFpgaData();
#endif

	ReadReg(PIM_REG::SBMR, null_ptr, WORD_SIZE);

	ReadMem(pim_out, out, len_out * UNIT_SIZE);

	if (DebugMode() && FpgaMode())
		PrintFpgaTime();

	return 1;
}

uint8_t *GemvReshape(uint8_t *w, int m, int n) {
	int m_ = Ceiling(m, 8);
	int n_ = Ceiling(n, 4096);
	uint8_t *w_ = (uint8_t *)malloc(sizeof(uint16_t) * m_ * n_);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			((uint16_t *)w_)[i * m_ + j] = ((uint16_t *)w)[i * m + j];
		}
	}
	return w_;
}

uint8_t *Transpose(uint8_t *w, int m, int n) {
	m = Ceiling(m, 8);
	n = Ceiling(n, 4096);
	uint8_t *w_ = (uint8_t *)malloc(sizeof(uint16_t) * m * n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			((uint16_t *)w_)[j * n + i] = ((uint16_t *)w)[i * m + j];
		}
	}
	return w_;
}
