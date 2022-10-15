#include "pim_runtime.h"
#include <stdio.h>

uint8_t *pim_mem;
uint64_t next_addr = 0;
unsigned int burstSize_ = 32;

uint8_t data_temp_[32];
uint64_t ukernel_access_size_ = WORD_SIZE * 8 * NUM_BANK;
uint64_t ukernel_count_per_pim_;

typedef struct
{
	uint64_t addr = 0;
	uint64_t size = 0;
} PIMAllocList_t;

PIMAllocList_t PIMAllocList[1024];
uint64_t PIMAllocList_idx;

typedef struct
{
	int ch;
	uint8_t *pim_addr;
	uint8_t *data;
	bool is_write;
} thr_param_t;

pthread_t thr[NUM_CHANNEL * 8];
thr_param_t thr_param[NUM_CHANNEL * 8];

pthread_t thr_grp[NUM_CHANNEL];
thr_param_t thr_grp_param[NUM_CHANNEL];

pthread_barrier_t thr_barrier[NUM_CHANNEL];
pthread_mutex_t print_mutex;

static void *TryThreadGroupAddTransaction(void *input_);
static void *TryThreadAddTransaction(void *input_);

FILE *fp = fopen("output.txt", "w");
uint64_t pim_base;
int clock_ = 0;

// For Compute Mode
PimFuncSim *pim_func_sim;

// For Fpga Mode
int num_fpga_addr = 0;
uint32_t *fpga_addr_queue;
uint64_t spend_time_ns = 0;

void runtime_init(uint64_t num)
{
	pim_mem = (uint8_t *)calloc(LEN_PIM, 1); // Just for a while
	pim_base = (uint64_t)pim_mem;
	if (FpgaMode())
		fpga_addr_queue = (uint32_t *)malloc(sizeof(uint32_t) * 64);

	if (ComputeMode())
	{
		pim_func_sim = new PimFuncSim();
		pim_func_sim->init(pim_mem, LEN_PIM, WORD_SIZE);
	}

	std::cout << "  PIM_BASE_ADDR : " << pim_base << "\n";
}

// PIM Preprocessor
bool isSuitableOps(PIM_OP op, int len0, int len1)
{
	std::cout << "  PIM_RUNTIME\t isSuitableOps!\n";
	// For now, just return True
	return true;
}

PIMKernel GetMicrokernelCode(PIM_OP op, PIM_OP_ATTRS op_attrs)
{
	std::cout << "  PIM_RUNTIME\t GetMicrokernelCode!\n";
	PIMKernel new_kernel;
	ukernel_count_per_pim_ = Ceiling(op_attrs.len_in * UNIT_SIZE, ukernel_access_size_) / ukernel_access_size_;
	// For now, Make same ukernel without considering op_attrs
	new_kernel.SetMicrokernelCode(op, op_attrs);
	return new_kernel;
}

uint8_t *MapMemory(uint8_t *data, size_t size)
{
	std::cout << "  PIM_RUNTIME\t MapMemory!\n";
	uint64_t addr = next_addr;
	uint64_t alloc_size = Ceiling(size, 8 * WORD_SIZE * NUM_BANK);

	PIMAllocList[PIMAllocList_idx].addr = addr;
	PIMAllocList[PIMAllocList_idx].size = alloc_size;
	PIMAllocList_idx++;

	WriteMem(pim_mem + addr, data, size); // Mapping!

	next_addr += alloc_size;
	return (uint8_t *)(pim_mem + addr);
}

// PIM Memory Manager
uint8_t *AllocMem(uint8_t *data, size_t size)
{
	std::cout << "  PIM_RUNTIME\t AllocMem!\n";
	uint64_t addr = next_addr;
	uint64_t alloc_size = Ceiling(size, 8 * WORD_SIZE * NUM_BANK);

	PIMAllocList[PIMAllocList_idx].addr = addr;
	PIMAllocList[PIMAllocList_idx].size = alloc_size;
	PIMAllocList_idx++;

	next_addr += alloc_size;
	return (uint8_t *)(pim_mem + addr);
}

void FreeMem(uint8_t *pim_addr)
{
	std::cout << "  PIM_RUNTIME\t FreeMem!\n";

	for (int i = 0; i < PIMAllocList_idx; i++)
	{
		if (PIMAllocList[i].addr == (uint64_t)(pim_addr - pim_base))
		{
			PIMAllocList[i].size = 0;
			return;
		}
	}
	std::cout << "  PIM_RUNTIME\t FreeMem → Not found... What?\n";
}

size_t ReadMem(uint8_t *pim_addr, uint8_t *data, size_t size)
{
	std::cout << "  PIM_RUNTIME\t ReadMem!\n";
	uint64_t strided_size = Ceiling(size, WORD_SIZE * NUM_BANK);
	for (int offset = 0; offset < strided_size; offset += WORD_SIZE)
		TryAddTransaction(pim_addr + offset, data + offset, false);
}

size_t WriteMem(uint8_t *pim_addr, uint8_t *data, size_t size)
{
	std::cout << "  PIM_RUNTIME\t WriteMem!\n";
	uint64_t strided_size = Ceiling(size, WORD_SIZE * NUM_BANK);
	for (int offset = 0; offset < strided_size; offset += WORD_SIZE)
		TryAddTransaction(pim_addr + offset, data + offset, true);
}

size_t ReadReg(PIM_REG pim_reg, uint8_t *data, size_t size)
{
	std::cout << "  PIM_RUNTIME\t ReadReg!\n";
	uint64_t strided_size = Ceiling(size, WORD_SIZE * NUM_BANK);

	switch (pim_reg)
	{
	case (PIM_REG::SRF):
		std::cout << "  MEM RD → SRF \n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_SRF, 0);
			for (int offset = 0; offset < strided_size; offset += WORD_SIZE)
				TryAddTransaction(pim_mem + hex_addr + offset, data_temp_ + offset, false);
		}
		break;
	case (PIM_REG::GRF):
		std::cout << "  MEM RD → GRF\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_GRF, 0);
			for (int offset = 0; offset < strided_size; offset += WORD_SIZE)
				TryAddTransaction(pim_mem + hex_addr + offset, data_temp_ + offset, false);
		}
		break;
	case (PIM_REG::CRF):
		std::cout << "  MEM RD → CRF\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_CRF, 0);
			for (int offset = 0; offset < strided_size; offset += WORD_SIZE)
				TryAddTransaction(pim_mem + hex_addr + offset, data_temp_ + offset, false);
		}
		break;
	case (PIM_REG::SBMR):
		std::cout << "  MEM RD → SRMR\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_SBMR, 0);
			TryAddTransaction(pim_mem + hex_addr, data_temp_, false);
		}
		break;
	case (PIM_REG::ABMR):
		std::cout << "  MEM RD → ABMR\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_ABMR, 0);
			TryAddTransaction(pim_mem + hex_addr, data_temp_, false);
		}
		break;
	case (PIM_REG::PIM_OP_MODE):
		std::cout << "  MEM RD → PIM_OP_MODE\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
			TryAddTransaction(pim_mem + hex_addr, data_temp_, false);
		}
		break;
	}
}

size_t WriteReg(PIM_REG pim_reg, uint8_t *data, size_t size)
{
	std::cout << "  PIM_RUNTIME\t WriteReg!\n";
	uint64_t strided_size = Ceiling(size, WORD_SIZE * NUM_BANK);
	uint64_t step = GetAddress(0, 0, 0, 0, 0, 1);

	switch (pim_reg)
	{
	case (PIM_REG::SRF): // TODO Remove SRF
		std::cout << "  MEM WR → SRF\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_SRF, 0);
			TryAddTransaction(pim_mem + hex_addr, data, true);
		}
		break;
	case (PIM_REG::SRF_A):
		std::cout << "  MEM WR → SRF_A\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_SRF, 0);
			TryAddTransaction(pim_mem + hex_addr, data, true);
		}
		break;
	case (PIM_REG::SRF_M):
		std::cout << "  MEM WR → SRF_M\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_SRF, 0);
			for (int i = 0; i < 16; i++)
				data_temp_[i + 16] = data[i];
			TryAddTransaction(pim_mem + hex_addr, data_temp_, true);
		}
		break;
	case (PIM_REG::GRF):
		std::cout << "  MEM WR → GRF\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_GRF, 0);
			TryAddTransaction(pim_mem + hex_addr, data, true);
		}
		break;
	case (PIM_REG::GRF_A):
		std::cout << "  MEM WR → GRF_A\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_GRF, 0);
			for (int co = 0; co < 8; co++)
				TryAddTransaction(pim_mem + hex_addr + co * step, data + co * step, true);
		}
		break;
	case (PIM_REG::GRF_B):
		std::cout << "  MEM WR → GRF_B\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_GRF, 8);
			for (int co = 0; co < 8; co++)
				TryAddTransaction(pim_mem + hex_addr + co * step, data + co * step, true);
		}
		break;

	case (PIM_REG::CRF):
		std::cout << "  MEM WR → CRF\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr;
			if (ch == 0)
				std::cout << "  CRF-0\n";
			hex_addr = GetAddress(ch, 0, 0, 0, MAP_CRF, 0);
			TryAddTransaction(pim_mem + hex_addr, data, true);
			if (ch == 0)
				std::cout << "  CRF-1\n";
			hex_addr = GetAddress(ch, 0, 0, 0, MAP_CRF, 1);
			TryAddTransaction(pim_mem + hex_addr, data + 32, true);
			if (ch == 0)
				std::cout << "  CRF-2\n";
			hex_addr = GetAddress(ch, 0, 0, 0, MAP_CRF, 2);
			TryAddTransaction(pim_mem + hex_addr, data + 64, true);
			if (ch == 0)
				std::cout << "  CRF-3\n";
			hex_addr = GetAddress(ch, 0, 0, 0, MAP_CRF, 3);
			TryAddTransaction(pim_mem + hex_addr, data + 96, true);
			if (ch == 0)
				std::cout << "  CRF-4\n";
		}
		break;
	case (PIM_REG::SBMR):
		std::cout << "  MEM WR → SBMR\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_SBMR, 0);
			TryAddTransaction(pim_mem + hex_addr, data, true);
		}
		break;
	case (PIM_REG::ABMR):
		std::cout << "  MEM WR → ABMR\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_ABMR, 0);
			TryAddTransaction(pim_mem + hex_addr, data, true);
		}
		break;
	case (PIM_REG::PIM_OP_MODE):
		std::cout << "  MEM WR → PIM_OP_MODE\n";
		for (int ch = 0; ch < NUM_CHANNEL; ch++)
		{
			uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
			TryAddTransaction(pim_mem + hex_addr, data, true);
		}
		break;
	}
}

void ExecuteKernel_1COL(uint8_t *pim_target, bool is_write, int bank)
{
	for (int ch = 0; ch < NUM_CHANNEL; ch++)
	{
		uint64_t hex_addr = GetAddress(ch, 0, 0, bank, 0, 0);
		TryAddTransaction(pim_target + hex_addr, data_temp_ + ch * WORD_SIZE, is_write);
	}
}

void ExecuteKernel_8COL(uint8_t *pim_target, bool is_write, int bank)
{
	for (int ch = 0; ch < NUM_CHANNEL; ch++)
	{
		uint64_t hex_addr = GetAddress(ch, 0, 0, bank, 0, 0);
		uint64_t step = GetAddress(0, 0, 0, 0, 0, 1);
#if 0 // with multi-thread
				thr_grp_param[ch].ch = ch;
				thr_grp_param[ch].pim_addr = pim_target + hex_addr;  // 512 : 2 << co_pos << shift_bits
				thr_grp_param[ch].data = data_temp_;
				thr_grp_param[ch].is_write = is_write;
				pthread_create(&(thr_grp[ch]), NULL, TryThreadGroupAddTransaction, (void*)&thr_grp_param[ch]);
#else // without multi-thread
		if (ComputeMode())
		{
			for (int co_i = 0; co_i < 8; co_i++)
			{
				pim_func_sim->AddTransaction((uint64_t)((pim_target + hex_addr + co_i * step) - pim_mem), data_temp_, is_write);
			}
		}
		else
		{
			for (int co_i = 0; co_i < 8; co_i++)
			{
				TryAddTransaction(pim_target + hex_addr + co_i * step, data_temp_, is_write);
			}
		}
#endif
	}
	for (int ch = 0; ch < NUM_CHANNEL; ch++)
		pthread_join(thr_grp[ch], NULL);
}

bool ExecuteKernel(uint8_t *pim_x, uint8_t *pim_y, uint8_t *pim_z, PIM_CMD pim_cmd, int bank)
{
	if (FpgaMode())
		spend_time_ns = spend_time_ns + (uint64_t)pimExecution((uint32_t)0, data_temp_, 1);

	switch (pim_cmd)
	{
	case (PIM_CMD::WRITE_SRF_INPUT):
		std::cout << "  Execute: WRITE_SRF_INPUT\n";
		WriteReg(PIM_REG::SRF, pim_x, WORD_SIZE);
		break;
	case (PIM_CMD::WRITE_GRF_INPUT):
		std::cout << "  Execute: WRITE_GRF_INPUT\n";
		WriteReg(PIM_REG::GRF, pim_x, WORD_SIZE * 16);
		break;
	case (PIM_CMD::SB_MODE):
		std::cout << "  Execute: SB_MODE\n";
		ReadReg(PIM_REG::SBMR, data_temp_, WORD_SIZE);
		break;
	case (PIM_CMD::AB_MODE):
		std::cout << "  Execute: AB_MODE\n";
		ReadReg(PIM_REG::ABMR, data_temp_, WORD_SIZE);
		break;
	case (PIM_CMD::PIM_OP_MODE):
		std::cout << "  Execute: PIM_OP_MODE\n";
		ReadReg(PIM_REG::PIM_OP_MODE, data_temp_, WORD_SIZE);
		break;
	case (PIM_CMD::READ_INPUT_1COL):
		std::cout << "  Execute: READ_INPUT_1COL\n";
		ExecuteKernel_1COL(pim_x, false, bank);
		break;
	case (PIM_CMD::READ_WEIGHT_1COL):
		std::cout << "  Execute: READ_WEIGHT_1COL\n";
		ExecuteKernel_1COL(pim_y, false, bank);
		break;
	case (PIM_CMD::READ_OUTPUT_1COL):
		std::cout << "  Execute: READ_OUTPUT_1COL\n";
		ExecuteKernel_1COL(pim_z, false, bank);
		break;
	case (PIM_CMD::WRITE_INPUT_1COL):
		std::cout << "  Execute: WRITE_INPUT_1COL\n";
		ExecuteKernel_1COL(pim_x, true, bank);
		break;
	case (PIM_CMD::WRITE_WEIGHT_1COL):
		std::cout << "  Execute: WRITE_WEIGHT_1COL\n";
		ExecuteKernel_1COL(pim_y, true, bank);
		break;
	case (PIM_CMD::WRITE_OUTPUT_1COL):
		std::cout << "  Execute: WRITE_OUTPUT_1COL\n";
		ExecuteKernel_1COL(pim_z, true, bank);
		break;
	case (PIM_CMD::READ_INPUT_8COL):
		std::cout << "  Execute: READ_INPUT_8COL\n";
		ExecuteKernel_8COL(pim_x, false, bank);
		break;
	case (PIM_CMD::READ_WEIGHT_8COL):
		std::cout << "  Execute: READ_WEIGHT_8COL\n";
		ExecuteKernel_8COL(pim_y, false, bank);
		break;
	case (PIM_CMD::READ_OUTPUT_8COL):
		std::cout << "  Execute: READ_OUTPUT_8COL\n";
		ExecuteKernel_8COL(pim_z, false, bank);
		break;
	case (PIM_CMD::WRITE_INPUT_8COL):
		std::cout << "  Execute: WRITE_INPUT_8COL\n";
		ExecuteKernel_8COL(pim_x, true, bank);
		break;
	case (PIM_CMD::WRITE_WEIGHT_8COL):
		std::cout << "  Execute: WRITE_WEIGHT_8COL\n";
		ExecuteKernel_8COL(pim_y, true, bank);
		break;
	case (PIM_CMD::WRITE_OUTPUT_8COL):
		std::cout << "  Execute: WRITE_OUTPUT_8COL\n";
		ExecuteKernel_8COL(pim_z, true, bank);
		break;
	}
	return 1;
}
// Some tool
uint64_t Ceiling(uint64_t num, uint64_t stride)
{
	return ((num + stride - 1) / stride) * stride;
}

static void *TryThreadGroupAddTransaction(void *input_)
{
	thr_param_t *input = (thr_param_t *)input_;
	int ch = input->ch;
	uint8_t *pim_addr = input->pim_addr;
	uint8_t *data = input->data;
	bool is_write = input->is_write;

	pthread_barrier_init(&thr_barrier[ch], NULL, 8 + 1);
	for (int offset = 0; offset < 8; offset++)
	{
		thr_param[ch * 8 + offset].ch = ch;
		thr_param[ch * 8 + offset].pim_addr = pim_addr + 32 * offset * 256; // 512 : 2 << co_pos << shift_bits  // why 256..? i dont know
		thr_param[ch * 8 + offset].data = data_temp_;
		thr_param[ch * 8 + offset].is_write = false;
		pthread_create(&(thr[ch * 8 + offset]), NULL, TryThreadAddTransaction, (void *)&thr_param[ch * 8 + offset]);
	}
	pthread_barrier_wait(&thr_barrier[ch]);
	pthread_barrier_destroy(&thr_barrier[ch]);

	return (NULL);
}

static void *TryThreadAddTransaction(void *input_)
{
	thr_param_t *input = (thr_param_t *)input_;
	int ch = input->ch;
	uint8_t *pim_addr = input->pim_addr;
	uint8_t *data = input->data;
	bool is_write = input->is_write;

	if (is_write)
		std::memcpy(pim_addr, data, burstSize_);
	else
		std::memcpy(data, pim_addr, burstSize_);

	pthread_mutex_lock(&print_mutex);
	int tmp = (is_write) ? 1 : 0;
	std::cout << ">> " << clock_ << "\t" << tmp << "\t addr: " << (uint64_t)(pim_addr - pim_base) << "\n";
	fprintf(fp, ">> %d\t%d\t addr: %llu\n", clock_, tmp, (uint64_t)(pim_addr - pim_base));
	clock_++;
	pthread_mutex_unlock(&print_mutex);

	pthread_barrier_wait(&thr_barrier[ch]);

	return (NULL);
}

uint32_t change(uint64_t tmp)
{
	uint32_t a;
	for (int i = 0; i < 32; i++)
	{
		a = a + tmp % 2;
		tmp = tmp / 2;
	}
}

union tmp_change
{
	struct
	{
		uint64_t : 5;
		uint64_t byte_0 : 32;
		uint64_t byte_1 : 27;
	};
	uint64_t change_body;
};

void TryAddTransaction(uint8_t *pim_addr, uint8_t *data, bool is_write)
{
	if (FpgaMode())
	{
		uint64_t hex_addr = (uint64_t)pim_addr - pim_base;
		Address addr = AddressMapping(hex_addr);
		int CH = addr.channel;
		int BA = addr.bank;
		int RA = addr.row;
		// std::cout << "PIM_Runtime's RA : " << std::hex << RA << std::dec << std::endl;
		tmp_change tc;
		tc.change_body = hex_addr;
		uint32_t tmp = tc.byte_0;
		if (CH == 0 && (BA == 0 || BA == 1))
			spend_time_ns = spend_time_ns + (uint64_t)pimExecution(tmp, data, 1);
	}
	else if (ComputeMode())
		pim_func_sim->AddTransaction((uint64_t)(pim_addr - pim_base), data, is_write);
	else
	{
		if (is_write)
			std::memcpy(pim_addr, data, burstSize_);
		else
			std::memcpy(data, pim_addr, burstSize_);
	}
	int tmp = (is_write) ? 1 : 0;
	// std::cout << ">> " << clock_ << "\t" << tmp << "\t addr: " << (uint64_t)(pim_addr - pim_base) << "\n";
	fprintf(fp, ">> %d\t%d\t addr: %llu\n", clock_, tmp, (uint64_t)(pim_addr - pim_base));
	clock_++;
}

uint64_t GetAddress(int channel, int rank, int bankgroup, int bank, int row, int column)
{
	uint64_t hex_addr = 0;
	hex_addr += ((uint64_t)channel) << ch_pos;
	hex_addr += ((uint64_t)rank) << ra_pos;
	hex_addr += ((uint64_t)bankgroup) << bg_pos;
	hex_addr += ((uint64_t)bank) << ba_pos;
	hex_addr += ((uint64_t)row) << ro_pos;
	hex_addr += ((uint64_t)column) << co_pos;
	return hex_addr << shift_bits;
}

bool DebugMode()
{
#ifdef debug_mode
	return true;
#endif
	return false;
}

bool FpgaMode()
{
#ifdef fpga_mode
	return true;
#endif
	return false;
}

bool ComputeMode()
{
#ifdef compute_mode
	return true;
#endif
	return false;
}

void GetFpgaAddr_1COL(uint8_t *pim_target, bool is_write, int bank)
{
	int ch = 0;
	uint64_t hex_addr = GetAddress(ch, 0, 0, bank, 0, 0);
	uint64_t tmp = (uint64_t)(pim_target + hex_addr);
	PushFpgaAddr(tmp);
}

void GetFpgaAddr_8COL(uint8_t *pim_target, bool is_write, int bank)
{
	int ch = 0;
	uint64_t hex_addr = GetAddress(ch, 0, 0, bank, 0, 0);
	uint64_t step = GetAddress(0, 0, 0, 0, 0, 1);
	for (int co_i = 0; co_i < 8; co_i++)
	{
		uint64_t tmp = hex_addr + co_i * step;
		PushFpgaAddr(tmp);
	}
}

bool GetFpgaAddr(uint8_t *pim_x, uint8_t *pim_y, uint8_t *pim_z, PIM_CMD pim_cmd, int bank)
{
	uint8_t *pim_target;
	bool is_write;
	switch (pim_cmd)
	{
	case (PIM_CMD::READ_INPUT_1COL):
		std::cout << "  Execute: READ_INPUT_1COL\n";
		GetFpgaAddr_1COL(pim_x, false, bank);
		break;
	case (PIM_CMD::READ_WEIGHT_1COL):
		std::cout << "  Execute: READ_WEIGHT_1COL\n";
		GetFpgaAddr_1COL(pim_y, false, bank);
		break;
	case (PIM_CMD::READ_OUTPUT_1COL):
		std::cout << "  Execute: READ_OUTPUT_1COL\n";
		GetFpgaAddr_1COL(pim_z, false, bank);
		break;
	case (PIM_CMD::WRITE_INPUT_1COL):
		std::cout << "  Execute: WRITE_INPUT_1COL\n";
		GetFpgaAddr_1COL(pim_x, true, bank);
		break;
	case (PIM_CMD::WRITE_WEIGHT_1COL):
		std::cout << "  Execute: WRITE_WEIGHT_1COL\n";
		GetFpgaAddr_1COL(pim_y, true, bank);
		break;
	case (PIM_CMD::WRITE_OUTPUT_1COL):
		std::cout << "  Execute: WRITE_OUTPUT_1COL\n";
		GetFpgaAddr_1COL(pim_z, true, bank);
		break;
	case (PIM_CMD::READ_INPUT_8COL):
		std::cout << "  Execute: READ_INPUT_8COL\n";
		GetFpgaAddr_8COL(pim_x, false, bank);
		break;
	case (PIM_CMD::READ_WEIGHT_8COL):
		std::cout << "  Execute: READ_WEIGHT_8COL\n";
		GetFpgaAddr_8COL(pim_y, false, bank);
		break;
	case (PIM_CMD::READ_OUTPUT_8COL):
		std::cout << "  Execute: READ_OUTPUT_8COL\n";
		GetFpgaAddr_8COL(pim_z, false, bank);
		break;
	case (PIM_CMD::WRITE_INPUT_8COL):
		std::cout << "  Execute: WRITE_INPUT_8COL\n";
		GetFpgaAddr_8COL(pim_x, true, bank);
		break;
	case (PIM_CMD::WRITE_WEIGHT_8COL):
		std::cout << "  Execute: WRITE_WEIGHT_8COL\n";
		GetFpgaAddr_8COL(pim_y, true, bank);
		break;
	case (PIM_CMD::WRITE_OUTPUT_8COL):
		std::cout << "  Execute: WRITE_OUTPUT_8COL\n";
		GetFpgaAddr_8COL(pim_z, true, bank);
		break;
	}
	return 1;
}

void PushFpgaAddr(uint64_t addr)
{
	fpga_addr_queue[num_fpga_addr] = (uint32_t)addr;
	num_fpga_addr++;
}

void SetFpgaAddr()
{
	int num_col = (num_fpga_addr + 8 - 1) / 8;
	WriteReg(PIM_REG::ADDR, (uint8_t *)&num_fpga_addr, WORD_SIZE * num_col);
	num_fpga_addr = 0;
}

void InitFpgaTime()
{
	spend_time_ns = 0;
}

void PrintFpgaTime()
{
	std::cout << "Spend Time : " << spend_time_ns << " ns\n";
}
