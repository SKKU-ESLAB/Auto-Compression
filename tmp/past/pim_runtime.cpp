#include "pim_runtime.h"
#include <stdio.h>

uint64_t next_addr = 0;
unsigned int burstSize_ = 32;

int ch_pos_ = 0;
int ba_pos_ = ch_pos_ + 4;
int bg_pos_ = ba_pos_ + 2;
int co_pos_ = bg_pos_ + 2;
int ra_pos_ = co_pos_ + 5;
int ro_pos_ = ra_pos_ + 0;
int shift_bits_ = 5;
uint8_t data_temp_[32];
uint64_t ukernel_access_size_ = SIZE_WORD * 8 * NUM_BANK;
uint64_t ukernel_count_per_pim_;

typedef struct {
	uint64_t addr=0;
	uint64_t len=0;
} PIMAllocList_t;

PIMAllocList_t PIMAllocList[1024];
uint64_t PIMAllocList_idx;

typedef struct {
	int ch;
	uint8_t* pim_mem;
	uint8_t* data;
	bool is_write;
} thr_param_t;

pthread_t thr[NUM_CHANNEL*8];
thr_param_t thr_param[NUM_CHANNEL*8];

pthread_t thr_grp[NUM_CHANNEL];
thr_param_t thr_grp_param[NUM_CHANNEL];

pthread_barrier_t thr_barrier[NUM_CHANNEL];
pthread_mutex_t print_mutex;

static void* TryThreadGroupAddTransaction(void *input_);
static void* TryThreadAddTransaction(void *input_);

FILE* fp = fopen("output.txt", "w");
uint64_t pim_base;
int clock_ = 0;

// PIM Preprocessor
bool isSuitableOps(PIM_OP op) {
	std::cout << "  PIM_RUNTIME\t isSuitableOps!\n";
	
	// For now, just return True
	return true;
}

PIMKernel GetMicrokernelCode(PIM_OP op, PIM_OP_ATTRS op_attrs) {
	std::cout << "  PIM_RUNTIME\t GetMicrokernelCode!\n";
	PIMKernel new_kernel;
	ukernel_count_per_pim_ = Ceiling(op_attrs.len_in * UNIT_SIZE, ukernel_access_size_) / ukernel_access_size_;
	// For now, Make same ukernel without considering op_attrs
	new_kernel.SetMicrokernelCode(op);
	return new_kernel;
}

uint8_t* MapMemory(uint8_t *pim_mem, uint8_t *data, size_t len) {
	std::cout << "  PIM_RUNTIME\t MapMemory!\n";
	uint64_t addr = next_addr;
	uint64_t alloc_len = Ceiling(len * UNIT_SIZE, 8 * WORD_SIZE * NUM_BANK);

	PIMAllocList[PIMAllocList_idx].addr = addr / (8 * WORD_SIZE * NUM_BANK);
	PIMAllocList[PIMAllocList_idx].len = alloc_len / (8 * WORD_SIZE * NUM_BANK);
	PIMAllocList_idx ++;

	WriteMem(pim_mem+addr, data, len);  // Mapping!

	next_addr += alloc_len;
	return (uint8_t*)(pim_mem+addr);
}

// PIM Memory Manager
uint8_t* AllocMem(uint8_t* pim_mem, uint8_t *data, size_t len) {
	std::cout << "  PIM_RUNTIME\t AllocMem!\n";
	uint64_t addr = next_addr;
	uint64_t alloc_len = Ceiling(len * UNIT_SIZE, 8 * WORD_SIZE * NUM_BANK);

	PIMAllocList[PIMAllocList_idx].addr = addr / (8 * WORD_SIZE * NUM_BANK);
	PIMAllocList[PIMAllocList_idx].len = alloc_len / (8 * WORD_SIZE * NUM_BANK);
	PIMAllocList_idx ++;

	next_addr += alloc_len;
	return (uint8_t*)(pim_mem+addr);
}

void FreeMem(uint8_t* pim_mem) {
	std::cout << "  PIM_RUNTIME\t FreeMem!\n";

	for (int i=0; i<PIMAllocList_idx; i++) {
		if (PIMAllocList[i].addr == (uint64_t)(pim_mem-pim_base)) {
			PIMAllocList[i].len = 0;
			return;
		}
	}
	std::cout << "  PIM_RUNTIME\t FreeMem → Not found... What?\n";
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

size_t ReadReg(uint8_t* pim_mem, PIM_REG pim_reg, uint8_t *data, size_t len) {
	std::cout << "  PIM_RUNTIME\t ReadReg!\n";
	uint64_t strided_size = Ceiling(len * UNIT_SIZE, SIZE_WORD * NUM_BANK);

	switch(pim_reg) {
		case (PIM_REG::SRF):
			std::cout << "   MEM RD → SRF \n";
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_SRF, 0);
				for (int offset = 0; offset < strided_size; offset += SIZE_WORD)
					TryAddTransaction(pim_mem + hex_addr + offset, data_temp_ + offset, false);
			}
			break;
		case (PIM_REG::GRF):
			std::cout << "   MEM RD → GRF\n";
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_GRF, 0);
				for (int offset = 0; offset < strided_size; offset += SIZE_WORD)
					TryAddTransaction(pim_mem + hex_addr + offset, data_temp_ + offset, false);
			}
			break;
		case (PIM_REG::CRF):
			std::cout << "   MEM RD → CRF\n";
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_CRF, 0);
				for (int offset = 0; offset < strided_size; offset += SIZE_WORD)
					TryAddTransaction(pim_mem + hex_addr + offset, data_temp_ + offset, false);
			}
			break;
		case (PIM_REG::SBMR):
			std::cout << "   MEM RD → SRMR\n";
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_SBMR, 0);
				TryAddTransaction(pim_mem + hex_addr, data_temp_, false);
			}
			break;
		case (PIM_REG::ABMR):
			std::cout << "   MEM RD → ABMR\n";
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_ABMR, 0);
				TryAddTransaction(pim_mem + hex_addr, data_temp_, false);
			}
			break;
		case (PIM_REG::PIM_OP_MODE):
			std::cout << "   MEM RD → PIM_OP_MODE\n";
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
				TryAddTransaction(pim_mem + hex_addr, data_temp_, false);
			}
			break;
	}
}

size_t WriteReg(uint8_t* pim_mem, PIM_REG pim_reg, uint8_t *data, size_t len) {
	std::cout << "  PIM_RUNTIME\t WriteReg!\n";
	uint64_t strided_size = Ceiling(len * UNIT_SIZE, SIZE_WORD * NUM_BANK);

	switch(pim_reg) {
		case (PIM_REG::SRF):
			std::cout << "   MEM WR → SRF\n";
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_SRF, 0);
				TryAddTransaction(pim_mem + hex_addr, data, true);
			}
			break;
		case (PIM_REG::GRF):
			std::cout << "   MEM WR → GRF\n";
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_GRF, 0);
				TryAddTransaction(pim_mem + hex_addr, data, true);
			}
			break;
		case (PIM_REG::CRF):
			std::cout << "   MEM WR → CRF\n";
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_CRF, 0);
				TryAddTransaction(pim_mem + hex_addr, data, true);
				TryAddTransaction(pim_mem + hex_addr + 32, data + 32, true);
				TryAddTransaction(pim_mem + hex_addr + 64, data + 64, true);
				TryAddTransaction(pim_mem + hex_addr + 96, data + 96, true);
			}
			break;
		case (PIM_REG::SBMR):
			std::cout << "   MEM WR → SBMR\n";
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_SBMR, 0);
				TryAddTransaction(pim_mem + hex_addr, data, true);
			}
			break;
		case (PIM_REG::ABMR):
			std::cout << "   MEM WR → ABMR\n";
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_ABMR, 0);
				TryAddTransaction(pim_mem + hex_addr, data, true);
			}
			break;
		case (PIM_REG::PIM_OP_MODE):
			std::cout << "   MEM WR → PIM_OP_MODE\n";
            for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, MAP_PIM_OP_MODE, 0);
				TryAddTransaction(pim_mem + hex_addr, data, true);
			}
			break;
	}
}

bool ExecuteKernel(uint8_t* pim_mem, uint8_t* pim_x, uint8_t* pim_y, uint8_t* pim_z, PIM_CMD pim_cmd) {
	switch(pim_cmd) {
		case (PIM_CMD::WRITE_SRF_INPUT):
			std::cout << "   Execute: WRITE_SRF_INPUT\n";
			WriteReg(pim_mem, PIM_REG::SRF, pim_x, WORD_SIZE);
			break;
		case (PIM_CMD::WRITE_GRF_INPUT):
			std::cout << "   Execute: WRITE_GRF_INPUT\n";
			WriteReg(pim_mem, PIM_REG::GRF, pim_x, WORD_SIZE*16);
			break;
		case (PIM_CMD::SB_MODE):
			std::cout << "   Execute: SB_MODE\n";
			ReadReg(pim_mem, PIM_REG::SBMR, data_temp_, WORD_SIZE);
			break;
		case (PIM_CMD::AB_MODE):
			std::cout << "   Execute: AB_MODE\n";
			ReadReg(pim_mem, PIM_REG::ABMR, data_temp_, WORD_SIZE);
			break;
		case (PIM_CMD::PIM_OP_MODE):
			std::cout << "   Execute: PIM_OP_MODE\n";
			ReadReg(pim_mem, PIM_REG::PIM_OP_MODE, data_temp_, WORD_SIZE);
			break;
		case (PIM_CMD::READ_INPUT_1COL):
			std::cout << "   Execute: READ_INPUT_1COL\n";
			for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, 0, 0);
				TryAddTransaction(pim_x + hex_addr, data_temp_ + ch * WORD_SIZE, false);
			}
			break;
		case (PIM_CMD::READ_WEIGHT_1COL):
			std::cout << "   Execute: READ_WEIGHT_1COL\n";
			for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, 0, 0);
				TryAddTransaction(pim_y + hex_addr, data_temp_ + ch * WORD_SIZE, false);
			}
			break;
		case (PIM_CMD::READ_WEIGHT_8COL):
			std::cout << "   Execute: READ_WEIGHT_8COL\n";
			for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, 0, 0);
#if 1  // with multi-thread
				thr_grp_param[ch].ch = ch;
				thr_grp_param[ch].pim_mem = pim_y + hex_addr;  // 512 : 2 << co_pos << shift_bits
				thr_grp_param[ch].data = data_temp_;
				thr_grp_param[ch].is_write = false;
				pthread_create(&(thr_grp[ch]), NULL, TryThreadGroupAddTransaction, (void*)&thr_grp_param[ch]);
#else  // without multi-thread
				for (int i=0; i<8; i++)
					TryAddTransaction(pim_y + hex_addr + 32*i*256, data_temp_, false);
#endif
			}
			for (int ch = 0; ch < NUM_CHANNEL; ch++)
				pthread_join(thr_grp[ch], NULL);
			break;
		case (PIM_CMD::READ_OUTPUT_1COL):
			std::cout << "   Execute: READ_OUTPUT_1COL\n";
			for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, 0, 0);
				TryAddTransaction(pim_z + hex_addr, data_temp_ + ch * WORD_SIZE, false);
			}
			break;
		case (PIM_CMD::WRITE_INPUT_1COL):
			std::cout << "   Execute: WRITE_INPUT_1COL\n";
			for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, 0, 0);
				TryAddTransaction(pim_x + hex_addr, data_temp_ + ch * WORD_SIZE, true);
			}
			break;
		case (PIM_CMD::WRITE_WEIGHT_1COL):
			std::cout << "   Execute: WRITE_WEIGHT_1COL\n";
			for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, 0, 0);
				TryAddTransaction(pim_y + hex_addr, data_temp_ + ch * WORD_SIZE, true);
			}
			break;
		case (PIM_CMD::WRITE_OUTPUT_1COL):
			std::cout << "   Execute: WRITE_OUTPUT_1COL\n";
			for (int ch = 0; ch < NUM_CHANNEL; ch++) {
				uint64_t hex_addr = GetAddress(ch, 0, 0, 0, 0, 0);
				TryAddTransaction(pim_z + hex_addr, data_temp_ + ch * WORD_SIZE, true);
			}
			break;
	}
	return 1;
}
// Some tool

void runtime_init(uint64_t pim_base_) {
	pim_base = pim_base_;
	std::cout << "PIM_BASE_ADDR : " << pim_base << "\n";
}

uint64_t Ceiling(uint64_t num, uint64_t stride) {
	return ((num + stride - 1) / stride) * stride;
}

static void* TryThreadGroupAddTransaction(void *input_) {
	thr_param_t *input = (thr_param_t*)input_;
	int ch = input->ch;
	uint8_t* pim_mem = input->pim_mem;
	uint8_t* data = input->data;
	bool is_write = input->is_write;

	pthread_barrier_init(&thr_barrier[ch], NULL, 8+1);
	for (int offset = 0; offset < 8; offset++) {
		thr_param[ch*8+offset].ch = ch;
		thr_param[ch*8+offset].pim_mem = pim_mem + 32*offset*256;  // 512 : 2 << co_pos << shift_bits  // why 256..? i dont know
		thr_param[ch*8+offset].data = data_temp_;
		thr_param[ch*8+offset].is_write = false;
		pthread_create(&(thr[ch*8+offset]), NULL, TryThreadAddTransaction, (void*)&thr_param[ch*8+offset]);	
	}
	pthread_barrier_wait(&thr_barrier[ch]);
	pthread_barrier_destroy(&thr_barrier[ch]);
	
	return (NULL);
}

static void* TryThreadAddTransaction(void *input_) {
	thr_param_t *input = (thr_param_t*)input_;
	int ch = input->ch;
	uint8_t* pim_mem = input->pim_mem;
	uint8_t* data = input->data;
	bool is_write = input->is_write;

	if (is_write)
		std::memcpy(pim_mem, data, burstSize_);
	else
		std::memcpy(data, pim_mem, burstSize_);

	pthread_mutex_lock(&print_mutex);
	int tmp = (is_write)? 1:0;
	std::cout << ">> " << clock_ << "\t" << tmp << "\t addr: " << (uint64_t)(pim_mem-pim_base) << "\n";
	fprintf(fp, ">> %d\t%d\t addr: %llu\n", clock_, tmp, (uint64_t)(pim_mem-pim_base));
	clock_ ++;
	pthread_mutex_unlock(&print_mutex);
	
	pthread_barrier_wait(&thr_barrier[ch]);
	
	return (NULL);
}

void TryAddTransaction(uint8_t* pim_mem, uint8_t* data, bool is_write) {
	if (is_write)
		std::memcpy(pim_mem, data, burstSize_);
	else
		std::memcpy(data, pim_mem, burstSize_);
	int tmp = (is_write)? 1:0;
	std::cout << ">> " << clock_ << "\t" << tmp << "\t addr: " << (uint64_t)(pim_mem-pim_base) << "\n";
	fprintf(fp, ">> %d\t%d\t addr: %llu\n", clock_, tmp, (uint64_t)(pim_mem-pim_base));
	clock_ ++;
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
