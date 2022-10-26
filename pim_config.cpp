#include "pim_config.h"

// std::random_device random_device;
// auto rng = std::mt19937(random_device());
// auto f32rng = std::bind(std::normal_distribution<float>(0, 1), std::ref(rng));
// int fd_ = open("/dev/PIM", O_RDWR|O_SYNC);
// uint8_t* pmemAddr_ = (uint8_t*)mmap(NULL, LEN_PIM, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);

int LogBase2(int power_of_two)
{
	int i = 0;
	while (power_of_two > 1)
	{
		power_of_two /= 2;
		i++;
	}
	return i;
}

Address AddressMapping(uint64_t hex_addr)
{
	hex_addr >>= shift_bits;
	int channel = (hex_addr >> ch_pos) & ch_mask;
	int rank = (hex_addr >> ra_pos) & ra_mask;
	int bg = (hex_addr >> bg_pos) & bg_mask;
	int ba = (hex_addr >> ba_pos) & ba_mask;
	int ro = (hex_addr >> ro_pos) & ro_mask;
	int co = (hex_addr >> co_pos) & co_mask;
	return Address(channel, rank, bg, ba, ro, co);
}

void PIM_OP_ATTRS::ADD(int len)
{
	if (DebugMode())
		std::cout << "  PIM_RUNTIME\t PIM_OP_ATTRS::ADD!\n";
	// 256 elements for each ukernel
	// total 128 PUs = NUM_BANK / 2
	pim_op = PIM_OP::ADD;
	len_in = len;
	code_iter = 2;
	code0_iter = (len + 32768 - 1) / 32768; // 32768 = 8*WORD_SIZE*NUM_BANK
	code1_iter = 0;
}

void PIM_OP_ATTRS::MUL(int len)
{
	if (DebugMode())
		std::cout << "  PIM_RUNTIME\t PIM_OP_ATTRS::MUL!\n";
	// 256 elements for each ukernel
	// total 128 PUs = NUM_BANK / 2
	pim_op = PIM_OP::MUL;
	len_in = len;
	code_iter = 2;
	code0_iter = (len + 32768 - 1) / 32768; // 32768 = 8*WORD_SIZE*NUM_BANK
	code1_iter = 0;
}

void PIM_OP_ATTRS::BN(uint8_t *pim_x, uint8_t *pim_y, uint8_t *pim_z, int len)
{
	if (DebugMode())
		std::cout << "  PIM_RUNTIME\t PIM_OP_ATTRS::BN!\n";
	len_in = len;
}

void PIM_OP_ATTRS::GEMV(int len_in_, int len_out_)
{
	if (DebugMode())
		std::cout << "  PIM_RUNTIME\t PIM_OP_ATTRS::GEMV!\n";
	len_in = len_in_;
	len_out = len_out_;
	code_iter = 2 * ((len_out_ + 4096 - 1) / 4096);
	code0_iter = (len_in_ + 8 - 1) / 8;
	// code0_iter = 1; // just for quick testing
	code1_iter = 1;
}

void PIM_OP_ATTRS::LSTM(uint8_t *pim_x, uint8_t *pim_y, uint8_t *pim_z, int len)
{
	if (DebugMode())
		std::cout << "  PIM_RUNTIME\t PIM_OP_ATTRS::LSTM!\n";
	len_in = len;
}

PIMKernel CPIMKernel::get_micro_kernel() {
	return micro_kernel;
}

void CPIMKernel::init_micro_kernel() {
	micro_kernel = PIMKernel();
}

void* CPIMKernel_getInstance() {
	CPIMKernel *c_micro_kernel = new CPIMKernel;
	c_micro_kernel->init_micro_kernel();
	return (void*)c_micro_kernel;
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
