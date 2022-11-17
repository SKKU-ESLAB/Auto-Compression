#include "pim_config.h"

//std::random_device random_device;
//auto rng = std::mt19937(random_device());
//auto f32rng = std::bind(std::normal_distribution<float>(0, 1), std::ref(rng));
//int fd_ = open("/dev/PIM", O_RDWR|O_SYNC);
//uint8_t* pmemAddr_ = (uint9_t*)mmap(NULL, LEN_PIM, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);

void PIM_OP_ATTRS::ADD(uint8_t* pim_x, uint8_t* pim_y, uint8_t* pim_z, int len) {
	std::cout << "  PIM_RUNTIME\t PIM_OP_ATTRS::ADD!\n";
	pim_op = PIM_OP::ADD;
	len_in = len;
}

void PIM_OP_ATTRS::MUL(uint8_t* pim_x, uint8_t* pim_y, uint8_t* pim_z, int len) {
	std::cout << "  PIM_RUNTIME\t PIM_OP_ATTRS::MUL!\n";
	len_in = len;
}

void PIM_OP_ATTRS::BN(uint8_t* pim_x, uint8_t* pim_y, uint8_t* pim_z, int len) {
	std::cout << "  PIM_RUNTIME\t PIM_OP_ATTRS::BN!\n";
	len_in = len;
}

void PIM_OP_ATTRS::GEMV(uint8_t* pim_y, uint8_t* pim_z, int len_in_, int len_out_) {
	std::cout << "  PIM_RUNTIME\t PIM_OP_ATTRS::GEMV!\n";
	len_in = len_in_;
	len_out = len_out_;
	code_iter = (len_out_+4096-1)/4096;
	code0_iter = len_in_ / 8;
	//code0_iter = 1; // just for quick testing
	code1_iter = 1;
}

void PIM_OP_ATTRS::LSTM(uint8_t* pim_x, uint8_t* pim_y, uint8_t* pim_z, int len) {
	std::cout << "  PIM_RUNTIME\t PIM_OP_ATTRS::LSTM!\n";
	len_in = len;
}

