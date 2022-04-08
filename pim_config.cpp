#include "pim_config.h"

//std::random_device random_device;
//auto rng = std::mt19937(random_device());
//auto f32rng = std::bind(std::normal_distribution<float>(0, 1), std::ref(rng));
//int fd_ = open("/dev/PIM", O_RDWR|O_SYNC);
//uint8_t* pmemAddr_ = (uint9_t*)mmap(NULL, LEN_PIM, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);

void PIM_OP_ATTRS::ADD(uint8_t* pim_x, uint8_t* pim_y, uint8_t* pim_z, int len) {
	std::cout << "PIM_OP_ATTRS::ADD!\n";
}

