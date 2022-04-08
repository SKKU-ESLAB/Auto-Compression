#include <iostream>
#include "pim_blas.h"
using half_float::half;
typedef unsigned short uint16;

//int fd = open("/dev/PIM", O_RDWR|O_SYNC);
//int fd = open("./PIM", O_RDWR|O_SYNC);
//uint8_t* pim_mem = (uint8_t*)mmap(NULL, LEN_PIM, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
uint8_t* pim_mem = (uint8_t*)malloc(LEN_PIM);
std::random_device random_device;
auto rng = std::mt19937(random_device());
auto f32rng = std::bind(std::normal_distribution<float>(0, 1), std::ref(rng));

int main() {
	int n = 4096;
	uint8_t *x = (uint8_t *)malloc(sizeof(uint16_t)*n);
	uint8_t *y = (uint8_t *)malloc(sizeof(uint16_t)*n);
	uint8_t *z = (uint8_t *)malloc(sizeof(uint16_t)*n);

	for (int i=0; i<n; i++) {
		half h_x = half(f32rng());
		half h_y = half(f32rng());
		((uint16_t*)x)[i] = *reinterpret_cast<uint16_t*>(&h_x);
		((uint16_t*)y)[i] = *reinterpret_cast<uint16_t*>(&h_y);
	}

	std::cout << "///// Testing... /////\n";
	pim_add(pim_mem, n, x, y, z);

	std::cout << "///// Test Ended!! /////\n";
	return 0;
}
