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

void test_add_blas() {
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

	std::cout << "///// Testing ADD BLAS... /////\n";
	pim_add(pim_mem, n, x, y, z);

	std::cout << "///// Test ADD BLAS Ended!! /////\n";
	return;
}

void test_mul_blas() {
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

	std::cout << "///// Testing MUL BLAS... /////\n";
	pim_mul(pim_mem, n, x, y, z);

	std::cout << "///// Test MUL BLAS Ended!! /////\n";
	return;
}

void test_bn_blas() {
	int l = 8;
	int f = 2048;
	uint8_t *x = (uint8_t *)malloc(sizeof(uint16_t)*l*f);  // input
	uint8_t *y = (uint8_t *)malloc(sizeof(uint16_t)*f);    // sig
	uint8_t *z = (uint8_t *)malloc(sizeof(uint16_t)*f);    // mu

	for (int i=0; i<f; i++) {
		half h_y = half(f32rng());
		half h_z = half(f32rng());
		((uint16_t*)y)[i] = *reinterpret_cast<uint16_t*>(&h_y);
		((uint16_t*)z)[i] = *reinterpret_cast<uint16_t*>(&h_z);
		for (int j=0; j<l; j++) {
			half h_x = half(f32rng());
			((uint16_t*)x)[j*f+i] = *reinterpret_cast<uint16_t*>(&h_x);
		}
	}

	std::cout << "///// Testing BN BLAS... /////\n";
	pim_bn(pim_mem, l, f, x, y, z);

	std::cout << "///// Test BN BLAS Ended!! /////\n";

	return;
}
void test_gemv_blas() {
	int m = 4096;
	int n = 32;
	uint8_t *x = (uint8_t *)malloc(sizeof(uint16_t)*m);
	uint8_t *y = (uint8_t *)malloc(sizeof(uint16_t)*m*n);
	uint8_t *z = (uint8_t *)malloc(sizeof(uint16_t)*n);

	for (int i=0; i<m; i++) {
		half h_x = half(f32rng());
		for (int j=0; j<n; j++) {
			half h_y = half(f32rng());
			((uint16_t*)y)[i*n + j] = *reinterpret_cast<uint16_t*>(&h_y);
		}
		((uint16_t*)x)[i] = *reinterpret_cast<uint16_t*>(&h_x);
	}

	std::cout << "///// Testing GEMV BLAS... /////\n";
	pim_gemv(pim_mem, m, n, x, y, z);

	std::cout << "///// Test GEMV BLAS Ended!! /////\n";
	return;
}
void test_lstm_blas() {return;}

int main(int argc, char **argv) {
	int option;
	if (argc >= 2)
		option= atoi(argv[1]);
	else {
		std::cout << "ALL: 0, ADD: 1, MUL: 2, BN: 3, GEMV: 4, LSTM: 5\nEnter: ";
		std::cin >> option;
	}

	if(option == 0) {
		test_add_blas();
		test_mul_blas();
		test_bn_blas();
		test_gemv_blas();
		test_lstm_blas();
	}
	else if(option == 1)
		test_add_blas();
	else if(option == 2)
		test_mul_blas();
	else if(option == 3)
		test_bn_blas();
	else if(option == 4)
		test_gemv_blas();
	else if(option == 5)
		test_lstm_blas();
	else
		std::cout << "ERROR: wrong number\n";

	return 0;
}
