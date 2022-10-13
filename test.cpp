#include <iostream>
#include <cstring>
#include "pim_blas.h"
using half_float::half;
typedef unsigned short uint16;

std::random_device random_device;
auto rng = std::mt19937(random_device());
auto f32rng = std::bind(std::normal_distribution<float>(0, 1), std::ref(rng));

void init()
{
	blas_init(0);
}

void transpose(uint8_t *w, int m, int n)
{
	uint8_t *w_ = (uint8_t *)malloc(sizeof(uint16_t) * m * n);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			((uint16_t *)w_)[j * n + i] = ((uint16_t *)w)[i * m + j];
		}
	}
	w = w_;
}

void test_add_blas()
{
	std::cout << "LEN_PIM: " << LEN_PIM << std::endl;
	int n = 4096 * 2;
	uint8_t *in0 = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *in1 = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *out = (uint8_t *)malloc(sizeof(uint16_t) * n);

	for (int i = 0; i < n; i++)
	{
		((uint16_t *)in0)[i] = i;
		((uint16_t *)in1)[i] = 1;
	}

	std::cout << "///// Preprocessing ADD BLAS... /////\n";
	PIM_OP pim_op = PIM_OP::ADD;
	PIM_OP_ATTRS add_attrs = PIM_OP_ATTRS();
	add_attrs.ADD(n);
	PIMKernel micro_kernel = GetMicrokernelCode(pim_op, add_attrs);
	in0 = MapMemory(in0, n * UNIT_SIZE);
	in1 = MapMemory(in1, n * UNIT_SIZE);

	std::cout << "///// Testing ADD BLAS... /////\n";
	pim_add(micro_kernel, n, in0, in1, out);

	std::cout << "///// Test ADD BLAS Ended!! /////\n";

	for (int i = 0; i < n; i++)
		std::cout << ((uint16_t *)out)[i] << " ";
	std::cout << std::endl;
	return;
}

void test_mul_blas()
{
	std::cout << "LEN_PIM: " << LEN_PIM << std::endl;
	int n = 4096 * 2;
	uint8_t *in0 = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *in1 = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *out = (uint8_t *)malloc(sizeof(uint16_t) * n);

	for (int i = 0; i < n; i++)
	{
		((uint16_t *)in0)[i] = i;
		((uint16_t *)in1)[i] = 2;
	}

	std::cout << "///// Preprocessing MUL BLAS... /////\n";
	PIM_OP pim_op = PIM_OP::MUL;
	PIM_OP_ATTRS mul_attrs = PIM_OP_ATTRS();
	mul_attrs.MUL(n);
	PIMKernel micro_kernel = GetMicrokernelCode(pim_op, mul_attrs);
	in0 = MapMemory(in0, n * UNIT_SIZE);
	in1 = MapMemory(in1, n * UNIT_SIZE);

	std::cout << "///// Testing MUL BLAS... /////\n";
	pim_mul(micro_kernel, n, in0, in1, out);

	std::cout << "///// Test MUL BLAS Ended!! /////\n";

	for (int i = 0; i < n; i++)
		std::cout << ((uint16_t *)out)[i] << " ";
	std::cout << std::endl;
	return;
}

void test_mac_blas()
{
	std::cout << "LEN_PIM: " << LEN_PIM << std::endl;
}

void test_bn_blas()
{
	std::cout << "LEN_PIM: " << LEN_PIM << std::endl;
}

void test_gemv_blas()
{
	std::cout << "LEN_PIM: " << LEN_PIM << std::endl;
	int m = 8;
	int n = 4096;
	uint8_t *in = (uint8_t *)malloc(sizeof(uint16_t) * m);
	uint8_t *w = (uint8_t *)malloc(sizeof(uint16_t) * m * n);
	uint8_t *out = (uint8_t *)malloc(sizeof(uint16_t) * n);

	for (int j = 0; j < m; j++)
	{
		for (int i = 0; i < n; i++)
		{
			((uint16_t *)w)[i * m + j] = 2;
		}
		((uint16_t *)in)[j] = 1;
	}

	std::cout << "///// Preprocessing GEMV BLAS... /////\n";
	PIM_OP pim_op = PIM_OP::GEMV;
	PIM_OP_ATTRS gemv_attrs = PIM_OP_ATTRS();
	gemv_attrs.GEMV(m, n);
	PIMKernel micro_kernel = GetMicrokernelCode(pim_op, gemv_attrs);

	if (micro_kernel.layout == 1)
		transpose(w, m, n);

	w = MapMemory(w, m * n * UNIT_SIZE);

	std::cout << "///// Testing GEMV BLAS... /////\n";
	pim_gemv(micro_kernel, m, n, in, w, out);

	std::cout << "///// Test GEMV BLAS Ended!! /////\n";

	for (int i = 0; i < n; i++)
		std::cout << (int)((uint16_t *)out)[i] << " ";
	std::cout << std::endl;
	return;
}
void test_lstm_blas()
{
	std::cout << "LEN_PIM: " << LEN_PIM << std::endl;
}

int main(int argc, char **argv)
{
	init();

	if (argc <= 1)
	{
		std::cout << "add, mul, mac, bn, gemv, lstm\n";
		return -1;
	}

	if (std::string(argv[1]) == "add")
		test_add_blas();
	else if (std::string(argv[1]) == "mul")
		test_mul_blas();
	else if (std::string(argv[1]) == "mac")
		test_mac_blas();
	else if (std::string(argv[1]) == "bn")
		test_bn_blas();
	else if (std::string(argv[1]) == "gemv")
		test_gemv_blas();
	else if (std::string(argv[1]) == "lstm")
		test_lstm_blas();
	else
		std::cout << "error... not supported\n";
	return 0;
}
