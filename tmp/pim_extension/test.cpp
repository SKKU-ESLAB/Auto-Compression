#include <iostream>
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
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			// w : [m, n]
			// w[i][j] = w[i*n + j]
			// w_ : [n, m]
			// w_[j][i] = w_[j*m + i]
			((uint16_t *)w_)[j * m + i] = ((uint16_t *)w)[i * n + j];
		}
	}
	w = w_;
}

void test_gemv_blas()
{
	std::cout << "LEN_PIM: " << LEN_PIM << std::endl;
	int m = 32;
	int n = 4096;
	uint8_t *in = (uint8_t *)malloc(sizeof(uint16_t) * m);
	uint8_t *w = (uint8_t *)malloc(sizeof(uint16_t) * m * n);
	uint8_t *out = (uint8_t *)malloc(sizeof(uint16_t) * n);

	for (int i = 0; i < m; i++)
	{
		half h_in = half(f32rng());
		for (int j = 0; j < n; j++)
		{
			half h_w = half(f32rng());
			((uint16_t *)w)[i * n + j] = *reinterpret_cast<uint16_t *>(&h_w);
		}
		((uint16_t *)in)[i] = *reinterpret_cast<uint16_t *>(&h_in);
	}

	std::cout << "///// Preprocessing GEMV BLAS... /////\n";
	PIM_OP pim_op = PIM_OP::GEMV;
	PIM_OP_ATTRS gemv_attrs = PIM_OP_ATTRS();
	gemv_attrs.GEMV(m, n);
	PIMKernel micro_kernel = GetMicrokernelCode(pim_op, gemv_attrs);

	if (micro_kernel.layout == 1)
		transpose(w, m, n);

	w = MapMemory(w, m * n);

	std::cout << "///// Testing GEMV BLAS... /////\n";
	pim_gemv(micro_kernel, m, n, in, w, out);

	std::cout << "///// Test GEMV BLAS Ended!! /////\n";
	return;
}

int main(int argc, char **argv)
{
	init();
	test_gemv_blas();
	return 0;
}
