#ifndef __PIM_BLAS_H_
#define __PIM_BLAS_H_

#include "pim_runtime.h"
#include "pim_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void blas_init(uint64_t num);
bool pimblasAddPreprocess(PIMKernel *micro_kernel, int len, uint8_t **in0, uint8_t **in1);
bool pimblasMulPreprocess(PIMKernel *micro_kernel, int len, uint8_t **in0, uint8_t **in1);
bool pimblasGemvPreprocess(PIMKernel *micro_kernel, int len_in, int len_out, uint8_t **w);
bool pimblasLstmPreprocess(PIMKernel *micro_kernel, int len_in, int len_out, uint8_t **w, uint8_t **b);
bool pimblasBn1dPreprocess(PIMKernel *micro_kernel, int len_batch, int len_feature, uint8_t **weight_mul,
						  uint8_t **weight_add);
bool pim_add(PIMKernel micro_kernel, int len, uint8_t *in0, uint8_t *in1, uint8_t *out);
bool pim_mul(PIMKernel micro_kernel, int len, uint8_t *in0, uint8_t *in1, uint8_t *out);
bool pim_bn1d(PIMKernel micro_kernel, int len_batch, int len_feature, uint8_t *in, uint8_t *weight_mul,
			  uint8_t *weight_add, uint8_t *out);
bool pim_gemv(PIMKernel micro_kernel, int m, int n, uint8_t *x, uint8_t *y, uint8_t *z);
bool pim_lstm(PIMKernel micro_kernel, int m, int n, uint8_t *in, uint8_t *w, uint8_t *b, uint8_t *out);

uint8_t *Bn1dReshape(uint8_t *w, int l, int f);
uint8_t *GemvReshape(uint8_t *w, int m, int n);
uint8_t *LstmReshape(uint8_t *w, int m, int n);
uint8_t *Transpose(uint8_t *w, int m, int n);

bool C_pimblasAddPreprocess(int len, uint8_t **in0, uint8_t **in1);
bool C_pimblasMulPreprocess(int len, uint8_t **in0, uint8_t **in1);
bool C_pimblasGemvPreprocess(int len_in, int len_out, uint8_t **w);
bool C_pim_add(int len, uint8_t *in0, uint8_t *in1, uint8_t *out);
bool C_pim_mul(int len, uint8_t *in0, uint8_t *in1, uint8_t *out);
bool C_pim_gemv(int len_in, int len_out, uint8_t *in, uint8_t *w, uint8_t *out);

#ifdef __cplusplus
}
#endif

#endif // __PIM_BLAS_H_
