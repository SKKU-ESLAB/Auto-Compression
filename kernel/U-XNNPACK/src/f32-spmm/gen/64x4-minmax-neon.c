// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/neon-blocked.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/spmm.h>


void xnn_f32_spmm_minmax_ukernel_64x4__neon(
    size_t mc,
    size_t nc,
    const float*restrict input,
    const float*restrict weights,
    const int32_t*restrict widx_dmap,
    const uint32_t*restrict nidx_nnzmap,
    float*restrict output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
  size_t output_decrement = output_stride * nc - 64 * sizeof(float);
  while XNN_LIKELY(mc >= 64 * sizeof(float)) {
    const float*restrict w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    while (n >= 4) {
      uint32_t nnz = *nnzmap++;
      float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n0 = vacc0123n0;
      float32x4_t vacc89ABn0 = vacc0123n0;
      float32x4_t vaccCDEFn0 = vacc0123n0;
      float32x4_t vaccGHIJn0 = vacc0123n0;
      float32x4_t vaccKLMNn0 = vacc0123n0;
      float32x4_t vaccOPQRn0 = vacc0123n0;
      float32x4_t vaccSTUVn0 = vacc0123n0;
      float32x4_t vaccWXYZn0 = vacc0123n0;
      float32x4_t vaccabcdn0 = vacc0123n0;
      float32x4_t vaccefghn0 = vacc0123n0;
      float32x4_t vaccijkln0 = vacc0123n0;
      float32x4_t vaccmnopn0 = vacc0123n0;
      float32x4_t vaccqrstn0 = vacc0123n0;
      float32x4_t vaccuvwxn0 = vacc0123n0;
      float32x4_t vaccyz01n0 = vacc0123n0;
      float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n1 = vacc0123n1;
      float32x4_t vacc89ABn1 = vacc0123n1;
      float32x4_t vaccCDEFn1 = vacc0123n1;
      float32x4_t vaccGHIJn1 = vacc0123n1;
      float32x4_t vaccKLMNn1 = vacc0123n1;
      float32x4_t vaccOPQRn1 = vacc0123n1;
      float32x4_t vaccSTUVn1 = vacc0123n1;
      float32x4_t vaccWXYZn1 = vacc0123n1;
      float32x4_t vaccabcdn1 = vacc0123n1;
      float32x4_t vaccefghn1 = vacc0123n1;
      float32x4_t vaccijkln1 = vacc0123n1;
      float32x4_t vaccmnopn1 = vacc0123n1;
      float32x4_t vaccqrstn1 = vacc0123n1;
      float32x4_t vaccuvwxn1 = vacc0123n1;
      float32x4_t vaccyz01n1 = vacc0123n1;
      float32x4_t vacc0123n2 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n2 = vacc0123n2;
      float32x4_t vacc89ABn2 = vacc0123n2;
      float32x4_t vaccCDEFn2 = vacc0123n2;
      float32x4_t vaccGHIJn2 = vacc0123n2;
      float32x4_t vaccKLMNn2 = vacc0123n2;
      float32x4_t vaccOPQRn2 = vacc0123n2;
      float32x4_t vaccSTUVn2 = vacc0123n2;
      float32x4_t vaccWXYZn2 = vacc0123n2;
      float32x4_t vaccabcdn2 = vacc0123n2;
      float32x4_t vaccefghn2 = vacc0123n2;
      float32x4_t vaccijkln2 = vacc0123n2;
      float32x4_t vaccmnopn2 = vacc0123n2;
      float32x4_t vaccqrstn2 = vacc0123n2;
      float32x4_t vaccuvwxn2 = vacc0123n2;
      float32x4_t vaccyz01n2 = vacc0123n2;
      float32x4_t vacc0123n3 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n3 = vacc0123n3;
      float32x4_t vacc89ABn3 = vacc0123n3;
      float32x4_t vaccCDEFn3 = vacc0123n3;
      float32x4_t vaccGHIJn3 = vacc0123n3;
      float32x4_t vaccKLMNn3 = vacc0123n3;
      float32x4_t vaccOPQRn3 = vacc0123n3;
      float32x4_t vaccSTUVn3 = vacc0123n3;
      float32x4_t vaccWXYZn3 = vacc0123n3;
      float32x4_t vaccabcdn3 = vacc0123n3;
      float32x4_t vaccefghn3 = vacc0123n3;
      float32x4_t vaccijkln3 = vacc0123n3;
      float32x4_t vaccmnopn3 = vacc0123n3;
      float32x4_t vaccqrstn3 = vacc0123n3;
      float32x4_t vaccuvwxn3 = vacc0123n3;
      float32x4_t vaccyz01n3 = vacc0123n3;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float32x4_t vi0123 = vld1q_f32(input);
          const float32x4_t vi4567 = vld1q_f32(input + 4);
          const float32x4_t vi89AB = vld1q_f32(input + 8);
          const float32x4_t viCDEF = vld1q_f32(input + 12);
          const float32x4_t viGHIJ = vld1q_f32(input + 16);
          const float32x4_t viKLMN = vld1q_f32(input + 20);
          const float32x4_t viOPQR = vld1q_f32(input + 24);
          const float32x4_t viSTUV = vld1q_f32(input + 28);
          const float32x4_t viWXYZ = vld1q_f32(input + 32);
          const float32x4_t viabcd = vld1q_f32(input + 36);
          const float32x4_t viefgh = vld1q_f32(input + 40);
          const float32x4_t viijkl = vld1q_f32(input + 44);
          const float32x4_t vimnop = vld1q_f32(input + 48);
          const float32x4_t viqrst = vld1q_f32(input + 52);
          const float32x4_t viuvwx = vld1q_f32(input + 56);
          const float32x4_t viyz01 = vld1q_f32(input + 60);
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          __builtin_prefetch(input + 16);
          __builtin_prefetch(input + 32);
          __builtin_prefetch(input + 48);
          __builtin_prefetch(input + 64);
          const float32x2_t vw01 = vld1_f32(w); w += 2;
          const float32x2_t vw23 = vld1_f32(w); w += 2;
          __builtin_prefetch(w + 32);
          vacc0123n0 = vmlaq_lane_f32(vacc0123n0, vi0123, vw01, 0);
          vacc4567n0 = vmlaq_lane_f32(vacc4567n0, vi4567, vw01, 0);
          vacc89ABn0 = vmlaq_lane_f32(vacc89ABn0, vi89AB, vw01, 0);
          vaccCDEFn0 = vmlaq_lane_f32(vaccCDEFn0, viCDEF, vw01, 0);
          vaccGHIJn0 = vmlaq_lane_f32(vaccGHIJn0, viGHIJ, vw01, 0);
          vaccKLMNn0 = vmlaq_lane_f32(vaccKLMNn0, viKLMN, vw01, 0);
          vaccOPQRn0 = vmlaq_lane_f32(vaccOPQRn0, viOPQR, vw01, 0);
          vaccSTUVn0 = vmlaq_lane_f32(vaccSTUVn0, viSTUV, vw01, 0);
          vaccWXYZn0 = vmlaq_lane_f32(vaccWXYZn0, viWXYZ, vw01, 0);
          vaccabcdn0 = vmlaq_lane_f32(vaccabcdn0, viabcd, vw01, 0);
          vaccefghn0 = vmlaq_lane_f32(vaccefghn0, viefgh, vw01, 0);
          vaccijkln0 = vmlaq_lane_f32(vaccijkln0, viijkl, vw01, 0);
          vaccmnopn0 = vmlaq_lane_f32(vaccmnopn0, vimnop, vw01, 0);
          vaccqrstn0 = vmlaq_lane_f32(vaccqrstn0, viqrst, vw01, 0);
          vaccuvwxn0 = vmlaq_lane_f32(vaccuvwxn0, viuvwx, vw01, 0);
          vaccyz01n0 = vmlaq_lane_f32(vaccyz01n0, viyz01, vw01, 0);
          vacc0123n1 = vmlaq_lane_f32(vacc0123n1, vi0123, vw01, 1);
          vacc4567n1 = vmlaq_lane_f32(vacc4567n1, vi4567, vw01, 1);
          vacc89ABn1 = vmlaq_lane_f32(vacc89ABn1, vi89AB, vw01, 1);
          vaccCDEFn1 = vmlaq_lane_f32(vaccCDEFn1, viCDEF, vw01, 1);
          vaccGHIJn1 = vmlaq_lane_f32(vaccGHIJn1, viGHIJ, vw01, 1);
          vaccKLMNn1 = vmlaq_lane_f32(vaccKLMNn1, viKLMN, vw01, 1);
          vaccOPQRn1 = vmlaq_lane_f32(vaccOPQRn1, viOPQR, vw01, 1);
          vaccSTUVn1 = vmlaq_lane_f32(vaccSTUVn1, viSTUV, vw01, 1);
          vaccWXYZn1 = vmlaq_lane_f32(vaccWXYZn1, viWXYZ, vw01, 1);
          vaccabcdn1 = vmlaq_lane_f32(vaccabcdn1, viabcd, vw01, 1);
          vaccefghn1 = vmlaq_lane_f32(vaccefghn1, viefgh, vw01, 1);
          vaccijkln1 = vmlaq_lane_f32(vaccijkln1, viijkl, vw01, 1);
          vaccmnopn1 = vmlaq_lane_f32(vaccmnopn1, vimnop, vw01, 1);
          vaccqrstn1 = vmlaq_lane_f32(vaccqrstn1, viqrst, vw01, 1);
          vaccuvwxn1 = vmlaq_lane_f32(vaccuvwxn1, viuvwx, vw01, 1);
          vaccyz01n1 = vmlaq_lane_f32(vaccyz01n1, viyz01, vw01, 1);
          vacc0123n2 = vmlaq_lane_f32(vacc0123n2, vi0123, vw23, 0);
          vacc4567n2 = vmlaq_lane_f32(vacc4567n2, vi4567, vw23, 0);
          vacc89ABn2 = vmlaq_lane_f32(vacc89ABn2, vi89AB, vw23, 0);
          vaccCDEFn2 = vmlaq_lane_f32(vaccCDEFn2, viCDEF, vw23, 0);
          vaccGHIJn2 = vmlaq_lane_f32(vaccGHIJn2, viGHIJ, vw23, 0);
          vaccKLMNn2 = vmlaq_lane_f32(vaccKLMNn2, viKLMN, vw23, 0);
          vaccOPQRn2 = vmlaq_lane_f32(vaccOPQRn2, viOPQR, vw23, 0);
          vaccSTUVn2 = vmlaq_lane_f32(vaccSTUVn2, viSTUV, vw23, 0);
          vaccWXYZn2 = vmlaq_lane_f32(vaccWXYZn2, viWXYZ, vw23, 0);
          vaccabcdn2 = vmlaq_lane_f32(vaccabcdn2, viabcd, vw23, 0);
          vaccefghn2 = vmlaq_lane_f32(vaccefghn2, viefgh, vw23, 0);
          vaccijkln2 = vmlaq_lane_f32(vaccijkln2, viijkl, vw23, 0);
          vaccmnopn2 = vmlaq_lane_f32(vaccmnopn2, vimnop, vw23, 0);
          vaccqrstn2 = vmlaq_lane_f32(vaccqrstn2, viqrst, vw23, 0);
          vaccuvwxn2 = vmlaq_lane_f32(vaccuvwxn2, viuvwx, vw23, 0);
          vaccyz01n2 = vmlaq_lane_f32(vaccyz01n2, viyz01, vw23, 0);
          vacc0123n3 = vmlaq_lane_f32(vacc0123n3, vi0123, vw23, 1);
          vacc4567n3 = vmlaq_lane_f32(vacc4567n3, vi4567, vw23, 1);
          vacc89ABn3 = vmlaq_lane_f32(vacc89ABn3, vi89AB, vw23, 1);
          vaccCDEFn3 = vmlaq_lane_f32(vaccCDEFn3, viCDEF, vw23, 1);
          vaccGHIJn3 = vmlaq_lane_f32(vaccGHIJn3, viGHIJ, vw23, 1);
          vaccKLMNn3 = vmlaq_lane_f32(vaccKLMNn3, viKLMN, vw23, 1);
          vaccOPQRn3 = vmlaq_lane_f32(vaccOPQRn3, viOPQR, vw23, 1);
          vaccSTUVn3 = vmlaq_lane_f32(vaccSTUVn3, viSTUV, vw23, 1);
          vaccWXYZn3 = vmlaq_lane_f32(vaccWXYZn3, viWXYZ, vw23, 1);
          vaccabcdn3 = vmlaq_lane_f32(vaccabcdn3, viabcd, vw23, 1);
          vaccefghn3 = vmlaq_lane_f32(vaccefghn3, viefgh, vw23, 1);
          vaccijkln3 = vmlaq_lane_f32(vaccijkln3, viijkl, vw23, 1);
          vaccmnopn3 = vmlaq_lane_f32(vaccmnopn3, vimnop, vw23, 1);
          vaccqrstn3 = vmlaq_lane_f32(vaccqrstn3, viqrst, vw23, 1);
          vaccuvwxn3 = vmlaq_lane_f32(vaccuvwxn3, viuvwx, vw23, 1);
          vaccyz01n3 = vmlaq_lane_f32(vaccyz01n3, viyz01, vw23, 1);
        } while (--nnz != 0);
      }
      float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
      float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
      float32x4_t vout89ABn0 = vminq_f32(vacc89ABn0, vmax);
      float32x4_t voutCDEFn0 = vminq_f32(vaccCDEFn0, vmax);
      float32x4_t voutGHIJn0 = vminq_f32(vaccGHIJn0, vmax);
      float32x4_t voutKLMNn0 = vminq_f32(vaccKLMNn0, vmax);
      float32x4_t voutOPQRn0 = vminq_f32(vaccOPQRn0, vmax);
      float32x4_t voutSTUVn0 = vminq_f32(vaccSTUVn0, vmax);
      float32x4_t voutWXYZn0 = vminq_f32(vaccWXYZn0, vmax);
      float32x4_t voutabcdn0 = vminq_f32(vaccabcdn0, vmax);
      float32x4_t voutefghn0 = vminq_f32(vaccefghn0, vmax);
      float32x4_t voutijkln0 = vminq_f32(vaccijkln0, vmax);
      float32x4_t voutmnopn0 = vminq_f32(vaccmnopn0, vmax);
      float32x4_t voutqrstn0 = vminq_f32(vaccqrstn0, vmax);
      float32x4_t voutuvwxn0 = vminq_f32(vaccuvwxn0, vmax);
      float32x4_t voutyz01n0 = vminq_f32(vaccyz01n0, vmax);
      float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
      float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
      float32x4_t vout89ABn1 = vminq_f32(vacc89ABn1, vmax);
      float32x4_t voutCDEFn1 = vminq_f32(vaccCDEFn1, vmax);
      float32x4_t voutGHIJn1 = vminq_f32(vaccGHIJn1, vmax);
      float32x4_t voutKLMNn1 = vminq_f32(vaccKLMNn1, vmax);
      float32x4_t voutOPQRn1 = vminq_f32(vaccOPQRn1, vmax);
      float32x4_t voutSTUVn1 = vminq_f32(vaccSTUVn1, vmax);
      float32x4_t voutWXYZn1 = vminq_f32(vaccWXYZn1, vmax);
      float32x4_t voutabcdn1 = vminq_f32(vaccabcdn1, vmax);
      float32x4_t voutefghn1 = vminq_f32(vaccefghn1, vmax);
      float32x4_t voutijkln1 = vminq_f32(vaccijkln1, vmax);
      float32x4_t voutmnopn1 = vminq_f32(vaccmnopn1, vmax);
      float32x4_t voutqrstn1 = vminq_f32(vaccqrstn1, vmax);
      float32x4_t voutuvwxn1 = vminq_f32(vaccuvwxn1, vmax);
      float32x4_t voutyz01n1 = vminq_f32(vaccyz01n1, vmax);
      float32x4_t vout0123n2 = vminq_f32(vacc0123n2, vmax);
      float32x4_t vout4567n2 = vminq_f32(vacc4567n2, vmax);
      float32x4_t vout89ABn2 = vminq_f32(vacc89ABn2, vmax);
      float32x4_t voutCDEFn2 = vminq_f32(vaccCDEFn2, vmax);
      float32x4_t voutGHIJn2 = vminq_f32(vaccGHIJn2, vmax);
      float32x4_t voutKLMNn2 = vminq_f32(vaccKLMNn2, vmax);
      float32x4_t voutOPQRn2 = vminq_f32(vaccOPQRn2, vmax);
      float32x4_t voutSTUVn2 = vminq_f32(vaccSTUVn2, vmax);
      float32x4_t voutWXYZn2 = vminq_f32(vaccWXYZn2, vmax);
      float32x4_t voutabcdn2 = vminq_f32(vaccabcdn2, vmax);
      float32x4_t voutefghn2 = vminq_f32(vaccefghn2, vmax);
      float32x4_t voutijkln2 = vminq_f32(vaccijkln2, vmax);
      float32x4_t voutmnopn2 = vminq_f32(vaccmnopn2, vmax);
      float32x4_t voutqrstn2 = vminq_f32(vaccqrstn2, vmax);
      float32x4_t voutuvwxn2 = vminq_f32(vaccuvwxn2, vmax);
      float32x4_t voutyz01n2 = vminq_f32(vaccyz01n2, vmax);
      float32x4_t vout0123n3 = vminq_f32(vacc0123n3, vmax);
      float32x4_t vout4567n3 = vminq_f32(vacc4567n3, vmax);
      float32x4_t vout89ABn3 = vminq_f32(vacc89ABn3, vmax);
      float32x4_t voutCDEFn3 = vminq_f32(vaccCDEFn3, vmax);
      float32x4_t voutGHIJn3 = vminq_f32(vaccGHIJn3, vmax);
      float32x4_t voutKLMNn3 = vminq_f32(vaccKLMNn3, vmax);
      float32x4_t voutOPQRn3 = vminq_f32(vaccOPQRn3, vmax);
      float32x4_t voutSTUVn3 = vminq_f32(vaccSTUVn3, vmax);
      float32x4_t voutWXYZn3 = vminq_f32(vaccWXYZn3, vmax);
      float32x4_t voutabcdn3 = vminq_f32(vaccabcdn3, vmax);
      float32x4_t voutefghn3 = vminq_f32(vaccefghn3, vmax);
      float32x4_t voutijkln3 = vminq_f32(vaccijkln3, vmax);
      float32x4_t voutmnopn3 = vminq_f32(vaccmnopn3, vmax);
      float32x4_t voutqrstn3 = vminq_f32(vaccqrstn3, vmax);
      float32x4_t voutuvwxn3 = vminq_f32(vaccuvwxn3, vmax);
      float32x4_t voutyz01n3 = vminq_f32(vaccyz01n3, vmax);

      vout0123n0 = vmaxq_f32(vout0123n0, vmin);
      vout4567n0 = vmaxq_f32(vout4567n0, vmin);
      vout89ABn0 = vmaxq_f32(vout89ABn0, vmin);
      voutCDEFn0 = vmaxq_f32(voutCDEFn0, vmin);
      voutGHIJn0 = vmaxq_f32(voutGHIJn0, vmin);
      voutKLMNn0 = vmaxq_f32(voutKLMNn0, vmin);
      voutOPQRn0 = vmaxq_f32(voutOPQRn0, vmin);
      voutSTUVn0 = vmaxq_f32(voutSTUVn0, vmin);
      voutWXYZn0 = vmaxq_f32(voutWXYZn0, vmin);
      voutabcdn0 = vmaxq_f32(voutabcdn0, vmin);
      voutefghn0 = vmaxq_f32(voutefghn0, vmin);
      voutijkln0 = vmaxq_f32(voutijkln0, vmin);
      voutmnopn0 = vmaxq_f32(voutmnopn0, vmin);
      voutqrstn0 = vmaxq_f32(voutqrstn0, vmin);
      voutuvwxn0 = vmaxq_f32(voutuvwxn0, vmin);
      voutyz01n0 = vmaxq_f32(voutyz01n0, vmin);
      vout0123n1 = vmaxq_f32(vout0123n1, vmin);
      vout4567n1 = vmaxq_f32(vout4567n1, vmin);
      vout89ABn1 = vmaxq_f32(vout89ABn1, vmin);
      voutCDEFn1 = vmaxq_f32(voutCDEFn1, vmin);
      voutGHIJn1 = vmaxq_f32(voutGHIJn1, vmin);
      voutKLMNn1 = vmaxq_f32(voutKLMNn1, vmin);
      voutOPQRn1 = vmaxq_f32(voutOPQRn1, vmin);
      voutSTUVn1 = vmaxq_f32(voutSTUVn1, vmin);
      voutWXYZn1 = vmaxq_f32(voutWXYZn1, vmin);
      voutabcdn1 = vmaxq_f32(voutabcdn1, vmin);
      voutefghn1 = vmaxq_f32(voutefghn1, vmin);
      voutijkln1 = vmaxq_f32(voutijkln1, vmin);
      voutmnopn1 = vmaxq_f32(voutmnopn1, vmin);
      voutqrstn1 = vmaxq_f32(voutqrstn1, vmin);
      voutuvwxn1 = vmaxq_f32(voutuvwxn1, vmin);
      voutyz01n1 = vmaxq_f32(voutyz01n1, vmin);
      vout0123n2 = vmaxq_f32(vout0123n2, vmin);
      vout4567n2 = vmaxq_f32(vout4567n2, vmin);
      vout89ABn2 = vmaxq_f32(vout89ABn2, vmin);
      voutCDEFn2 = vmaxq_f32(voutCDEFn2, vmin);
      voutGHIJn2 = vmaxq_f32(voutGHIJn2, vmin);
      voutKLMNn2 = vmaxq_f32(voutKLMNn2, vmin);
      voutOPQRn2 = vmaxq_f32(voutOPQRn2, vmin);
      voutSTUVn2 = vmaxq_f32(voutSTUVn2, vmin);
      voutWXYZn2 = vmaxq_f32(voutWXYZn2, vmin);
      voutabcdn2 = vmaxq_f32(voutabcdn2, vmin);
      voutefghn2 = vmaxq_f32(voutefghn2, vmin);
      voutijkln2 = vmaxq_f32(voutijkln2, vmin);
      voutmnopn2 = vmaxq_f32(voutmnopn2, vmin);
      voutqrstn2 = vmaxq_f32(voutqrstn2, vmin);
      voutuvwxn2 = vmaxq_f32(voutuvwxn2, vmin);
      voutyz01n2 = vmaxq_f32(voutyz01n2, vmin);
      vout0123n3 = vmaxq_f32(vout0123n3, vmin);
      vout4567n3 = vmaxq_f32(vout4567n3, vmin);
      vout89ABn3 = vmaxq_f32(vout89ABn3, vmin);
      voutCDEFn3 = vmaxq_f32(voutCDEFn3, vmin);
      voutGHIJn3 = vmaxq_f32(voutGHIJn3, vmin);
      voutKLMNn3 = vmaxq_f32(voutKLMNn3, vmin);
      voutOPQRn3 = vmaxq_f32(voutOPQRn3, vmin);
      voutSTUVn3 = vmaxq_f32(voutSTUVn3, vmin);
      voutWXYZn3 = vmaxq_f32(voutWXYZn3, vmin);
      voutabcdn3 = vmaxq_f32(voutabcdn3, vmin);
      voutefghn3 = vmaxq_f32(voutefghn3, vmin);
      voutijkln3 = vmaxq_f32(voutijkln3, vmin);
      voutmnopn3 = vmaxq_f32(voutmnopn3, vmin);
      voutqrstn3 = vmaxq_f32(voutqrstn3, vmin);
      voutuvwxn3 = vmaxq_f32(voutuvwxn3, vmin);
      voutyz01n3 = vmaxq_f32(voutyz01n3, vmin);

      vst1q_f32(output + 0, vout0123n0);
      vst1q_f32(output + 4, vout4567n0);
      vst1q_f32(output + 8, vout89ABn0);
      vst1q_f32(output + 12, voutCDEFn0);
      vst1q_f32(output + 16, voutGHIJn0);
      vst1q_f32(output + 20, voutKLMNn0);
      vst1q_f32(output + 24, voutOPQRn0);
      vst1q_f32(output + 28, voutSTUVn0);
      vst1q_f32(output + 32, voutWXYZn0);
      vst1q_f32(output + 36, voutabcdn0);
      vst1q_f32(output + 40, voutefghn0);
      vst1q_f32(output + 44, voutijkln0);
      vst1q_f32(output + 48, voutmnopn0);
      vst1q_f32(output + 52, voutqrstn0);
      vst1q_f32(output + 56, voutuvwxn0);
      vst1q_f32(output + 60, voutyz01n0);
      output = (float*restrict) ((uintptr_t) output + output_stride);
      vst1q_f32(output + 0, vout0123n1);
      vst1q_f32(output + 4, vout4567n1);
      vst1q_f32(output + 8, vout89ABn1);
      vst1q_f32(output + 12, voutCDEFn1);
      vst1q_f32(output + 16, voutGHIJn1);
      vst1q_f32(output + 20, voutKLMNn1);
      vst1q_f32(output + 24, voutOPQRn1);
      vst1q_f32(output + 28, voutSTUVn1);
      vst1q_f32(output + 32, voutWXYZn1);
      vst1q_f32(output + 36, voutabcdn1);
      vst1q_f32(output + 40, voutefghn1);
      vst1q_f32(output + 44, voutijkln1);
      vst1q_f32(output + 48, voutmnopn1);
      vst1q_f32(output + 52, voutqrstn1);
      vst1q_f32(output + 56, voutuvwxn1);
      vst1q_f32(output + 60, voutyz01n1);
      output = (float*restrict) ((uintptr_t) output + output_stride);
      vst1q_f32(output + 0, vout0123n2);
      vst1q_f32(output + 4, vout4567n2);
      vst1q_f32(output + 8, vout89ABn2);
      vst1q_f32(output + 12, voutCDEFn2);
      vst1q_f32(output + 16, voutGHIJn2);
      vst1q_f32(output + 20, voutKLMNn2);
      vst1q_f32(output + 24, voutOPQRn2);
      vst1q_f32(output + 28, voutSTUVn2);
      vst1q_f32(output + 32, voutWXYZn2);
      vst1q_f32(output + 36, voutabcdn2);
      vst1q_f32(output + 40, voutefghn2);
      vst1q_f32(output + 44, voutijkln2);
      vst1q_f32(output + 48, voutmnopn2);
      vst1q_f32(output + 52, voutqrstn2);
      vst1q_f32(output + 56, voutuvwxn2);
      vst1q_f32(output + 60, voutyz01n2);
      output = (float*restrict) ((uintptr_t) output + output_stride);
      vst1q_f32(output + 0, vout0123n3);
      vst1q_f32(output + 4, vout4567n3);
      vst1q_f32(output + 8, vout89ABn3);
      vst1q_f32(output + 12, voutCDEFn3);
      vst1q_f32(output + 16, voutGHIJn3);
      vst1q_f32(output + 20, voutKLMNn3);
      vst1q_f32(output + 24, voutOPQRn3);
      vst1q_f32(output + 28, voutSTUVn3);
      vst1q_f32(output + 32, voutWXYZn3);
      vst1q_f32(output + 36, voutabcdn3);
      vst1q_f32(output + 40, voutefghn3);
      vst1q_f32(output + 44, voutijkln3);
      vst1q_f32(output + 48, voutmnopn3);
      vst1q_f32(output + 52, voutqrstn3);
      vst1q_f32(output + 56, voutuvwxn3);
      vst1q_f32(output + 60, voutyz01n3);
      output = (float*restrict) ((uintptr_t) output + output_stride);
      n -= 4;
    }

    // clean up loop, fall back to nr=1
    if XNN_UNLIKELY(n != 0) {
      do {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567 = vacc0123;
        float32x4_t vacc89AB = vacc0123;
        float32x4_t vaccCDEF = vacc0123;
        float32x4_t vaccGHIJ = vacc0123;
        float32x4_t vaccKLMN = vacc0123;
        float32x4_t vaccOPQR = vacc0123;
        float32x4_t vaccSTUV = vacc0123;
        float32x4_t vaccWXYZ = vacc0123;
        float32x4_t vaccabcd = vacc0123;
        float32x4_t vaccefgh = vacc0123;
        float32x4_t vaccijkl = vacc0123;
        float32x4_t vaccmnop = vacc0123;
        float32x4_t vaccqrst = vacc0123;
        float32x4_t vaccuvwx = vacc0123;
        float32x4_t vaccyz01 = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            const float32x4_t viGHIJ = vld1q_f32(input + 16);
            const float32x4_t viKLMN = vld1q_f32(input + 20);
            const float32x4_t viOPQR = vld1q_f32(input + 24);
            const float32x4_t viSTUV = vld1q_f32(input + 28);
            const float32x4_t viWXYZ = vld1q_f32(input + 32);
            const float32x4_t viabcd = vld1q_f32(input + 36);
            const float32x4_t viefgh = vld1q_f32(input + 40);
            const float32x4_t viijkl = vld1q_f32(input + 44);
            const float32x4_t vimnop = vld1q_f32(input + 48);
            const float32x4_t viqrst = vld1q_f32(input + 52);
            const float32x4_t viuvwx = vld1q_f32(input + 56);
            const float32x4_t viyz01 = vld1q_f32(input + 60);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            __builtin_prefetch(input + 16);
            __builtin_prefetch(input + 32);
            __builtin_prefetch(input + 48);
            __builtin_prefetch(input + 64);
            const float32x4_t vw = vld1q_dup_f32(w); w += 1;
            __builtin_prefetch(w + 32);
            vacc0123 = vmlaq_f32(vacc0123, vi0123, vw);
            vacc4567 = vmlaq_f32(vacc4567, vi4567, vw);
            vacc89AB = vmlaq_f32(vacc89AB, vi89AB, vw);
            vaccCDEF = vmlaq_f32(vaccCDEF, viCDEF, vw);
            vaccGHIJ = vmlaq_f32(vaccGHIJ, viGHIJ, vw);
            vaccKLMN = vmlaq_f32(vaccKLMN, viKLMN, vw);
            vaccOPQR = vmlaq_f32(vaccOPQR, viOPQR, vw);
            vaccSTUV = vmlaq_f32(vaccSTUV, viSTUV, vw);
            vaccWXYZ = vmlaq_f32(vaccWXYZ, viWXYZ, vw);
            vaccabcd = vmlaq_f32(vaccabcd, viabcd, vw);
            vaccefgh = vmlaq_f32(vaccefgh, viefgh, vw);
            vaccijkl = vmlaq_f32(vaccijkl, viijkl, vw);
            vaccmnop = vmlaq_f32(vaccmnop, vimnop, vw);
            vaccqrst = vmlaq_f32(vaccqrst, viqrst, vw);
            vaccuvwx = vmlaq_f32(vaccuvwx, viuvwx, vw);
            vaccyz01 = vmlaq_f32(vaccyz01, viyz01, vw);
          } while (--nnz != 0);
        }
        float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
        float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
        float32x4_t vout89AB = vminq_f32(vacc89AB, vmax);
        float32x4_t voutCDEF = vminq_f32(vaccCDEF, vmax);
        float32x4_t voutGHIJ = vminq_f32(vaccGHIJ, vmax);
        float32x4_t voutKLMN = vminq_f32(vaccKLMN, vmax);
        float32x4_t voutOPQR = vminq_f32(vaccOPQR, vmax);
        float32x4_t voutSTUV = vminq_f32(vaccSTUV, vmax);
        float32x4_t voutWXYZ = vminq_f32(vaccWXYZ, vmax);
        float32x4_t voutabcd = vminq_f32(vaccabcd, vmax);
        float32x4_t voutefgh = vminq_f32(vaccefgh, vmax);
        float32x4_t voutijkl = vminq_f32(vaccijkl, vmax);
        float32x4_t voutmnop = vminq_f32(vaccmnop, vmax);
        float32x4_t voutqrst = vminq_f32(vaccqrst, vmax);
        float32x4_t voutuvwx = vminq_f32(vaccuvwx, vmax);
        float32x4_t voutyz01 = vminq_f32(vaccyz01, vmax);

        vout0123 = vmaxq_f32(vout0123, vmin);
        vout4567 = vmaxq_f32(vout4567, vmin);
        vout89AB = vmaxq_f32(vout89AB, vmin);
        voutCDEF = vmaxq_f32(voutCDEF, vmin);
        voutGHIJ = vmaxq_f32(voutGHIJ, vmin);
        voutKLMN = vmaxq_f32(voutKLMN, vmin);
        voutOPQR = vmaxq_f32(voutOPQR, vmin);
        voutSTUV = vmaxq_f32(voutSTUV, vmin);
        voutWXYZ = vmaxq_f32(voutWXYZ, vmin);
        voutabcd = vmaxq_f32(voutabcd, vmin);
        voutefgh = vmaxq_f32(voutefgh, vmin);
        voutijkl = vmaxq_f32(voutijkl, vmin);
        voutmnop = vmaxq_f32(voutmnop, vmin);
        voutqrst = vmaxq_f32(voutqrst, vmin);
        voutuvwx = vmaxq_f32(voutuvwx, vmin);
        voutyz01 = vmaxq_f32(voutyz01, vmin);

        vst1q_f32(output + 0, vout0123);
        vst1q_f32(output + 4, vout4567);
        vst1q_f32(output + 8, vout89AB);
        vst1q_f32(output + 12, voutCDEF);
        vst1q_f32(output + 16, voutGHIJ);
        vst1q_f32(output + 20, voutKLMN);
        vst1q_f32(output + 24, voutOPQR);
        vst1q_f32(output + 28, voutSTUV);
        vst1q_f32(output + 32, voutWXYZ);
        vst1q_f32(output + 36, voutabcd);
        vst1q_f32(output + 40, voutefgh);
        vst1q_f32(output + 44, voutijkl);
        vst1q_f32(output + 48, voutmnop);
        vst1q_f32(output + 52, voutqrst);
        vst1q_f32(output + 56, voutuvwx);
        vst1q_f32(output + 60, voutyz01);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      } while (n != 0);
    }
    output = (float*restrict) ((uintptr_t) output - output_decrement);
    input += 64;
    mc -= 64 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 32 * sizeof(float);
    if (mc & (32 * sizeof(float))) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n0 = vacc0123n0;
        float32x4_t vacc89ABn0 = vacc0123n0;
        float32x4_t vaccCDEFn0 = vacc0123n0;
        float32x4_t vaccGHIJn0 = vacc0123n0;
        float32x4_t vaccKLMNn0 = vacc0123n0;
        float32x4_t vaccOPQRn0 = vacc0123n0;
        float32x4_t vaccSTUVn0 = vacc0123n0;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n1 = vacc0123n1;
        float32x4_t vacc89ABn1 = vacc0123n1;
        float32x4_t vaccCDEFn1 = vacc0123n1;
        float32x4_t vaccGHIJn1 = vacc0123n1;
        float32x4_t vaccKLMNn1 = vacc0123n1;
        float32x4_t vaccOPQRn1 = vacc0123n1;
        float32x4_t vaccSTUVn1 = vacc0123n1;
        float32x4_t vacc0123n2 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n2 = vacc0123n2;
        float32x4_t vacc89ABn2 = vacc0123n2;
        float32x4_t vaccCDEFn2 = vacc0123n2;
        float32x4_t vaccGHIJn2 = vacc0123n2;
        float32x4_t vaccKLMNn2 = vacc0123n2;
        float32x4_t vaccOPQRn2 = vacc0123n2;
        float32x4_t vaccSTUVn2 = vacc0123n2;
        float32x4_t vacc0123n3 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n3 = vacc0123n3;
        float32x4_t vacc89ABn3 = vacc0123n3;
        float32x4_t vaccCDEFn3 = vacc0123n3;
        float32x4_t vaccGHIJn3 = vacc0123n3;
        float32x4_t vaccKLMNn3 = vacc0123n3;
        float32x4_t vaccOPQRn3 = vacc0123n3;
        float32x4_t vaccSTUVn3 = vacc0123n3;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            const float32x4_t viGHIJ = vld1q_f32(input + 16);
            const float32x4_t viKLMN = vld1q_f32(input + 20);
            const float32x4_t viOPQR = vld1q_f32(input + 24);
            const float32x4_t viSTUV = vld1q_f32(input + 28);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw01 = vld1_f32(w); w += 2;
            const float32x2_t vw23 = vld1_f32(w); w += 2;

            vacc0123n0 = vmlaq_lane_f32(vacc0123n0, vi0123, vw01, 0);
            vacc4567n0 = vmlaq_lane_f32(vacc4567n0, vi4567, vw01, 0);
            vacc89ABn0 = vmlaq_lane_f32(vacc89ABn0, vi89AB, vw01, 0);
            vaccCDEFn0 = vmlaq_lane_f32(vaccCDEFn0, viCDEF, vw01, 0);
            vaccGHIJn0 = vmlaq_lane_f32(vaccGHIJn0, viGHIJ, vw01, 0);
            vaccKLMNn0 = vmlaq_lane_f32(vaccKLMNn0, viKLMN, vw01, 0);
            vaccOPQRn0 = vmlaq_lane_f32(vaccOPQRn0, viOPQR, vw01, 0);
            vaccSTUVn0 = vmlaq_lane_f32(vaccSTUVn0, viSTUV, vw01, 0);
            vacc0123n1 = vmlaq_lane_f32(vacc0123n1, vi0123, vw01, 1);
            vacc4567n1 = vmlaq_lane_f32(vacc4567n1, vi4567, vw01, 1);
            vacc89ABn1 = vmlaq_lane_f32(vacc89ABn1, vi89AB, vw01, 1);
            vaccCDEFn1 = vmlaq_lane_f32(vaccCDEFn1, viCDEF, vw01, 1);
            vaccGHIJn1 = vmlaq_lane_f32(vaccGHIJn1, viGHIJ, vw01, 1);
            vaccKLMNn1 = vmlaq_lane_f32(vaccKLMNn1, viKLMN, vw01, 1);
            vaccOPQRn1 = vmlaq_lane_f32(vaccOPQRn1, viOPQR, vw01, 1);
            vaccSTUVn1 = vmlaq_lane_f32(vaccSTUVn1, viSTUV, vw01, 1);
            vacc0123n2 = vmlaq_lane_f32(vacc0123n2, vi0123, vw23, 0);
            vacc4567n2 = vmlaq_lane_f32(vacc4567n2, vi4567, vw23, 0);
            vacc89ABn2 = vmlaq_lane_f32(vacc89ABn2, vi89AB, vw23, 0);
            vaccCDEFn2 = vmlaq_lane_f32(vaccCDEFn2, viCDEF, vw23, 0);
            vaccGHIJn2 = vmlaq_lane_f32(vaccGHIJn2, viGHIJ, vw23, 0);
            vaccKLMNn2 = vmlaq_lane_f32(vaccKLMNn2, viKLMN, vw23, 0);
            vaccOPQRn2 = vmlaq_lane_f32(vaccOPQRn2, viOPQR, vw23, 0);
            vaccSTUVn2 = vmlaq_lane_f32(vaccSTUVn2, viSTUV, vw23, 0);
            vacc0123n3 = vmlaq_lane_f32(vacc0123n3, vi0123, vw23, 1);
            vacc4567n3 = vmlaq_lane_f32(vacc4567n3, vi4567, vw23, 1);
            vacc89ABn3 = vmlaq_lane_f32(vacc89ABn3, vi89AB, vw23, 1);
            vaccCDEFn3 = vmlaq_lane_f32(vaccCDEFn3, viCDEF, vw23, 1);
            vaccGHIJn3 = vmlaq_lane_f32(vaccGHIJn3, viGHIJ, vw23, 1);
            vaccKLMNn3 = vmlaq_lane_f32(vaccKLMNn3, viKLMN, vw23, 1);
            vaccOPQRn3 = vmlaq_lane_f32(vaccOPQRn3, viOPQR, vw23, 1);
            vaccSTUVn3 = vmlaq_lane_f32(vaccSTUVn3, viSTUV, vw23, 1);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
        float32x4_t vout89ABn0 = vminq_f32(vacc89ABn0, vmax);
        float32x4_t voutCDEFn0 = vminq_f32(vaccCDEFn0, vmax);
        float32x4_t voutGHIJn0 = vminq_f32(vaccGHIJn0, vmax);
        float32x4_t voutKLMNn0 = vminq_f32(vaccKLMNn0, vmax);
        float32x4_t voutOPQRn0 = vminq_f32(vaccOPQRn0, vmax);
        float32x4_t voutSTUVn0 = vminq_f32(vaccSTUVn0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
        float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
        float32x4_t vout89ABn1 = vminq_f32(vacc89ABn1, vmax);
        float32x4_t voutCDEFn1 = vminq_f32(vaccCDEFn1, vmax);
        float32x4_t voutGHIJn1 = vminq_f32(vaccGHIJn1, vmax);
        float32x4_t voutKLMNn1 = vminq_f32(vaccKLMNn1, vmax);
        float32x4_t voutOPQRn1 = vminq_f32(vaccOPQRn1, vmax);
        float32x4_t voutSTUVn1 = vminq_f32(vaccSTUVn1, vmax);
        float32x4_t vout0123n2 = vminq_f32(vacc0123n2, vmax);
        float32x4_t vout4567n2 = vminq_f32(vacc4567n2, vmax);
        float32x4_t vout89ABn2 = vminq_f32(vacc89ABn2, vmax);
        float32x4_t voutCDEFn2 = vminq_f32(vaccCDEFn2, vmax);
        float32x4_t voutGHIJn2 = vminq_f32(vaccGHIJn2, vmax);
        float32x4_t voutKLMNn2 = vminq_f32(vaccKLMNn2, vmax);
        float32x4_t voutOPQRn2 = vminq_f32(vaccOPQRn2, vmax);
        float32x4_t voutSTUVn2 = vminq_f32(vaccSTUVn2, vmax);
        float32x4_t vout0123n3 = vminq_f32(vacc0123n3, vmax);
        float32x4_t vout4567n3 = vminq_f32(vacc4567n3, vmax);
        float32x4_t vout89ABn3 = vminq_f32(vacc89ABn3, vmax);
        float32x4_t voutCDEFn3 = vminq_f32(vaccCDEFn3, vmax);
        float32x4_t voutGHIJn3 = vminq_f32(vaccGHIJn3, vmax);
        float32x4_t voutKLMNn3 = vminq_f32(vaccKLMNn3, vmax);
        float32x4_t voutOPQRn3 = vminq_f32(vaccOPQRn3, vmax);
        float32x4_t voutSTUVn3 = vminq_f32(vaccSTUVn3, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout4567n0 = vmaxq_f32(vout4567n0, vmin);
        vout89ABn0 = vmaxq_f32(vout89ABn0, vmin);
        voutCDEFn0 = vmaxq_f32(voutCDEFn0, vmin);
        voutGHIJn0 = vmaxq_f32(voutGHIJn0, vmin);
        voutKLMNn0 = vmaxq_f32(voutKLMNn0, vmin);
        voutOPQRn0 = vmaxq_f32(voutOPQRn0, vmin);
        voutSTUVn0 = vmaxq_f32(voutSTUVn0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);
        vout4567n1 = vmaxq_f32(vout4567n1, vmin);
        vout89ABn1 = vmaxq_f32(vout89ABn1, vmin);
        voutCDEFn1 = vmaxq_f32(voutCDEFn1, vmin);
        voutGHIJn1 = vmaxq_f32(voutGHIJn1, vmin);
        voutKLMNn1 = vmaxq_f32(voutKLMNn1, vmin);
        voutOPQRn1 = vmaxq_f32(voutOPQRn1, vmin);
        voutSTUVn1 = vmaxq_f32(voutSTUVn1, vmin);
        vout0123n2 = vmaxq_f32(vout0123n2, vmin);
        vout4567n2 = vmaxq_f32(vout4567n2, vmin);
        vout89ABn2 = vmaxq_f32(vout89ABn2, vmin);
        voutCDEFn2 = vmaxq_f32(voutCDEFn2, vmin);
        voutGHIJn2 = vmaxq_f32(voutGHIJn2, vmin);
        voutKLMNn2 = vmaxq_f32(voutKLMNn2, vmin);
        voutOPQRn2 = vmaxq_f32(voutOPQRn2, vmin);
        voutSTUVn2 = vmaxq_f32(voutSTUVn2, vmin);
        vout0123n3 = vmaxq_f32(vout0123n3, vmin);
        vout4567n3 = vmaxq_f32(vout4567n3, vmin);
        vout89ABn3 = vmaxq_f32(vout89ABn3, vmin);
        voutCDEFn3 = vmaxq_f32(voutCDEFn3, vmin);
        voutGHIJn3 = vmaxq_f32(voutGHIJn3, vmin);
        voutKLMNn3 = vmaxq_f32(voutKLMNn3, vmin);
        voutOPQRn3 = vmaxq_f32(voutOPQRn3, vmin);
        voutSTUVn3 = vmaxq_f32(voutSTUVn3, vmin);

        vst1q_f32(output + 0, vout0123n0);
        vst1q_f32(output + 4, vout4567n0);
        vst1q_f32(output + 8, vout89ABn0);
        vst1q_f32(output + 12, voutCDEFn0);
        vst1q_f32(output + 16, voutGHIJn0);
        vst1q_f32(output + 20, voutKLMNn0);
        vst1q_f32(output + 24, voutOPQRn0);
        vst1q_f32(output + 28, voutSTUVn0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        vst1q_f32(output + 4, vout4567n1);
        vst1q_f32(output + 8, vout89ABn1);
        vst1q_f32(output + 12, voutCDEFn1);
        vst1q_f32(output + 16, voutGHIJn1);
        vst1q_f32(output + 20, voutKLMNn1);
        vst1q_f32(output + 24, voutOPQRn1);
        vst1q_f32(output + 28, voutSTUVn1);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n2);
        vst1q_f32(output + 4, vout4567n2);
        vst1q_f32(output + 8, vout89ABn2);
        vst1q_f32(output + 12, voutCDEFn2);
        vst1q_f32(output + 16, voutGHIJn2);
        vst1q_f32(output + 20, voutKLMNn2);
        vst1q_f32(output + 24, voutOPQRn2);
        vst1q_f32(output + 28, voutSTUVn2);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n3);
        vst1q_f32(output + 4, vout4567n3);
        vst1q_f32(output + 8, vout89ABn3);
        vst1q_f32(output + 12, voutCDEFn3);
        vst1q_f32(output + 16, voutGHIJn3);
        vst1q_f32(output + 20, voutKLMNn3);
        vst1q_f32(output + 24, voutOPQRn3);
        vst1q_f32(output + 28, voutSTUVn3);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 4;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
          float32x4_t vacc4567 = vacc0123;
          float32x4_t vacc89AB = vacc0123;
          float32x4_t vaccCDEF = vacc0123;
          float32x4_t vaccGHIJ = vacc0123;
          float32x4_t vaccKLMN = vacc0123;
          float32x4_t vaccOPQR = vacc0123;
          float32x4_t vaccSTUV = vacc0123;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x4_t vi0123 = vld1q_f32(input);
              const float32x4_t vi4567 = vld1q_f32(input + 4);
              const float32x4_t vi89AB = vld1q_f32(input + 8);
              const float32x4_t viCDEF = vld1q_f32(input + 12);
              const float32x4_t viGHIJ = vld1q_f32(input + 16);
              const float32x4_t viKLMN = vld1q_f32(input + 20);
              const float32x4_t viOPQR = vld1q_f32(input + 24);
              const float32x4_t viSTUV = vld1q_f32(input + 28);
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vmlaq_f32(vacc0123, vi0123, vw);
              vacc4567 = vmlaq_f32(vacc4567, vi4567, vw);
              vacc89AB = vmlaq_f32(vacc89AB, vi89AB, vw);
              vaccCDEF = vmlaq_f32(vaccCDEF, viCDEF, vw);
              vaccGHIJ = vmlaq_f32(vaccGHIJ, viGHIJ, vw);
              vaccKLMN = vmlaq_f32(vaccKLMN, viKLMN, vw);
              vaccOPQR = vmlaq_f32(vaccOPQR, viOPQR, vw);
              vaccSTUV = vmlaq_f32(vaccSTUV, viSTUV, vw);
            } while (--nnz != 0);
          }
          float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
          float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
          float32x4_t vout89AB = vminq_f32(vacc89AB, vmax);
          float32x4_t voutCDEF = vminq_f32(vaccCDEF, vmax);
          float32x4_t voutGHIJ = vminq_f32(vaccGHIJ, vmax);
          float32x4_t voutKLMN = vminq_f32(vaccKLMN, vmax);
          float32x4_t voutOPQR = vminq_f32(vaccOPQR, vmax);
          float32x4_t voutSTUV = vminq_f32(vaccSTUV, vmax);

          vout0123 = vmaxq_f32(vout0123, vmin);
          vout4567 = vmaxq_f32(vout4567, vmin);
          vout89AB = vmaxq_f32(vout89AB, vmin);
          voutCDEF = vmaxq_f32(voutCDEF, vmin);
          voutGHIJ = vmaxq_f32(voutGHIJ, vmin);
          voutKLMN = vmaxq_f32(voutKLMN, vmin);
          voutOPQR = vmaxq_f32(voutOPQR, vmin);
          voutSTUV = vmaxq_f32(voutSTUV, vmin);

          vst1q_f32(output + 0, vout0123);
          vst1q_f32(output + 4, vout4567);
          vst1q_f32(output + 8, vout89AB);
          vst1q_f32(output + 12, voutCDEF);
          vst1q_f32(output + 16, voutGHIJ);
          vst1q_f32(output + 20, voutKLMN);
          vst1q_f32(output + 24, voutOPQR);
          vst1q_f32(output + 28, voutSTUV);
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 32;
    }
    output_decrement += 16 * sizeof(float);
    if (mc & (16 * sizeof(float))) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n0 = vacc0123n0;
        float32x4_t vacc89ABn0 = vacc0123n0;
        float32x4_t vaccCDEFn0 = vacc0123n0;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n1 = vacc0123n1;
        float32x4_t vacc89ABn1 = vacc0123n1;
        float32x4_t vaccCDEFn1 = vacc0123n1;
        float32x4_t vacc0123n2 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n2 = vacc0123n2;
        float32x4_t vacc89ABn2 = vacc0123n2;
        float32x4_t vaccCDEFn2 = vacc0123n2;
        float32x4_t vacc0123n3 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n3 = vacc0123n3;
        float32x4_t vacc89ABn3 = vacc0123n3;
        float32x4_t vaccCDEFn3 = vacc0123n3;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw01 = vld1_f32(w); w += 2;
            const float32x2_t vw23 = vld1_f32(w); w += 2;

            vacc0123n0 = vmlaq_lane_f32(vacc0123n0, vi0123, vw01, 0);
            vacc4567n0 = vmlaq_lane_f32(vacc4567n0, vi4567, vw01, 0);
            vacc89ABn0 = vmlaq_lane_f32(vacc89ABn0, vi89AB, vw01, 0);
            vaccCDEFn0 = vmlaq_lane_f32(vaccCDEFn0, viCDEF, vw01, 0);
            vacc0123n1 = vmlaq_lane_f32(vacc0123n1, vi0123, vw01, 1);
            vacc4567n1 = vmlaq_lane_f32(vacc4567n1, vi4567, vw01, 1);
            vacc89ABn1 = vmlaq_lane_f32(vacc89ABn1, vi89AB, vw01, 1);
            vaccCDEFn1 = vmlaq_lane_f32(vaccCDEFn1, viCDEF, vw01, 1);
            vacc0123n2 = vmlaq_lane_f32(vacc0123n2, vi0123, vw23, 0);
            vacc4567n2 = vmlaq_lane_f32(vacc4567n2, vi4567, vw23, 0);
            vacc89ABn2 = vmlaq_lane_f32(vacc89ABn2, vi89AB, vw23, 0);
            vaccCDEFn2 = vmlaq_lane_f32(vaccCDEFn2, viCDEF, vw23, 0);
            vacc0123n3 = vmlaq_lane_f32(vacc0123n3, vi0123, vw23, 1);
            vacc4567n3 = vmlaq_lane_f32(vacc4567n3, vi4567, vw23, 1);
            vacc89ABn3 = vmlaq_lane_f32(vacc89ABn3, vi89AB, vw23, 1);
            vaccCDEFn3 = vmlaq_lane_f32(vaccCDEFn3, viCDEF, vw23, 1);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
        float32x4_t vout89ABn0 = vminq_f32(vacc89ABn0, vmax);
        float32x4_t voutCDEFn0 = vminq_f32(vaccCDEFn0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
        float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
        float32x4_t vout89ABn1 = vminq_f32(vacc89ABn1, vmax);
        float32x4_t voutCDEFn1 = vminq_f32(vaccCDEFn1, vmax);
        float32x4_t vout0123n2 = vminq_f32(vacc0123n2, vmax);
        float32x4_t vout4567n2 = vminq_f32(vacc4567n2, vmax);
        float32x4_t vout89ABn2 = vminq_f32(vacc89ABn2, vmax);
        float32x4_t voutCDEFn2 = vminq_f32(vaccCDEFn2, vmax);
        float32x4_t vout0123n3 = vminq_f32(vacc0123n3, vmax);
        float32x4_t vout4567n3 = vminq_f32(vacc4567n3, vmax);
        float32x4_t vout89ABn3 = vminq_f32(vacc89ABn3, vmax);
        float32x4_t voutCDEFn3 = vminq_f32(vaccCDEFn3, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout4567n0 = vmaxq_f32(vout4567n0, vmin);
        vout89ABn0 = vmaxq_f32(vout89ABn0, vmin);
        voutCDEFn0 = vmaxq_f32(voutCDEFn0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);
        vout4567n1 = vmaxq_f32(vout4567n1, vmin);
        vout89ABn1 = vmaxq_f32(vout89ABn1, vmin);
        voutCDEFn1 = vmaxq_f32(voutCDEFn1, vmin);
        vout0123n2 = vmaxq_f32(vout0123n2, vmin);
        vout4567n2 = vmaxq_f32(vout4567n2, vmin);
        vout89ABn2 = vmaxq_f32(vout89ABn2, vmin);
        voutCDEFn2 = vmaxq_f32(voutCDEFn2, vmin);
        vout0123n3 = vmaxq_f32(vout0123n3, vmin);
        vout4567n3 = vmaxq_f32(vout4567n3, vmin);
        vout89ABn3 = vmaxq_f32(vout89ABn3, vmin);
        voutCDEFn3 = vmaxq_f32(voutCDEFn3, vmin);

        vst1q_f32(output + 0, vout0123n0);
        vst1q_f32(output + 4, vout4567n0);
        vst1q_f32(output + 8, vout89ABn0);
        vst1q_f32(output + 12, voutCDEFn0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        vst1q_f32(output + 4, vout4567n1);
        vst1q_f32(output + 8, vout89ABn1);
        vst1q_f32(output + 12, voutCDEFn1);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n2);
        vst1q_f32(output + 4, vout4567n2);
        vst1q_f32(output + 8, vout89ABn2);
        vst1q_f32(output + 12, voutCDEFn2);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n3);
        vst1q_f32(output + 4, vout4567n3);
        vst1q_f32(output + 8, vout89ABn3);
        vst1q_f32(output + 12, voutCDEFn3);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 4;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
          float32x4_t vacc4567 = vacc0123;
          float32x4_t vacc89AB = vacc0123;
          float32x4_t vaccCDEF = vacc0123;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x4_t vi0123 = vld1q_f32(input);
              const float32x4_t vi4567 = vld1q_f32(input + 4);
              const float32x4_t vi89AB = vld1q_f32(input + 8);
              const float32x4_t viCDEF = vld1q_f32(input + 12);
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vmlaq_f32(vacc0123, vi0123, vw);
              vacc4567 = vmlaq_f32(vacc4567, vi4567, vw);
              vacc89AB = vmlaq_f32(vacc89AB, vi89AB, vw);
              vaccCDEF = vmlaq_f32(vaccCDEF, viCDEF, vw);
            } while (--nnz != 0);
          }
          float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
          float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
          float32x4_t vout89AB = vminq_f32(vacc89AB, vmax);
          float32x4_t voutCDEF = vminq_f32(vaccCDEF, vmax);

          vout0123 = vmaxq_f32(vout0123, vmin);
          vout4567 = vmaxq_f32(vout4567, vmin);
          vout89AB = vmaxq_f32(vout89AB, vmin);
          voutCDEF = vmaxq_f32(voutCDEF, vmin);

          vst1q_f32(output + 0, vout0123);
          vst1q_f32(output + 4, vout4567);
          vst1q_f32(output + 8, vout89AB);
          vst1q_f32(output + 12, voutCDEF);
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 16;
    }
    output_decrement += 8 * sizeof(float);
    if (mc & (8 * sizeof(float))) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n0 = vacc0123n0;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n1 = vacc0123n1;
        float32x4_t vacc0123n2 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n2 = vacc0123n2;
        float32x4_t vacc0123n3 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n3 = vacc0123n3;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw01 = vld1_f32(w); w += 2;
            const float32x2_t vw23 = vld1_f32(w); w += 2;

            vacc0123n0 = vmlaq_lane_f32(vacc0123n0, vi0123, vw01, 0);
            vacc4567n0 = vmlaq_lane_f32(vacc4567n0, vi4567, vw01, 0);
            vacc0123n1 = vmlaq_lane_f32(vacc0123n1, vi0123, vw01, 1);
            vacc4567n1 = vmlaq_lane_f32(vacc4567n1, vi4567, vw01, 1);
            vacc0123n2 = vmlaq_lane_f32(vacc0123n2, vi0123, vw23, 0);
            vacc4567n2 = vmlaq_lane_f32(vacc4567n2, vi4567, vw23, 0);
            vacc0123n3 = vmlaq_lane_f32(vacc0123n3, vi0123, vw23, 1);
            vacc4567n3 = vmlaq_lane_f32(vacc4567n3, vi4567, vw23, 1);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
        float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
        float32x4_t vout0123n2 = vminq_f32(vacc0123n2, vmax);
        float32x4_t vout4567n2 = vminq_f32(vacc4567n2, vmax);
        float32x4_t vout0123n3 = vminq_f32(vacc0123n3, vmax);
        float32x4_t vout4567n3 = vminq_f32(vacc4567n3, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout4567n0 = vmaxq_f32(vout4567n0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);
        vout4567n1 = vmaxq_f32(vout4567n1, vmin);
        vout0123n2 = vmaxq_f32(vout0123n2, vmin);
        vout4567n2 = vmaxq_f32(vout4567n2, vmin);
        vout0123n3 = vmaxq_f32(vout0123n3, vmin);
        vout4567n3 = vmaxq_f32(vout4567n3, vmin);

        vst1q_f32(output + 0, vout0123n0);
        vst1q_f32(output + 4, vout4567n0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        vst1q_f32(output + 4, vout4567n1);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n2);
        vst1q_f32(output + 4, vout4567n2);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n3);
        vst1q_f32(output + 4, vout4567n3);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 4;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
          float32x4_t vacc4567 = vacc0123;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x4_t vi0123 = vld1q_f32(input);
              const float32x4_t vi4567 = vld1q_f32(input + 4);
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vmlaq_f32(vacc0123, vi0123, vw);
              vacc4567 = vmlaq_f32(vacc4567, vi4567, vw);
            } while (--nnz != 0);
          }
          float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
          float32x4_t vout4567 = vminq_f32(vacc4567, vmax);

          vout0123 = vmaxq_f32(vout0123, vmin);
          vout4567 = vmaxq_f32(vout4567, vmin);

          vst1q_f32(output + 0, vout0123);
          vst1q_f32(output + 4, vout4567);
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 8;
    }
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123n2 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123n3 = vld1q_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw01 = vld1_f32(w); w += 2;
            const float32x2_t vw23 = vld1_f32(w); w += 2;

            vacc0123n0 = vmlaq_lane_f32(vacc0123n0, vi0123, vw01, 0);
            vacc0123n1 = vmlaq_lane_f32(vacc0123n1, vi0123, vw01, 1);
            vacc0123n2 = vmlaq_lane_f32(vacc0123n2, vi0123, vw23, 0);
            vacc0123n3 = vmlaq_lane_f32(vacc0123n3, vi0123, vw23, 1);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
        float32x4_t vout0123n2 = vminq_f32(vacc0123n2, vmax);
        float32x4_t vout0123n3 = vminq_f32(vacc0123n3, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);
        vout0123n2 = vmaxq_f32(vout0123n2, vmin);
        vout0123n3 = vmaxq_f32(vout0123n3, vmin);

        vst1q_f32(output + 0, vout0123n0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n2);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n3);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 4;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x4_t vi0123 = vld1q_f32(input);
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vmlaq_f32(vacc0123, vi0123, vw);
            } while (--nnz != 0);
          }
          float32x4_t vout0123 = vminq_f32(vacc0123, vmax);

          vout0123 = vmaxq_f32(vout0123, vmin);

          vst1q_f32(output + 0, vout0123);
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc01n0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01n1 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01n2 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01n3 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi01 = vld1_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw01 = vld1_f32(w); w += 2;
            const float32x2_t vw23 = vld1_f32(w); w += 2;

            vacc01n0 = vmla_lane_f32(vacc01n0, vi01, vw01, 0);
            vacc01n1 = vmla_lane_f32(vacc01n1, vi01, vw01, 1);
            vacc01n2 = vmla_lane_f32(vacc01n2, vi01, vw23, 0);
            vacc01n3 = vmla_lane_f32(vacc01n3, vi01, vw23, 1);
          } while (--nnz != 0);
        }
        float32x2_t vout01n0 = vmin_f32(vacc01n0, vget_low_f32(vmax));
        float32x2_t vout01n1 = vmin_f32(vacc01n1, vget_low_f32(vmax));
        float32x2_t vout01n2 = vmin_f32(vacc01n2, vget_low_f32(vmax));
        float32x2_t vout01n3 = vmin_f32(vacc01n3, vget_low_f32(vmax));

        vout01n0 = vmax_f32(vout01n0, vget_low_f32(vmin));
        vout01n1 = vmax_f32(vout01n1, vget_low_f32(vmin));
        vout01n2 = vmax_f32(vout01n2, vget_low_f32(vmin));
        vout01n3 = vmax_f32(vout01n3, vget_low_f32(vmin));

        vst1_f32(output + 0, vout01n0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1_f32(output + 0, vout01n1);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1_f32(output + 0, vout01n2);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1_f32(output + 0, vout01n3);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 4;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x2_t vacc01 = vld1_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x2_t vi01 = vld1_f32(input);
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float32x2_t vw = vld1_dup_f32(w); w += 1;
              vacc01 = vmla_f32(vacc01, vi01, vw);
            } while (--nnz != 0);
          }
          float32x2_t vout01 = vmin_f32(vacc01, vget_low_f32(vmax));
          vout01 = vmax_f32(vout01, vget_low_f32(vmin));

          vst1_f32(output, vout01);
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc0n0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0n1 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0n2 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0n3 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi0 = vld1_dup_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw01 = vld1_f32(w); w += 2;
            const float32x2_t vw23 = vld1_f32(w); w += 2;

            vacc0n0 = vmla_lane_f32(vacc0n0, vi0, vw01, 0);
            vacc0n1 = vmla_lane_f32(vacc0n1, vi0, vw01, 1);
            vacc0n2 = vmla_lane_f32(vacc0n2, vi0, vw23, 0);
            vacc0n3 = vmla_lane_f32(vacc0n3, vi0, vw23, 1);
          } while (--nnz != 0);
        }
        float32x2_t vout0n0 = vmin_f32(vacc0n0, vget_low_f32(vmax));
        float32x2_t vout0n1 = vmin_f32(vacc0n1, vget_low_f32(vmax));
        float32x2_t vout0n2 = vmin_f32(vacc0n2, vget_low_f32(vmax));
        float32x2_t vout0n3 = vmin_f32(vacc0n3, vget_low_f32(vmax));

        vout0n0 = vmax_f32(vout0n0, vget_low_f32(vmin));
        vout0n1 = vmax_f32(vout0n1, vget_low_f32(vmin));
        vout0n2 = vmax_f32(vout0n2, vget_low_f32(vmin));
        vout0n3 = vmax_f32(vout0n3, vget_low_f32(vmin));

        vst1_lane_f32(output + 0, vout0n0, 0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1_lane_f32(output + 0, vout0n1, 0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1_lane_f32(output + 0, vout0n2, 0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        vst1_lane_f32(output + 0, vout0n3, 0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 4;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x2_t vacc0 = vld1_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x2_t vi0 = vld1_dup_f32(input);
              input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
              const float32x2_t vw = vld1_dup_f32(w); w += 1;
              vacc0 = vmla_f32(vacc0, vi0, vw);
            } while (--nnz != 0);
          }
          float32x2_t vout0 = vmin_f32(vacc0, vget_low_f32(vmax));
          vout0 = vmax_f32(vout0, vget_low_f32(vmin));

          vst1_lane_f32(output, vout0, 1);
          output = (float*restrict) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
    }
}
