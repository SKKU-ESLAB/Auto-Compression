// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/neon-blocked-unaligned.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/spmm.h>


void xnn_f32_spmm_minmax_ukernel_64x2__neon_unaligned(
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
    size_t n = nc - 1;

    // For the first blocked row
    float32x4_t vacc0123n0;
    float32x4_t vacc4567n0;
    float32x4_t vacc89ABn0;
    float32x4_t vaccCDEFn0;
    float32x4_t vaccGHIJn0;
    float32x4_t vaccKLMNn0;
    float32x4_t vaccOPQRn0;
    float32x4_t vaccSTUVn0;
    float32x4_t vaccWXYZn0;
    float32x4_t vaccabcdn0;
    float32x4_t vaccefghn0;
    float32x4_t vaccijkln0;
    float32x4_t vaccmnopn0;
    float32x4_t vaccqrstn0;
    float32x4_t vaccuvwxn0;
    float32x4_t vaccyz01n0;
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

    while (n != 0) {
      uint32_t nnz = *nnzmap++;

      // Temporary output pipelining
      vacc0123n0 = vacc0123n1;
      vacc4567n0 = vacc4567n1;
      vacc89ABn0 = vacc89ABn1;
      vaccCDEFn0 = vaccCDEFn1;
      vaccGHIJn0 = vaccGHIJn1;
      vaccKLMNn0 = vaccKLMNn1;
      vaccOPQRn0 = vaccOPQRn1;
      vaccSTUVn0 = vaccSTUVn1;
      vaccWXYZn0 = vaccWXYZn1;
      vaccabcdn0 = vaccabcdn1;
      vaccefghn0 = vaccefghn1;
      vaccijkln0 = vaccijkln1;
      vaccmnopn0 = vaccmnopn1;
      vaccqrstn0 = vaccqrstn1;
      vaccuvwxn0 = vaccuvwxn1;
      vaccyz01n0 = vaccyz01n1;
      vacc0123n1 = vld1q_dup_f32(w); w += 1;
      vacc4567n1 = vacc0123n1;
      vacc89ABn1 = vacc0123n1;
      vaccCDEFn1 = vacc0123n1;
      vaccGHIJn1 = vacc0123n1;
      vaccKLMNn1 = vacc0123n1;
      vaccOPQRn1 = vacc0123n1;
      vaccSTUVn1 = vacc0123n1;
      vaccWXYZn1 = vacc0123n1;
      vaccabcdn1 = vacc0123n1;
      vaccefghn1 = vacc0123n1;
      vaccijkln1 = vacc0123n1;
      vaccmnopn1 = vacc0123n1;
      vaccqrstn1 = vacc0123n1;
      vaccuvwxn1 = vacc0123n1;
      vaccyz01n1 = vacc0123n1;
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
          const float32x2_t vw = vld1_f32(w); w += 2;
          __builtin_prefetch(w + 32);
          vacc0123n0 = vmlaq_lane_f32(vacc0123n0, vi0123, vw, 0);
          vacc4567n0 = vmlaq_lane_f32(vacc4567n0, vi4567, vw, 0);
          vacc89ABn0 = vmlaq_lane_f32(vacc89ABn0, vi89AB, vw, 0);
          vaccCDEFn0 = vmlaq_lane_f32(vaccCDEFn0, viCDEF, vw, 0);
          vaccGHIJn0 = vmlaq_lane_f32(vaccGHIJn0, viGHIJ, vw, 0);
          vaccKLMNn0 = vmlaq_lane_f32(vaccKLMNn0, viKLMN, vw, 0);
          vaccOPQRn0 = vmlaq_lane_f32(vaccOPQRn0, viOPQR, vw, 0);
          vaccSTUVn0 = vmlaq_lane_f32(vaccSTUVn0, viSTUV, vw, 0);
          vaccWXYZn0 = vmlaq_lane_f32(vaccWXYZn0, viWXYZ, vw, 0);
          vaccabcdn0 = vmlaq_lane_f32(vaccabcdn0, viabcd, vw, 0);
          vaccefghn0 = vmlaq_lane_f32(vaccefghn0, viefgh, vw, 0);
          vaccijkln0 = vmlaq_lane_f32(vaccijkln0, viijkl, vw, 0);
          vaccmnopn0 = vmlaq_lane_f32(vaccmnopn0, vimnop, vw, 0);
          vaccqrstn0 = vmlaq_lane_f32(vaccqrstn0, viqrst, vw, 0);
          vaccuvwxn0 = vmlaq_lane_f32(vaccuvwxn0, viuvwx, vw, 0);
          vaccyz01n0 = vmlaq_lane_f32(vaccyz01n0, viyz01, vw, 0);
          vacc0123n1 = vmlaq_lane_f32(vacc0123n1, vi0123, vw, 1);
          vacc4567n1 = vmlaq_lane_f32(vacc4567n1, vi4567, vw, 1);
          vacc89ABn1 = vmlaq_lane_f32(vacc89ABn1, vi89AB, vw, 1);
          vaccCDEFn1 = vmlaq_lane_f32(vaccCDEFn1, viCDEF, vw, 1);
          vaccGHIJn1 = vmlaq_lane_f32(vaccGHIJn1, viGHIJ, vw, 1);
          vaccKLMNn1 = vmlaq_lane_f32(vaccKLMNn1, viKLMN, vw, 1);
          vaccOPQRn1 = vmlaq_lane_f32(vaccOPQRn1, viOPQR, vw, 1);
          vaccSTUVn1 = vmlaq_lane_f32(vaccSTUVn1, viSTUV, vw, 1);
          vaccWXYZn1 = vmlaq_lane_f32(vaccWXYZn1, viWXYZ, vw, 1);
          vaccabcdn1 = vmlaq_lane_f32(vaccabcdn1, viabcd, vw, 1);
          vaccefghn1 = vmlaq_lane_f32(vaccefghn1, viefgh, vw, 1);
          vaccijkln1 = vmlaq_lane_f32(vaccijkln1, viijkl, vw, 1);
          vaccmnopn1 = vmlaq_lane_f32(vaccmnopn1, vimnop, vw, 1);
          vaccqrstn1 = vmlaq_lane_f32(vaccqrstn1, viqrst, vw, 1);
          vaccuvwxn1 = vmlaq_lane_f32(vaccuvwxn1, viuvwx, vw, 1);
          vaccyz01n1 = vmlaq_lane_f32(vaccyz01n1, viyz01, vw, 1);
        } while (--nnz != 0);
      }
      // Only process for n0
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
      n -= 1;
    }
    // For remained blocked rows
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
      size_t n = nc - 1;

      // For the first blocked row
      float32x4_t vacc0123n0;
      float32x4_t vacc4567n0;
      float32x4_t vacc89ABn0;
      float32x4_t vaccCDEFn0;
      float32x4_t vaccGHIJn0;
      float32x4_t vaccKLMNn0;
      float32x4_t vaccOPQRn0;
      float32x4_t vaccSTUVn0;
      float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n1 = vacc0123n1;
      float32x4_t vacc89ABn1 = vacc0123n1;
      float32x4_t vaccCDEFn1 = vacc0123n1;
      float32x4_t vaccGHIJn1 = vacc0123n1;
      float32x4_t vaccKLMNn1 = vacc0123n1;
      float32x4_t vaccOPQRn1 = vacc0123n1;
      float32x4_t vaccSTUVn1 = vacc0123n1;

      while (n != 0) {
        uint32_t nnz = *nnzmap++;

        // Temporary output pipelining
        vacc0123n0 = vacc0123n1;
        vacc4567n0 = vacc4567n1;
        vacc89ABn0 = vacc89ABn1;
        vaccCDEFn0 = vaccCDEFn1;
        vaccGHIJn0 = vaccGHIJn1;
        vaccKLMNn0 = vaccKLMNn1;
        vaccOPQRn0 = vaccOPQRn1;
        vaccSTUVn0 = vaccSTUVn1;
        vacc0123n1 = vld1q_dup_f32(w); w += 1;
        vacc4567n1 = vacc0123n1;
        vacc89ABn1 = vacc0123n1;
        vaccCDEFn1 = vacc0123n1;
        vaccGHIJn1 = vacc0123n1;
        vaccKLMNn1 = vacc0123n1;
        vaccOPQRn1 = vacc0123n1;
        vaccSTUVn1 = vacc0123n1;

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
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0123n0 = vmlaq_lane_f32(vacc0123n0, vi0123, vw, 0);
            vacc4567n0 = vmlaq_lane_f32(vacc4567n0, vi4567, vw, 0);
            vacc89ABn0 = vmlaq_lane_f32(vacc89ABn0, vi89AB, vw, 0);
            vaccCDEFn0 = vmlaq_lane_f32(vaccCDEFn0, viCDEF, vw, 0);
            vaccGHIJn0 = vmlaq_lane_f32(vaccGHIJn0, viGHIJ, vw, 0);
            vaccKLMNn0 = vmlaq_lane_f32(vaccKLMNn0, viKLMN, vw, 0);
            vaccOPQRn0 = vmlaq_lane_f32(vaccOPQRn0, viOPQR, vw, 0);
            vaccSTUVn0 = vmlaq_lane_f32(vaccSTUVn0, viSTUV, vw, 0);
            vacc0123n1 = vmlaq_lane_f32(vacc0123n1, vi0123, vw, 1);
            vacc4567n1 = vmlaq_lane_f32(vacc4567n1, vi4567, vw, 1);
            vacc89ABn1 = vmlaq_lane_f32(vacc89ABn1, vi89AB, vw, 1);
            vaccCDEFn1 = vmlaq_lane_f32(vaccCDEFn1, viCDEF, vw, 1);
            vaccGHIJn1 = vmlaq_lane_f32(vaccGHIJn1, viGHIJ, vw, 1);
            vaccKLMNn1 = vmlaq_lane_f32(vaccKLMNn1, viKLMN, vw, 1);
            vaccOPQRn1 = vmlaq_lane_f32(vaccOPQRn1, viOPQR, vw, 1);
            vaccSTUVn1 = vmlaq_lane_f32(vaccSTUVn1, viSTUV, vw, 1);
          } while (--nnz != 0);
        }
        // Only process for n0
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
        float32x4_t vout89ABn0 = vminq_f32(vacc89ABn0, vmax);
        float32x4_t voutCDEFn0 = vminq_f32(vaccCDEFn0, vmax);
        float32x4_t voutGHIJn0 = vminq_f32(vaccGHIJn0, vmax);
        float32x4_t voutKLMNn0 = vminq_f32(vaccKLMNn0, vmax);
        float32x4_t voutOPQRn0 = vminq_f32(vaccOPQRn0, vmax);
        float32x4_t voutSTUVn0 = vminq_f32(vaccSTUVn0, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout4567n0 = vmaxq_f32(vout4567n0, vmin);
        vout89ABn0 = vmaxq_f32(vout89ABn0, vmin);
        voutCDEFn0 = vmaxq_f32(voutCDEFn0, vmin);
        voutGHIJn0 = vmaxq_f32(voutGHIJn0, vmin);
        voutKLMNn0 = vmaxq_f32(voutKLMNn0, vmin);
        voutOPQRn0 = vmaxq_f32(voutOPQRn0, vmin);
        voutSTUVn0 = vmaxq_f32(voutSTUVn0, vmin);

        vst1q_f32(output + 0, vout0123n0);
        vst1q_f32(output + 4, vout4567n0);
        vst1q_f32(output + 8, vout89ABn0);
        vst1q_f32(output + 12, voutCDEFn0);
        vst1q_f32(output + 16, voutGHIJn0);
        vst1q_f32(output + 20, voutKLMNn0);
        vst1q_f32(output + 24, voutOPQRn0);
        vst1q_f32(output + 28, voutSTUVn0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      }
      // For remained blocked rows
      float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
      float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
      float32x4_t vout89ABn1 = vminq_f32(vacc89ABn1, vmax);
      float32x4_t voutCDEFn1 = vminq_f32(vaccCDEFn1, vmax);
      float32x4_t voutGHIJn1 = vminq_f32(vaccGHIJn1, vmax);
      float32x4_t voutKLMNn1 = vminq_f32(vaccKLMNn1, vmax);
      float32x4_t voutOPQRn1 = vminq_f32(vaccOPQRn1, vmax);
      float32x4_t voutSTUVn1 = vminq_f32(vaccSTUVn1, vmax);

      vout0123n1 = vmaxq_f32(vout0123n1, vmin);
      vout4567n1 = vmaxq_f32(vout4567n1, vmin);
      vout89ABn1 = vmaxq_f32(vout89ABn1, vmin);
      voutCDEFn1 = vmaxq_f32(voutCDEFn1, vmin);
      voutGHIJn1 = vmaxq_f32(voutGHIJn1, vmin);
      voutKLMNn1 = vmaxq_f32(voutKLMNn1, vmin);
      voutOPQRn1 = vmaxq_f32(voutOPQRn1, vmin);
      voutSTUVn1 = vmaxq_f32(voutSTUVn1, vmin);

      vst1q_f32(output + 0, vout0123n1);
      vst1q_f32(output + 4, vout4567n1);
      vst1q_f32(output + 8, vout89ABn1);
      vst1q_f32(output + 12, voutCDEFn1);
      vst1q_f32(output + 16, voutGHIJn1);
      vst1q_f32(output + 20, voutKLMNn1);
      vst1q_f32(output + 24, voutOPQRn1);
      vst1q_f32(output + 28, voutSTUVn1);
      output = (float*restrict) ((uintptr_t) output + output_stride);

      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 32;
    }
    output_decrement += 16 * sizeof(float);
    if (mc & (16 * sizeof(float))) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc - 1;

      // For the first blocked row
      float32x4_t vacc0123n0;
      float32x4_t vacc4567n0;
      float32x4_t vacc89ABn0;
      float32x4_t vaccCDEFn0;
      float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n1 = vacc0123n1;
      float32x4_t vacc89ABn1 = vacc0123n1;
      float32x4_t vaccCDEFn1 = vacc0123n1;

      while (n != 0) {
        uint32_t nnz = *nnzmap++;

        // Temporary output pipelining
        vacc0123n0 = vacc0123n1;
        vacc4567n0 = vacc4567n1;
        vacc89ABn0 = vacc89ABn1;
        vaccCDEFn0 = vaccCDEFn1;
        vacc0123n1 = vld1q_dup_f32(w); w += 1;
        vacc4567n1 = vacc0123n1;
        vacc89ABn1 = vacc0123n1;
        vaccCDEFn1 = vacc0123n1;

        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0123n0 = vmlaq_lane_f32(vacc0123n0, vi0123, vw, 0);
            vacc4567n0 = vmlaq_lane_f32(vacc4567n0, vi4567, vw, 0);
            vacc89ABn0 = vmlaq_lane_f32(vacc89ABn0, vi89AB, vw, 0);
            vaccCDEFn0 = vmlaq_lane_f32(vaccCDEFn0, viCDEF, vw, 0);
            vacc0123n1 = vmlaq_lane_f32(vacc0123n1, vi0123, vw, 1);
            vacc4567n1 = vmlaq_lane_f32(vacc4567n1, vi4567, vw, 1);
            vacc89ABn1 = vmlaq_lane_f32(vacc89ABn1, vi89AB, vw, 1);
            vaccCDEFn1 = vmlaq_lane_f32(vaccCDEFn1, viCDEF, vw, 1);
          } while (--nnz != 0);
        }
        // Only process for n0
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
        float32x4_t vout89ABn0 = vminq_f32(vacc89ABn0, vmax);
        float32x4_t voutCDEFn0 = vminq_f32(vaccCDEFn0, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout4567n0 = vmaxq_f32(vout4567n0, vmin);
        vout89ABn0 = vmaxq_f32(vout89ABn0, vmin);
        voutCDEFn0 = vmaxq_f32(voutCDEFn0, vmin);

        vst1q_f32(output + 0, vout0123n0);
        vst1q_f32(output + 4, vout4567n0);
        vst1q_f32(output + 8, vout89ABn0);
        vst1q_f32(output + 12, voutCDEFn0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      }
      // For remained blocked rows
      float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
      float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
      float32x4_t vout89ABn1 = vminq_f32(vacc89ABn1, vmax);
      float32x4_t voutCDEFn1 = vminq_f32(vaccCDEFn1, vmax);

      vout0123n1 = vmaxq_f32(vout0123n1, vmin);
      vout4567n1 = vmaxq_f32(vout4567n1, vmin);
      vout89ABn1 = vmaxq_f32(vout89ABn1, vmin);
      voutCDEFn1 = vmaxq_f32(voutCDEFn1, vmin);

      vst1q_f32(output + 0, vout0123n1);
      vst1q_f32(output + 4, vout4567n1);
      vst1q_f32(output + 8, vout89ABn1);
      vst1q_f32(output + 12, voutCDEFn1);
      output = (float*restrict) ((uintptr_t) output + output_stride);

      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 16;
    }
    output_decrement += 8 * sizeof(float);
    if (mc & (8 * sizeof(float))) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc - 1;

      // For the first blocked row
      float32x4_t vacc0123n0;
      float32x4_t vacc4567n0;
      float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n1 = vacc0123n1;

      while (n != 0) {
        uint32_t nnz = *nnzmap++;

        // Temporary output pipelining
        vacc0123n0 = vacc0123n1;
        vacc4567n0 = vacc4567n1;
        vacc0123n1 = vld1q_dup_f32(w); w += 1;
        vacc4567n1 = vacc0123n1;

        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0123n0 = vmlaq_lane_f32(vacc0123n0, vi0123, vw, 0);
            vacc4567n0 = vmlaq_lane_f32(vacc4567n0, vi4567, vw, 0);
            vacc0123n1 = vmlaq_lane_f32(vacc0123n1, vi0123, vw, 1);
            vacc4567n1 = vmlaq_lane_f32(vacc4567n1, vi4567, vw, 1);
          } while (--nnz != 0);
        }
        // Only process for n0
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout4567n0 = vmaxq_f32(vout4567n0, vmin);

        vst1q_f32(output + 0, vout0123n0);
        vst1q_f32(output + 4, vout4567n0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      }
      // For remained blocked rows
      float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
      float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);

      vout0123n1 = vmaxq_f32(vout0123n1, vmin);
      vout4567n1 = vmaxq_f32(vout4567n1, vmin);

      vst1q_f32(output + 0, vout0123n1);
      vst1q_f32(output + 4, vout4567n1);
      output = (float*restrict) ((uintptr_t) output + output_stride);

      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 8;
    }
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc - 1;

      // For the first blocked row
      float32x4_t vacc0123n0;
      float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;

      while (n != 0) {
        uint32_t nnz = *nnzmap++;

        // Temporary output pipelining
        vacc0123n0 = vacc0123n1;
        vacc0123n1 = vld1q_dup_f32(w); w += 1;

        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0123n0 = vmlaq_lane_f32(vacc0123n0, vi0123, vw, 0);
            vacc0123n1 = vmlaq_lane_f32(vacc0123n1, vi0123, vw, 1);
          } while (--nnz != 0);
        }
        // Only process for n0
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);

        vst1q_f32(output + 0, vout0123n0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      }
      // For remained blocked rows
      float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);

      vout0123n1 = vmaxq_f32(vout0123n1, vmin);

      vst1q_f32(output + 0, vout0123n1);
      output = (float*restrict) ((uintptr_t) output + output_stride);

      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc - 1;

      // For the first blocked row
      float32x2_t vacc01n0;
      float32x2_t vacc01n1 = vld1_dup_f32(w); w += 1;

      while (n != 0) {
        uint32_t nnz = *nnzmap++;

        // Temporary output pipelining
        vacc01n0 = vacc01n1;
        vacc01n1 = vld1_dup_f32(w); w += 1;

        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi01 = vld1_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc01n0 = vmla_lane_f32(vacc01n0, vi01, vw, 0);
            vacc01n1 = vmla_lane_f32(vacc01n1, vi01, vw, 1);
          } while (--nnz != 0);
        }
        // Only process for n0
        float32x2_t vout01n0 = vmin_f32(vacc01n0, vget_low_f32(vmax));

        vout01n0 = vmax_f32(vout01n0, vget_low_f32(vmin));

        vst1_f32(output + 0, vout01n0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      }
      // For remained blocked rows
      float32x2_t vout01n1 = vmin_f32(vacc01n1, vget_low_f32(vmax));

      vout01n1 = vmax_f32(vout01n1, vget_low_f32(vmin));

      vst1_f32(output + 0, vout01n1);
      output = (float*restrict) ((uintptr_t) output + output_stride);

      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc - 1;

      // For the first blocked row
      float32x2_t vacc0n0;
      float32x2_t vacc0n1 = vld1_dup_f32(w); w += 1;

      while (n != 0) {
        uint32_t nnz = *nnzmap++;

        // Temporary output pipelining
        vacc0n0 = vacc0n1;
        vacc0n1 = vld1_dup_f32(w); w += 1;

        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi0 = vld1_dup_f32(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0n0 = vmla_lane_f32(vacc0n0, vi0, vw, 0);
            vacc0n1 = vmla_lane_f32(vacc0n1, vi0, vw, 1);
          } while (--nnz != 0);
        }
        // Only process for n0
        float32x2_t vout0n0 = vmin_f32(vacc0n0, vget_low_f32(vmax));

        vout0n0 = vmax_f32(vout0n0, vget_low_f32(vmin));

        vst1_lane_f32(output + 0, vout0n0, 0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
        n -= 1;
      }
      // For remained blocked rows
      float32x2_t vout0n1 = vmin_f32(vacc0n1, vget_low_f32(vmax));

      vout0n1 = vmax_f32(vout0n1, vget_low_f32(vmin));

      vst1_lane_f32(output + 0, vout0n1, 0);
      output = (float*restrict) ((uintptr_t) output + output_stride);

      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
    }
}
