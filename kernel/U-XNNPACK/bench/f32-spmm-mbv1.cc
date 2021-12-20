// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/f32-spmm-mbv1.h"
#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>
#include <xnnpack/spmm.h>


static void SpMMBenchmark(benchmark::State& state,
  xnn_f32_spmm_minmax_ukernel_function spmm, uint32_t mr, uint32_t nr, float sparsity,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  // if using blocks, generate the reduced matrix first and then extrude along
  // the block dimension (n), to get the full matrix
  size_t ncols = nc / nr + nc % nr;
  std::vector<float> b(ncols * kc);
  std::vector<float> bias(nc);
  std::vector<float> w;
  std::vector<uint32_t> nmap;
  std::vector<int32_t> dmap;
  const size_t sparse_end = std::min(size_t(float(b.size()) * sparsity), b.size());
  const size_t num_nonzeroes = nr * (b.size() - sparse_end);

  const size_t w_elements = num_nonzeroes + nc;
  const size_t c_elements = mc * nc;
  const size_t dmap_elements = num_nonzeroes / nr;
  const size_t nmap_elements = nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements) + sizeof(uint32_t) * (dmap_elements + nmap_elements));

  // Micro-kernel can access one element beyond w and dmap for software pipelining.
  w.reserve(num_buffers * w_elements + 1);
  dmap.reserve(num_buffers * dmap_elements + 1);
  nmap.resize(num_buffers * nmap_elements);

  std::vector<size_t> a_offsets(num_buffers);

  for (size_t buffer_index = 0; buffer_index < num_buffers; buffer_index++) {
    // Re-generate weights. Note: each re-generation produces the number of non-zeroes.
    std::fill(b.begin(), b.begin() + sparse_end, 0.0f);
    std::generate(b.begin() + sparse_end, b.end(), std::ref(f32rng));
    std::shuffle(b.begin(), b.end(), rng);
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));

    uint32_t first_j = 0, last_j = 0;
    bool is_first_nonzero = true;
    for (uint32_t i = 0; i < nc / nr; i++) {
      for (uint32_t n = 0; n < nr; n++)
        w.push_back(bias[nr * i + n]);
      for (uint32_t j = 0; j < kc; j++) {
        if (b[i * kc + j] != 0.0f) {
          for (size_t l = 0; l < nr; l++)
            w.push_back(b[i * kc + j] + static_cast<float>(i));
          if (is_first_nonzero) {
            first_j = j;
          } else {
            const ptrdiff_t increment = int32_t(j - last_j) * int32_t(mc) * int32_t(sizeof(float));
            dmap.push_back(increment);
          }
          last_j = j;
          is_first_nonzero = false;
          nmap[buffer_index * nmap_elements + i] += 1;
        }
      }
    }
    for (uint32_t i = nc / nr; i < ncols; i++) {
      w.push_back(bias[i]);
      for (uint32_t j = 0; j < kc; j++) {
        if (b[i * kc + j] != 0.0f) {
          w.push_back(b[i * kc + j]);
          if (is_first_nonzero) {
            first_j = j;
          } else {
            const ptrdiff_t increment = int32_t(j - last_j) * int32_t(mc) * int32_t(sizeof(float));
            dmap.push_back(increment);
          }
          last_j = j;
          is_first_nonzero = false;
          nmap[buffer_index * nmap_elements + i] += 1;
        }
      }
    }
    {
      const ptrdiff_t increment = int32_t(first_j - last_j) * int32_t(mc) * int32_t(sizeof(float));
      dmap.push_back(increment);
    }

    a_offsets[buffer_index] = first_j * mc;
  }

  // Micro-kernel can access one element beyond w and dmap for software pipelining.
  w.resize(w.size() + 1);
  dmap.resize(dmap.size() + 1);

  std::vector<float, AlignedAllocator<float, 64>> a(kc * mc);
  std::vector<float, AlignedAllocator<float, 64>> c(num_buffers * c_elements);

  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::fill(c.begin(), c.end(), nanf(""));

  xnn_f32_minmax_params params;
  xnn_init_f32_minmax_params(&params, -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W, Kmap, and Nmap is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    spmm(mc * sizeof(float), nc,
      a.data() + a_offsets[buffer_index],
      w.data() + buffer_index * w_elements,
      dmap.data() + buffer_index * dmap_elements,
      nmap.data() + buffer_index * nmap_elements,
      c.data() + buffer_index * c_elements, mc * sizeof(float),
      &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * num_nonzeroes, benchmark::Counter::kIsRate);

  state.counters["EffFLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void spmm80_16x1__neon(benchmark::State& state, const char* net) {
    SpMMBenchmark(state, xnn_f32_spmm_minmax_ukernel_16x1__neon, 32, 1, 0.8f, benchmark::utils::CheckNEON);
  }

  static void spmm80_16x2__neon(benchmark::State& state, const char* net) {
    SpMMBenchmark(state, xnn_f32_spmm_minmax_ukernel_16x2__neon, 32, 4, 0.8f, benchmark::utils::CheckNEON);
  }

  static void spmm80_16x4__neon(benchmark::State& state, const char* net) {
    SpMMBenchmark(state, xnn_f32_spmm_minmax_ukernel_16x4__neon, 32, 4, 0.8f, benchmark::utils::CheckNEON);
  }
  static void spmm80_32x1__neon(benchmark::State& state, const char* net) {
    SpMMBenchmark(state, xnn_f32_spmm_minmax_ukernel_32x1__neon, 32, 1, 0.8f, benchmark::utils::CheckNEON);
  }

  static void spmm80_32x2__neon(benchmark::State& state, const char* net) {
    SpMMBenchmark(state, xnn_f32_spmm_minmax_ukernel_32x2__neon, 32, 4, 0.8f, benchmark::utils::CheckNEON);
  }

  static void spmm80_32x4__neon(benchmark::State& state, const char* net) {
    SpMMBenchmark(state, xnn_f32_spmm_minmax_ukernel_32x4__neon, 32, 4, 0.8f, benchmark::utils::CheckNEON);
  }

  static void spmm80_64x2__neon(benchmark::State& state, const char* net) {
    SpMMBenchmark(state, xnn_f32_spmm_minmax_ukernel_64x2__neon, 32, 4, 0.8f, benchmark::utils::CheckNEON);
  }

  static void spmm80_64x4__neon(benchmark::State& state, const char* net) {
    SpMMBenchmark(state, xnn_f32_spmm_minmax_ukernel_64x4__neon, 32, 4, 0.8f, benchmark::utils::CheckNEON);
  }


  BENCHMARK_SPMM(spmm80_16x1__neon)
  BENCHMARK_SPMM(spmm80_16x2__neon)
  BENCHMARK_SPMM(spmm80_16x4__neon)
  BENCHMARK_SPMM(spmm80_32x1__neon)
  BENCHMARK_SPMM(spmm80_32x2__neon)
  BENCHMARK_SPMM(spmm80_32x4__neon)
  BENCHMARK_SPMM(spmm80_64x2__neon)
  BENCHMARK_SPMM(spmm80_64x4__neon)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
