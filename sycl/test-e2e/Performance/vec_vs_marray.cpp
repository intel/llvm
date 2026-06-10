//==------- vec_vs_marray.cpp --- sycl::vec vs sycl::marray performance ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Performance comparison between sycl::vec and sycl::marray.
//
// Motivation
//   sycl::marray (SYCL 2020 sec. 4.14.3) is specified as a plain array-like
//   type with no over-alignment requirement, whereas sycl::vec (SYCL 2020 sec.
//   4.14.2) is over-aligned to the size of the whole vector. When user code is
//   migrated from sycl::vec to sycl::marray the relaxed alignment can pessimize
//   vectorized memory accesses, which is especially visible for 16-bit element
//   types such as sycl::half. This test measures both containers side by side
//   so that such differences become visible.
//
// What is compared
//   The timed kernels use only the subset of the API that is common to both
//   containers, so the exact same kernel body is instantiated for each and any
//   timing difference is attributable to the container type alone:
//     * construction from a scalar    vec(const T&) / marray(const T&)
//     * copy/load                     C v = in[i];
//     * store/assignment              out[i] = v;
//     * operator[] / size()           element access (smoke + init)
//     * arithmetic operators + - *    v * a + a
//     * compound assignment           v -= a
//
//   sycl::vec only supports 1, 2, 3, 4, 8 and 16 components, while sycl::marray
//   can have an arbitrary number of components. Sizes supported by both are
//   compared head to head; sizes that only marray supports are reported on
//   their own ("marray-only").
//
// The test is informational for performance: it prints the per-(type, size)
// kernel times for both containers and highlights rows where the marray/vec
// ratio exceeds threshold. It never fails on a performance difference.
//
//
// RUN: %{build} -o %t_non_prev.out
// RUN: %{build} -fpreview-breaking-changes -o %t_prev.out
// RUN: %{run} %t_non_prev.out
// RUN: %{run} %t_prev.out
//
// UNSUPPORTED: linux || windows
// UNSUPPORTED-INTENDED: This test is intended to be run manually to compare
// the performance of vec and marray. It doesn't check or assert anything.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/marray.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>
#include <sycl/vector.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace sycl;

namespace {

// -- problem size / measurement parameters ---------------------------------
constexpr size_t NumElems = 1 << 18; // work-items per kernel launch
constexpr int ComputeIters = 64;     // arithmetic iterations (compute kernel)
constexpr int Warmup = 2;            // un-timed launches
constexpr int Repeats = 10;          // timed launches; median is kept

// marray/vec ratio above which a row is highlighted (purely informational).
constexpr double threshold = 1.5;

// -- container introspection ------------------------------------------------
// Extracts the element type from a vec or marray container type.
template <typename ContainerT> struct ContainerElement;
template <typename ElementT, int NumComponents>
struct ContainerElement<vec<ElementT, NumComponents>> {
  using type = ElementT;
};
template <typename ElementT, size_t NumComponents>
struct ContainerElement<marray<ElementT, NumComponents>> {
  using type = ElementT;
};
template <typename ContainerT>
using ContainerElementT = typename ContainerElement<ContainerT>::type;

enum class BenchKind { Stream, Compute };

// One timed launch. Returns the device-side kernel duration in nanoseconds as
// reported by event profiling.
template <BenchKind Mode, typename ContainerT>
double launchOnce(queue &q, const ContainerT *in, ContainerT *out,
                  [[maybe_unused]] ContainerElementT<ContainerT> one) {
  event e = q.parallel_for(range<1>(NumElems), [=](id<1> idx) {
    const size_t i = idx[0];
    const ContainerT a = in[i];
    if constexpr (Mode == BenchKind::Stream) {
      // Load + vector add + store: sensitive to element alignment.
      out[i] = a + ContainerT(one);
    } else {
      // Arithmetic heavy: sensitive to per-element compute throughput.
      ContainerT acc = a;
      for (int k = 0; k < ComputeIters; ++k) {
        acc = acc * a + a; // multiply + add
        acc -= a;          // compound subtract
      }
      out[i] = acc;
    }
  });
  e.wait_and_throw();
  const auto s = e.get_profiling_info<info::event_profiling::command_start>();
  const auto f = e.get_profiling_info<info::event_profiling::command_end>();
  return static_cast<double>(f - s);
}

// Warm-up + timed launches for one container type and bench kind. Returns the
// median kernel time in nanoseconds, or -1 on allocation failure (so over-sized
// configurations are skipped instead of aborting).
template <BenchKind Mode, typename ContainerT> double bench(queue &q) {
  using ElementT = ContainerElementT<ContainerT>;
  const ElementT one = static_cast<ElementT>(static_cast<float>(1));

  ContainerT *in = nullptr;
  ContainerT *out = nullptr;
  try {
    in = malloc_device<ContainerT>(NumElems, q);
    out = malloc_device<ContainerT>(NumElems, q);
    if (!in || !out)
      throw std::runtime_error("allocation returned null");
  } catch (const std::exception &) {
    if (in)
      sycl::free(in, q);
    if (out)
      sycl::free(out, q);
    return -1.0;
  }

  // Initialise input with index-dependent values to defeat constant folding.
  q.parallel_for(range<1>(NumElems), [=](id<1> idx) {
     const size_t i = idx[0];
     in[i] = ContainerT(static_cast<ElementT>(static_cast<float>((i % 7) + 1)));
   }).wait_and_throw();

  for (int w = 0; w < Warmup; ++w)
    (void)launchOnce<Mode>(q, in, out, one);

  std::array<double, Repeats> samples;
  for (int r = 0; r < Repeats; ++r)
    samples[r] = launchOnce<Mode>(q, in, out, one);

  sycl::free(in, q);
  sycl::free(out, q);

  // Median of the timed launches: robust to outliers while still representative
  // of typical (not just best-case) performance.
  std::sort(samples.begin(), samples.end());
  if constexpr (Repeats % 2 == 1)
    return samples[Repeats / 2];
  else
    return 0.5 * (samples[Repeats / 2 - 1] + samples[Repeats / 2]);
}

// Compare vec<ElementT, NumComponents> against marray<ElementT, NumComponents>
// for one (type, size). Prints the timings and highlights the row when the
// marray/vec ratio exceeds threshold. Never affects the test's pass/fail
// status.
template <typename ElementT, int NumComponents>
void compareVecAndMarray(queue &q, const char *typeName) {
  using VecT = vec<ElementT, NumComponents>;
  using MarrayT = marray<ElementT, static_cast<size_t>(NumComponents)>;

  const double vStream = bench<BenchKind::Stream, VecT>(q);
  const double mStream = bench<BenchKind::Stream, MarrayT>(q);
  const double vCompute = bench<BenchKind::Compute, VecT>(q);
  const double mCompute = bench<BenchKind::Compute, MarrayT>(q);

  if (vStream < 0 || mStream < 0 || vCompute < 0 || mCompute < 0) {
    std::cout << "  " << std::left << std::setw(8) << typeName << " x"
              << NumComponents << "   <skipped: allocation failed>\n";
    return;
  }

  const double eps = 1.0;
  const double rStream = mStream / std::max(vStream, eps);
  const double rCompute = mCompute / std::max(vCompute, eps);
  const bool highlight = rStream > threshold || rCompute > threshold;

  std::cout << "  " << std::left << std::setw(8) << typeName << "x"
            << std::setw(3) << NumComponents << std::right << std::fixed
            << std::setprecision(2) << "  stream(vec/marr ns):" << std::setw(12)
            << vStream << " /" << std::setw(12) << mStream << "  x"
            << std::setw(6) << rStream << "   compute:" << std::setw(12)
            << vCompute << " /" << std::setw(12) << mCompute << "  x"
            << std::setw(6) << rCompute
            << (highlight ? "   <== LARGE DIFFERENCE" : "") << "\n";
}

// Benchmark marray<ElementT, NumComponents> for a size that vec cannot
// represent. Only marray timings are printed; there is nothing to compare
// against.
template <typename ElementT, size_t NumComponents>
void benchMarrayOnly(queue &q, const char *typeName) {
  using MarrayT = marray<ElementT, NumComponents>;

  const double mStream = bench<BenchKind::Stream, MarrayT>(q);
  const double mCompute = bench<BenchKind::Compute, MarrayT>(q);

  if (mStream < 0 || mCompute < 0) {
    std::cout << "  " << std::left << std::setw(8) << typeName << " x"
              << NumComponents
              << "   marray-only  <skipped: allocation failed>\n";
    return;
  }

  std::cout << "  " << std::left << std::setw(8) << typeName << "x"
            << std::setw(3) << NumComponents << std::right << std::fixed
            << std::setprecision(2)
            << "  marray-only  stream(ns):" << std::setw(12) << mStream
            << "   compute(ns):" << std::setw(12) << mCompute << "\n";
}

template <typename ElementT>
void benchAllSizes(queue &q, const char *typeName) {
  // Sizes supported by both vec and marray: compared head to head.
  compareVecAndMarray<ElementT, 2>(q, typeName);
  compareVecAndMarray<ElementT, 3>(q, typeName);
  compareVecAndMarray<ElementT, 4>(q, typeName);
  compareVecAndMarray<ElementT, 8>(q, typeName);
  compareVecAndMarray<ElementT, 16>(q, typeName);

  // Sizes only marray supports (vec is limited to 1, 2, 3, 4, 8, 16):
  // marray timings reported on their own.
  benchMarrayOnly<ElementT, 5>(q, typeName);
  benchMarrayOnly<ElementT, 6>(q, typeName);
  benchMarrayOnly<ElementT, 7>(q, typeName);
  benchMarrayOnly<ElementT, 32>(q, typeName);
}

} // namespace

int main() {
  queue q{property::queue::enable_profiling{}};
  device dev = q.get_device();

  std::cout << "Device: " << dev.get_info<info::device::name>() << "\n";
  std::cout << "Highlight threshold: x" << threshold << "\n";
  std::cout << "Lower ns is better; marray/vec ratio > threshold is "
               "highlighted.\n\n";

  benchAllSizes<int8_t>(q, "int8");
  benchAllSizes<int16_t>(q, "int16");
  benchAllSizes<int32_t>(q, "int32");
  benchAllSizes<int64_t>(q, "int64");
  if (dev.has(aspect::fp16))
    benchAllSizes<half>(q, "half");
  benchAllSizes<float>(q, "float");
  if (dev.has(aspect::fp64))
    benchAllSizes<double>(q, "double");
  benchAllSizes<sycl::ext::oneapi::bfloat16>(q, "bf16");

  return 0;
}