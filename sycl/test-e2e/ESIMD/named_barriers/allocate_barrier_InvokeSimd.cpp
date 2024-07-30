//==-- allocate_barrier_InvokeSimd.cpp - DPC++ ESIMD on-device test ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// NOTE: named barrier supported only since PVC
// REQUIRES: arch-intel_gpu_pvc
//
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// vISA LTO run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/*
 * Test checks basic support for allocate_named_barrier in invoke_simd context.
 */

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

/* Subgroup size attribute is optional
 * In case it is absent compiler decides what subgroup size to use
 */
#ifdef IMPL_SUBGROUP
#define SUBGROUP_ATTR
#else
#define SUBGROUP_ATTR [[intel::reqd_sub_group_size(VL)]]
#endif

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;
namespace experimental_esimd = sycl::ext::intel::experimental::esimd;

constexpr int Size = 4096;
constexpr int VL = 16;

class Test {};

ESIMD_INLINE void ESIMD_CALLEE_nbarrier(nd_item<1> *ndi) SYCL_ESIMD_FUNCTION {
  const uint8_t BARNUM = 32;
  esimd::named_barrier_init<1>();

  uint8_t barrier_id = __ESIMD_ENS::named_barrier_allocate<BARNUM>();
  uint8_t producer_consumer_mode = 0;
  uint32_t num_producers = 16;
  uint32_t num_consumers = 16;
  __ESIMD_NS::named_barrier_signal(barrier_id, producer_consumer_mode,
                                   num_producers, num_consumers);
  __ESIMD_NS::named_barrier_wait(barrier_id);
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall SIMD_CALLEE_nbarrier(
    nd_item<1> *ndi) SYCL_ESIMD_FUNCTION;

int main(void) {
  queue Queue;
  auto Device = Queue.get_device();

  std::cout << "Running on " << Device.get_info<sycl::info::device::name>()
            << "\n";
  try {
    // We need that many workgroups
    sycl::range<1> GlobalRange{16 * 16};
    // We need that many threads in each group
    sycl::range<1> LocalRange{16 * 16};

    auto e = Queue.submit([&](handler &cgh) {
      cgh.parallel_for<Test>(nd_range<1>(GlobalRange, LocalRange),
                             [=](nd_item<1> item) SUBGROUP_ATTR {
                               sycl::group<1> g = item.get_group();
                               sycl::sub_group sg = item.get_sub_group();
                               invoke_simd(sg, SIMD_CALLEE_nbarrier,
                                           uniform{&item});
                             });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
  }

  return 0;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall SIMD_CALLEE_nbarrier(
    nd_item<1> *ndi) SYCL_ESIMD_FUNCTION {
  ESIMD_CALLEE_nbarrier(ndi);
  return;
}
