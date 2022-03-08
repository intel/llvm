//==------- esimd_pack_unpack_mask.cpp  - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented 'single_task()' method
// XFAIL: esimd_emulator
// TODO: fails on OpenCL - https://github.com/intel/llvm-test-suite/issues/901
// UNSUPPORTED: opencl
// RUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Smoke test for the esimd pack_mask/unpack_mask APIs.

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::intel::experimental::esimd;

template <int, int> struct test_id;
using MaskRawElemT = typename simd_mask<1>::raw_element_type;
static inline constexpr int MAX_N = 32;

// The main test routine.
template <int N, int TestCase, class InitF>
bool test_impl(queue q, const char *title, InitF init_f) {
  std::cout << "Testing N=" << N << ", " << title << "...\n";

  MaskRawElemT *test_data = sycl::malloc_shared<MaskRawElemT>(N, q);
  init_f(test_data);
  uint32_t *res_packed = sycl::malloc_shared<uint32_t>(1, q);
  MaskRawElemT *res_unpacked = sycl::malloc_shared<MaskRawElemT>(MAX_N, q);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.single_task<test_id<N, TestCase>>([=]() SYCL_ESIMD_KERNEL {
        simd_mask<N> m(test_data);
        uint32_t packed_m = pack_mask(m);
        res_packed[0] = packed_m;
        unpack_mask<MAX_N>(packed_m).copy_to(res_unpacked);
      });
    });
    e.wait_and_throw();
  } catch (sycl::exception const &e) {
    std::cout << "  SYCL exception caught: " << e.what() << '\n';
    sycl::free(test_data, q);
    sycl::free(res_packed, q);
    sycl::free(res_unpacked, q);
    return false;
  }
  uint32_t gold_packed = 0;

  for (int i = 0; i < N; i++) {
    if (test_data[i] != 0) {
      gold_packed |= (1 << i);
    }
  }
  int err_cnt = 0;

  if (gold_packed != *res_packed) {
    ++err_cnt;
    std::cout << "    ERROR in pack_mask: 0x" << std::hex << (*res_packed)
              << " != 0x" << gold_packed << std::dec << " [gold]\n";
  }
  for (unsigned i = 0; i < N; ++i) {
    if ((test_data[i] != 0) != (res_unpacked[i] == 1)) {
      ++err_cnt;
      std::cout << "    ERROR in lane " << i << ": (0x" << std::hex
                << test_data[i] << "!=0) != (0x" << res_unpacked[i] << std::dec
                << "==1) [gold]\n";
    }
  }
  for (unsigned i = N; i < MAX_N; ++i) {
    if (test_data[i] != 0) {
      ++err_cnt;
      std::cout << "    ERROR: non-zero lane " << i << ": 0x" << std::hex
                << test_data[i] << std::dec << "\n";
    }
  }
  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  sycl::free(test_data, q);
  sycl::free(res_packed, q);
  sycl::free(res_unpacked, q);
  return err_cnt > 0 ? false : true;
}

template <int N> bool test(queue q) {
  bool passed = true;
  passed &= test_impl<N, 0>(q, "all zero", [=](MaskRawElemT *x) {
    for (int i = 0; i < N; i++) {
      x[i] = 0;
    }
  });
  passed &= test_impl<N, 1>(q, "all one", [=](MaskRawElemT *x) {
    for (int i = 0; i < N; i++) {
      x[i] = 1;
    }
  });
  passed &= test_impl<N, 2>(q, "misc", [=](MaskRawElemT *x) {
    for (int i = 0; i < N; i++) {
      x[i] = i % 3;
    }
  });
  return passed;
}

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test<1>(q);
  passed &= test<2>(q);
  passed &= test<7>(q);
  passed &= test<8>(q);
  passed &= test<16>(q);
  // TODO disabled due to compiler bug
  // passed &= test<31>(q);
  // passed &= test<32>(q);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
