//==----- lsc_load_2d_compare.cpp - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The tests makes sure old and new load_2d API produce identical
// results.
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/usm.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;
template <typename T> bool test() {
  sycl::queue Q(sycl::gpu_selector_v);
  auto dev = Q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  constexpr int TM = 8;
  constexpr int TN = 8;
  constexpr int NBLOCKS = 2;
  constexpr int WIDTH = 2 * TN;
  constexpr int HEIGHT = TM;
  constexpr int PITCH = WIDTH;
  constexpr int SIZE = WIDTH * HEIGHT;

  auto *A = malloc_shared<T>(SIZE, Q);
  auto *B = malloc_shared<T>(SIZE, Q);
  auto *C = malloc_shared<T>(SIZE, Q);
  auto *C1 = malloc_shared<T>(SIZE, Q);

  for (int i = 0; i < SIZE; i++) {
    A[i] = static_cast<T>(i);
  }

  Q.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1>
                                                  item) SYCL_ESIMD_KERNEL {
     config_2d_mem_access<T, TN, TM, NBLOCKS> my_config(
         A, WIDTH * sizeof(T) - 1, HEIGHT - 1, PITCH * sizeof(T) - 1, 0, 0);

     simd<T, NBLOCKS * TM * TN> tmp =
         lsc_load_2d<T, TN, TM, NBLOCKS, false, false>(my_config);
     simd<T, NBLOCKS * TM * TN> tmp1 = lsc_load_2d<T, TN, TM, NBLOCKS>(
         my_config.get_data_pointer(), my_config.get_surface_width(),
         my_config.get_surface_height(), my_config.get_surface_pitch(),
         my_config.get_x(), my_config.get_y());

     tmp.copy_to(C);
     tmp1.copy_to(C1);
   }).wait();

  bool error = false;
  for (auto i = 0; i < SIZE; ++i)
    error |= C[i] != C1[i];

  free(A, Q);
  free(C, Q);
  free(C1, Q);
  return error;
}

int main() {
  bool result = false;
  result |= test<float>();
  result |= test<uint32_t>();
  result |= test<uint16_t>();
  result |= test<uint8_t>();
  result |= test<sycl::half>();

  std::cout << (result ? "FAILED" : "passed") << std::endl;
  return 0;
}
