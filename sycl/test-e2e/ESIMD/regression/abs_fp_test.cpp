// REQUIRES: gpu-intel-dg2 || arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//==- abs_fp_test.cpp - Test for abs function -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <sycl/usm.hpp>
#include <sycl/usm/usm_allocator.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using bf16 = sycl::ext::oneapi::bfloat16;
using tfloat32 = sycl::ext::intel::experimental::esimd::tfloat32;

template <typename DataT>
using shared_allocator = sycl::usm_allocator<DataT, sycl::usm::alloc::shared>;
template <typename DataT>
using shared_vector = std::vector<DataT, shared_allocator<DataT>>;

template <typename T, int N> bool test(sycl::queue &Queue, T testValue) {
  shared_allocator<T> Allocator(Queue);

  shared_vector<T> Output(N, 0, Allocator);

  auto *OutputPtr = Output.data();

  Queue.submit([&](sycl::handler &cgh) {
    auto Kernel = ([=]() SYCL_ESIMD_KERNEL {
      simd<T, N> Input = testValue;
      simd<T, N> Result = __ESIMD_NS::abs(Input);
      Result.copy_to(OutputPtr);
    });
    cgh.single_task(Kernel);
  });
  Queue.wait();

  for (int I = 0; I < N; I++) {
    if (std::abs(testValue) != Output[I]) {
      std::cout << "Incorrect value at index " << I << " "
                << std::abs(testValue) << " != " << Output[I] << std::endl;
      return false;
    }
  }

  return true;
}

int main() {

  bool Pass = true;
  sycl::queue Q;
  Pass &= test<bf16, 8>(Q, -1);
  Pass &= test<tfloat32, 8>(Q, -1);

  if (Pass)
    std::cout << "Pass" << std::endl;
  else
    std::cout << "Fail" << std::endl;

  return !Pass;
}
