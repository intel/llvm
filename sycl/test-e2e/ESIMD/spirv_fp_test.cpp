//==- spirv_fp_test.cpp - Test for abs function -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

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

template <typename T, int N>
bool test(sycl::queue &Queue, T testValue1, T testValue2) {
  shared_allocator<T> Allocator(Queue);

  shared_vector<T> OutputAbs(N, 0, Allocator);
  shared_vector<T> OutputMin(N, 0, Allocator);
  shared_vector<T> OutputMax(N, 0, Allocator);

  auto *OutputAbsPtr = OutputAbs.data();
  auto *OutputMinPtr = OutputMin.data();
  auto *OutputMaxPtr = OutputMax.data();

  Queue.submit([&](sycl::handler &cgh) {
    auto Kernel = ([=]() SYCL_ESIMD_KERNEL {
      simd<T, N> Input1 = testValue1;
      simd<T, N> Input2 = testValue2;
      simd<T, N> ResultAbs = __ESIMD_NS::abs(Input1);
      simd<T, N> ResultMin = __ESIMD_NS::min(Input1, Input2);
      simd<T, N> ResultMax = __ESIMD_NS::max(Input1, Input2);
      ResultAbs.copy_to(OutputAbsPtr);
      ResultMin.copy_to(OutputMinPtr);
      ResultMax.copy_to(OutputMaxPtr);
    });
    cgh.single_task(Kernel);
  });
  Queue.wait();

  for (int I = 0; I < N; I++) {
    if (std::abs(testValue1) != OutputAbs[I]) {
      std::cout << "Incorrect value for abs at index " << I << " "
                << std::abs(testValue1) << " != " << OutputAbs[I] << std::endl;
      return false;
    }
    if (std::min(testValue1, testValue2) != OutputMin[I]) {
      std::cout << "Incorrect value for min at index " << I << " "
                << std::min(testValue1, testValue2) << " != " << OutputMin[I]
                << std::endl;
      return false;
    }

    if (std::max(testValue1, testValue2) != OutputMax[I]) {
      std::cout << "Incorrect value for max at index " << I << " "
                << std::max(testValue1, testValue2) << " != " << OutputMax[I]
                << std::endl;
      return false;
    }
  }

  return true;
}

int main() {

  bool Pass = true;
  sycl::queue Q;
  Pass &= test<bf16, 8>(Q, -1, -2);
  Pass &= test<tfloat32, 8>(Q, -1, -2);

  if (Pass)
    std::cout << "Pass" << std::endl;
  else
    std::cout << "Fail" << std::endl;

  return !Pass;
}
