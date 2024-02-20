//==- block_store_slm_acc_pvc.cpp - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------==//
// REQUIRES: gpu-intel-pvc

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::slm_block_store() functions writing to SLM memory
// and using optional compile-time esimd::properties and local accessors.
// The slm_block_store() calls in this test use the mask operand and
// requires PVC features.

#include "Inputs/block_store.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  constexpr auto TestFeatures = TestFeatures::PVC;
  bool Passed = true;

  Passed &= test_block_store_local_acc_slm<int8_t, TestFeatures>(Q);
  Passed &= test_block_store_local_acc_slm<int16_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= test_block_store_local_acc_slm<sycl::half, TestFeatures>(Q);
  Passed &= test_block_store_local_acc_slm<uint32_t, TestFeatures>(Q);
  Passed &= test_block_store_local_acc_slm<float, TestFeatures>(Q);
  Passed &=
      test_block_store_local_acc_slm<ext::intel::experimental::esimd::tfloat32,
                                     TestFeatures>(Q);
  Passed &=
      test_block_store_local_acc_slm<ext::intel::experimental::esimd::tfloat32,
                                     TestFeatures>(Q);
  Passed &= test_block_store_local_acc_slm<int64_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= test_block_store_local_acc_slm<double, TestFeatures>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
