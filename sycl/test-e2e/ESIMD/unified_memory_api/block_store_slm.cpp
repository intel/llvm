//==------- block_store_slm.cpp - DPC++ ESIMD on-device test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//
// REQUIRES-INTEL-DRIVER: win: 101.4887
// Somehow the driver version check above does not work, i.e. Windows CI runs
// the test with 31.0.101.4502 (if opencl:gpu) and 1.3.26370 (if level-zero:gpu)
// It seems the driver check infrastructure may need some fix/tuning.
// TODO: Enable the test when Windows CI driver reaches 101.4887 version, or
// driver version check is fixed/tuned.
// UNSUPPORTED: windows

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test verifies esimd::slm_block_store() functions writing to SLM memory
// and using optional compile-time esimd::properties.
// The slm_block_store() calls in this test do not use the mask operand and
// do not require PVC features.

#include "Inputs/block_store.hpp"

int main() {
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  constexpr auto TestFeatures = TestFeatures::Generic;
  bool Passed = true;

  Passed &= test_block_store_slm<int8_t, TestFeatures>(Q);
  Passed &= test_block_store_slm<int16_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= test_block_store_slm<sycl::half, TestFeatures>(Q);
  Passed &= test_block_store_slm<uint32_t, TestFeatures>(Q);
  Passed &= test_block_store_slm<float, TestFeatures>(Q);
  Passed &= test_block_store_slm<int64_t, TestFeatures>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= test_block_store_slm<double, TestFeatures>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
