//==-------- joint_matrix_bfloat16.cpp  - DPC++ joint_matrix----------- ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: target-spir

// REQUIRES: aspect-ext_intel_matrix

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=2 %{run} %t.out %}
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=1 %{run} %t.out %}
// RUN: %if gpu %{ env IGC_JointMatrixLoadStoreOpt=0 %{run} %t.out %}

#include "common.hpp"

#include "joint_matrix_16bit_impl.hpp"

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      test<bfloat16, float, float, /*TM*/ 16, /*TN*/ 16, /*TK*/ 16,
           layout::ext_intel_packed, 2>();
      test<bfloat16, float, float, /*TM*/ 16, /*TN*/ 16, /*TK*/ 32,
           layout::ext_intel_packed, 2>();
      test<bfloat16, float, float, /*TM*/ 16, /*TN*/ 16, /*TK*/ 16,
           layout::row_major, 1>();
      test<bfloat16, float, float, /*TM*/ 16, /*TN*/ 16, /*TK*/ 32,
           layout::row_major, 1>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      test<bfloat16, float, float, /*TM*/ 8, /*TN*/ 16, /*TK*/ 16,
           layout::row_major, 1>();
      test<bfloat16, float, float, /*TM*/ 8, /*TN*/ 16, /*TK*/ 16,
           layout::ext_intel_packed, 2>();
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      test<bfloat16, float, float, /*TM*/ 8, /*TN*/ 8, /*TK*/ 16,
           layout::row_major, 1>();
      test<bfloat16, float, float, /*TM*/ 8, /*TN*/ 8, /*TK*/ 16,
           layout::ext_intel_packed, 2>();
      break;
    }
  }
  return 0;
}
