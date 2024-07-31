//===---joint_matrix_opt_kernel_feature_unsupported_hw_impl.cpp------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu-intel-gen12, gpu

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test checks that exception will be thrown in case object of joint_matrix type
// is used on unsupported HW, in this case, on Gen12.

#include "common.hpp"

int main() {
  sycl::queue q;

  try {
    q.submit([&](sycl::handler &cgh) {
      cgh.single_task([]() {
        joint_matrix<sycl::sub_group, double, use::b, 2, 2, layout::row_major>
            m; // matrix type and sizes do not matter
      });
    });
  } catch (const sycl::exception &e) {
    assert((e.code() == sycl::errc::kernel_not_supported) &&
           (std::string(e.what()) ==
            std::string("no matrix hardware on the target device, joint_matrix "
                        "is not supported")));
  }
  return 0;
}
