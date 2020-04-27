// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------ vec_convert_half.cpp - SYCL vec class convert method test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vec_convert.hpp"

int main() {
  //automatic
  test<double, half, 4, rounding_mode::automatic>(
      double4{12345.0, 100.0, -50.0, 11111.111},
      half4{12344.0f, 100.0, -50.0, 11112});

  //rte
  test<double, half, 4, rounding_mode::rte>(
      double4{12345.0, 100.0, -50.0, 11111.111},
      half4{12344.0f, 100.0, -50.0, 11112});
  //rtp
  test<double, half, 4, rounding_mode::rtp>(
      double4{12345.0, 100.0, -50.0, 11111.111},
      half4{12352.0f, 100.0, -50.0, 11112});

  //rtn
  test<double, half, 4, rounding_mode::rtn>(
      double4{12345.0, 100.0, -50.0, 11111.111},
      half4{12344.0f, 100.0, -50.0, 11104});

  //rtz
  test<double, half, 4, rounding_mode::rtz>(
      double4{12345.0, 100.0, -50.0, 11111.111},
      half4{12344.0f, 100.0, -50.0, 11104});

  return 0;
}
