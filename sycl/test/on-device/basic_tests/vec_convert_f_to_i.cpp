// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------ vec_convert.cpp - SYCL vec class convert method test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vec_convert.hpp"

// TODO make the test to pass on cuda

int main() {
  // automatic
  test<float, int, 8, rounding_mode::automatic>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      int8{2, 2, 3, -2, -2, -3, 0, 0});
  test<int, float, 8, rounding_mode::automatic>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      float8{2.f, 3.f, 3.f, -2.f, -3.f, -3.f, 0.f, 0.f});

  // rte
  test<float, int, 8, rounding_mode::rte>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      int8{2, 2, 3, -2, -2, -3, 0, 0});
  test<int, float, 8, rounding_mode::rte>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      float8{2.f, 3.f, 3.f, -2.f, -3.f, -3.f, 0.f, 0.f});

  // rtz
  test<float, int, 8, rounding_mode::rtz>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      int8{2, 2, 2, -2, -2, -2, 0, 0});
  test<int, float, 8, rounding_mode::rtz>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      float8{2.f, 3.f, 3.f, -2.f, -3.f, -3.f, 0.f, 0.f});

  // rtp
  test<float, int, 8, rounding_mode::rtp>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      int8{3, 3, 3, -2, -2, -2, 0, 0});
  test<int, float, 8, rounding_mode::rtp>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      float8{2.f, 3.f, 3.f, -2.f, -3.f, -3.f, 0.f, 0.f});

  // rtn
  test<float, int, 8, rounding_mode::rtn>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      int8{2, 2, 2, -3, -3, -3, 0, 0});
  test<int, float, 8, rounding_mode::rtn>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      float8{2.f, 3.f, 3.f, -2.f, -3.f, -3.f, 0.f, 0.f});
  return 0;
}
