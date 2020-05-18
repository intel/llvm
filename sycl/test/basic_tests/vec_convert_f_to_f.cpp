// XFAIL: cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
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

// TODO make the convertion on CPU and HOST identical
// TODO make the test to pass on cuda

int main() {
  // automatic
  test<double, float, 8, rounding_mode::automatic>(
      double8{1234567890.0, 987654304.0, 100.0, -50.0, 111111.111, 625.625, 50625.0009765625, -2500000.875},
      float8{1234567936.0f, 987654272.0f, 100.0f, -50.0f, 111111.109375f, 625.625f, 50625.0f, -2500001.0f});
  test<float, double, 8, rounding_mode::automatic>(
      float8{1234567936.0f, 987654272.0f, 100.0f, -50.0f, 111111.109375f, 625.625f, 50625.0f, -2500001.0f},
      double8{1234567936.0, 987654272.0, 100.0, -50.0, 111111.109375, 625.625, 50625.0, -2500001.0});

  // rte
  test<double, float, 8, rounding_mode::rte>(
      double8{1234567890.0, 987654304.0, 100.0, -50.0, 111111.111, 625.625, 50625.0009765625, -2500000.875},
      float8{1234567936.0f, 987654272.0f, 100.0f, -50.0f, 111111.109375f, 625.625f, 50625.0f, -2500001.0f});
  test<float, double, 8, rounding_mode::rte>(
      float8{1234567936.0f, 987654272.0f, 100.0f, -50.0f, 111111.109375f, 625.625f, 50625.0f, -2500001.0f},
      double8{1234567936.0, 987654272.0, 100.0, -50.0, 111111.109375, 625.625, 50625.0, -2500001.0});

  // rtp
  test<double, float, 8, rounding_mode::rtp>(
      double8{1234567890.0, 987654304.0, 100.0, -50.0, 111111.111, 625.625, 50625.0009765625, -2500000.875},
      float8{1234567936.0f, 987654336.0f, 100.0f, -50.0f, 111111.1171875f, 625.625f, 50625.00390625f, -2500000.75f});
  test<float, double, 8, rounding_mode::rtp>(
      float8{1234567936.0f, 987654272.0f, 100.0f, -50.0f, 111111.109375f, 625.625f, 50625.0f, -2500001.0f},
      double8{1234567936.0, 987654272.0, 100.0, -50.0, 111111.109375, 625.625, 50625.0, -2500001.0});

  // rtn
  test<double, float, 8, rounding_mode::rtn>(
      double8{1234567890.0, 987654304.0, 100.0, -50.0, 111111.111, 625.625, 50625.0009765625, -2500000.875},
      float8{1234567808.0f, 987654272.0f, 100.0f, -50.0f, 111111.109375f, 625.625f, 50625.0f, -2500001.0f});
  test<float, double, 8, rounding_mode::rtn>(
      float8{1234567936.0f, 987654272.0f, 100.0f, -50.0f, 111111.109375f, 625.625f, 50625.0f, -2500001.0f},
      double8{1234567936.0, 987654272.0, 100.0, -50.0, 111111.109375, 625.625, 50625.0, -2500001.0});

  // rtz
  test<double, float, 8, rounding_mode::rtz>(
      double8{1234567890.0, 987654304.0, 100.0, -50.0, 111111.111, 625.625, 50625.0009765625, -2500000.875},
      float8{1234567808.0f, 987654272.0f, 100.0f, -50.0f, 111111.109375f, 625.625f, 50625.0f, -2500000.75f});
  test<float, double, 8, rounding_mode::rtz>(
      float8{1234567936.0f, 987654272.0f, 100.0f, -50.0f, 111111.109375f, 625.625f, 50625.0f, -2500001.0f},
      double8{1234567936.0, 987654272.0, 100.0, -50.0, 111111.109375, 625.625, 50625.0, -2500001.0});

  return 0;
}
