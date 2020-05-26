// XFAIL: cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==------------ vec_convert_i_to_i.cpp - SYCL vec class convert method test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vec_convert.hpp"

// TODO make the test to pass on cuda

int main() {

  test<short, char, 8, rounding_mode::automatic>(
      short8{300, -300, 100, -50, 128, -129, 0, 1},
      char8{44, -44, 100, -50, -128, 127, 0, 1});
  test<int, short, 8, rounding_mode::automatic>(
      int8{100000, -100000, 100, -50, 32768, -32769, 0, 1},
      short8{-31072, 31072, 100, -50, -32768, 32767, 0, 1});
  test<long, int, 8, rounding_mode::automatic>(
      long8{3000000000, -3000000000, 100, -50, 2147483648, -2147483649, 0, 1},
      int8{-1294967296, 1294967296, 100, -50, -2147483648, 2147483647, 0, 1});

  test<ushort, uchar, 8, rounding_mode::automatic>(
      ushort8{300, 255, 100, 150, 128, 256, 0, 1},
      uchar8{44, 255, 100, 150, 128, 0, 0, 1});
  test<uint, ushort, 8, rounding_mode::automatic>(
      uint8{100000, 65535, 100, 150, 32768, 65536, 0, 1},
      ushort8{34464, 65535, 100, 150, 32768, 0, 0, 1});
  test<ulong, uint, 8, rounding_mode::automatic>(
      ulong8{10000000000, 4294967295, 100, 150, 2147483648, 4294967296, 0, 1},
      uint8{1410065408, 4294967295, 100, 150, 2147483648, 0, 0, 1});

  test<int, uint, 8, rounding_mode::automatic>(
      int8{2147483647, -1, 100, 150, -100, -2147483648, 0, 1},
      uint8{2147483647, 4294967295, 100, 150, 4294967196, 2147483648, 0, 1});
  test<short, uint, 8, rounding_mode::automatic>(
      short8{32767, -1, 100, 150, -100, -32768, 0, 1},
      uint8{32767, 4294967295, 100, 150, 4294967196, 4294934528, 0, 1});
  test<ulong, int, 8, rounding_mode::automatic>(
      ulong8{3000000000, 2147483647, 100, 150, 2147483648, 1000, 0, 1},
      int8{-1294967296, 2147483647, 100, 150, -2147483648, 1000, 0, 1});

  test<longlong, ulonglong, 1, rounding_mode::automatic>(
      longlong{1},
      ulonglong{1});

    test<ulonglong, longlong, 1, rounding_mode::automatic>(
      ulonglong{1},
      longlong{1});

  return 0;
}
