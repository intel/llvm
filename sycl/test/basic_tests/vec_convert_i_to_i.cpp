// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=CPU %t.out


//==------------ vec_convert.cpp - SYCL vec class convert method test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <cassert>
#include <iomanip> 

// TODO uncomment run lines on non-host devices when the rounding modes will
// be implemented.

using namespace cl::sycl;

template <typename T, typename convertT, int roundingMode> class kernel_name;

template <int N> struct helper;

template <> struct helper<0> {
  template <typename T, int NumElements>
  static void compare(const vec<T, NumElements> &x,
                      const vec<T, NumElements> &y) {
    const T xs = x.template swizzle<0>();
    const T ys = y.template swizzle<0>();
    if (xs != ys) {
      std::cerr << "sometihng failed " << std::setprecision(30) << int(xs) << " || "<< int(ys);;
      exit(1);
    }
  }
};

template <int N> struct helper {
  template <typename T, int NumElements>
  static void compare(const vec<T, NumElements> &x,
                      const vec<T, NumElements> &y) {
    const T xs = x.template swizzle<N>();
    const T ys = y.template swizzle<N>();
    helper<N - 1>::compare(x, y);
    if (xs != ys) {
      std::cerr << "sometihng failed " << std::setprecision(30) << xs << " || "<< ys;
      exit(1);
    }
  }
};

template <typename T, typename convertT, int NumElements,
          rounding_mode roundingMode>
void test(const vec<T, NumElements> &ToConvert,
          const vec<convertT, NumElements> &Expected) {
  vec<convertT, NumElements> Converted{0};
  {
    buffer<vec<convertT, NumElements>, 1> Buffer{&Converted, range<1>{1}};
    queue Queue;
    Queue.submit([&](handler &CGH) {
      accessor<vec<convertT, NumElements>, 1, access::mode::write> Accessor(
          Buffer, CGH);
        CGH.single_task<class kernel_name<T, convertT, static_cast<int>(roundingMode)>>([=]() {
          Accessor[0] = ToConvert.template convert<convertT, roundingMode>();
        });
    });
  }
  helper<NumElements - 1>::compare(Converted, Expected);
}

int main() {

  /*test<short, char, 8, rounding_mode::automatic>(
      short8{300, -300, 100, -50, 128, -129, 0, 1},
      char8{127, -128, 100, -50, 127, -128, 0, 1});
  test<int, short, 8, rounding_mode::automatic>(
      int8{100000, -100000, 100, -50, 32768, -32769, 0, 1},
      short8{32767, -32768, 100, -50, 32767, -32768, 0, 1});  
  test<long, int, 8, rounding_mode::automatic>(
      long8{3000000000, -3000000000, 100, -50, 2147483648, -2147483649, 0, 1},
      int8{2147483647, -2147483648, 100, -50, 2147483647, -2147483648, 0, 1});*/

  /*test<ushort, uchar, 8, rounding_mode::automatic>(
      ushort8{300, 255, 100, 150, 128, 256, 0, 1},
      uchar8{255, 255, 100, 150, 128, 255, 0, 1});
  test<uint, ushort, 8, rounding_mode::automatic>(
      uint8{100000, 65535, 100, 150, 32768, 65536, 0, 1},
      ushort8{65535, 65535, 100, 150, 32768, 65535, 0, 1}); 
  test<ulong, uint, 8, rounding_mode::automatic>(
      ulong8{10000000000, 4294967295, 100, 150, 2147483648, 4294967296, 0, 1},
      uint8{4294967295, 4294967295, 100, 150, 2147483648, 4294967295, 0, 1});  */

  test<int, uint, 8, rounding_mode::automatic>(
      int8{2147483647, -1, 100, 150, -100, -2147483648, 0, 1},
      uint8{2147483647, 0, 100, 150, 0, 0, 0, 1});
  test<short, uint, 8, rounding_mode::automatic>(
      short8{32767, -1, 100, 150, -100, -32768, 0, 1},
      uint8{32767, 0, 100, 150, 0, 0, 0, 1}); 
  test<ulong, int, 8, rounding_mode::automatic>(
      ulong8{3000000000, 2147483647, 100, 150, 2147483648, 1000, 0, 1},
      int8{2147483647, 2147483647, 100, 150, 2147483647, 1000, 0, 1}); 

    return 0;
}