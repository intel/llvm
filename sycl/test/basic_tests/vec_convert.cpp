// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUNx: %GPU_RUN_PLACEHOLDER %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out
//==------------ vec_convert.cpp - SYCL vec class convert method test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <cassert>

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
    assert(xs == ys);
  }
};

template <int N> struct helper {
  template <typename T, int NumElements>
  static void compare(const vec<T, NumElements> &x,
                      const vec<T, NumElements> &y) {
    const T xs = x.template swizzle<N>();
    const T ys = y.template swizzle<N>();
    helper<N - 1>::compare(x, y);
    assert(xs == ys);
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
  // automatic
  test<int, int, 8, rounding_mode::automatic>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      int8{2, 3, 3, -2, -3, -3, 0, 0});
  test<float, int, 8, rounding_mode::automatic>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      int8{2, 2, 3, -2, -2, -3, 0, 0});
  test<int, float, 8, rounding_mode::automatic>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      float8{2.f, 3.f, 3.f, -2.f, -3.f, -3.f, 0.f, 0.f});
  test<float, float, 8, rounding_mode::automatic>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f});
  test<float, half, 8, rounding_mode::automatic>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      half8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f});

  // rte
  test<int, int, 8, rounding_mode::rte>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      int8{2, 3, 3, -2, -3, -3, 0, 0});
  test<float, int, 8, rounding_mode::rte>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      int8{2, 2, 3, -2, -2, -3, 0, 0});
  test<int, float, 8, rounding_mode::rte>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      float8{2.f, 3.f, 3.f, -2.f, -3.f, -3.f, 0.f, 0.f});
  test<float, float, 8, rounding_mode::rte>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f});
  test<float, half, 8, rounding_mode::rte>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      half8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f});

  // rtz
  test<int, int, 8, rounding_mode::rtz>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      int8{2, 3, 3, -2, -3, -3, 0, 0});
  test<float, int, 8, rounding_mode::rtz>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      int8{2, 2, 2, -2, -2, -2, 0, 0});
  test<int, float, 8, rounding_mode::rtz>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      float8{2.f, 3.f, 3.f, -2.f, -3.f, -3.f, 0.f, 0.f});
  test<float, float, 8, rounding_mode::rtz>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f});
  test<float, half, 8, rounding_mode::rtz>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      half8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f});

  // rtp
  test<int, int, 8, rounding_mode::rtp>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      int8{2, 3, 3, -2, -3, -3, 0, 0});
  test<float, int, 8, rounding_mode::rtp>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      int8{3, 3, 3, -2, -2, -2, 0, 0});
  test<int, float, 8, rounding_mode::rtp>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      float8{2.f, 3.f, 3.f, -2.f, -3.f, -3.f, 0.f, 0.f});
  test<float, float, 8, rounding_mode::rtp>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f});
  test<float, half, 8, rounding_mode::rtp>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      half8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f});

  // rtn
  test<int, int, 8, rounding_mode::rtn>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      int8{2, 3, 3, -2, -3, -3, 0, 0});
  test<float, int, 8, rounding_mode::rtn>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      int8{2, 2, 2, -3, -3, -3, 0, 0});
  test<int, float, 8, rounding_mode::rtn>(
      int8{2, 3, 3, -2, -3, -3, 0, 0},
      float8{2.f, 3.f, 3.f, -2.f, -3.f, -3.f, 0.f, 0.f});
  test<float, float, 8, rounding_mode::rtn>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f});
  test<float, half, 8, rounding_mode::rtn>(
      float8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f},
      half8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f});

  return 0;
}
