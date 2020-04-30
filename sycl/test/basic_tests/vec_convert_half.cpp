// XFAIL: cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
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

#include <CL/sycl.hpp>

#include <cassert>

// TODO make the convertion on CPU and HOST identical
// TODO make the test to pass on cuda

using namespace cl::sycl;

template <typename T, typename convertT, int roundingMode>
class kernel_name;

template <int N>
struct helper;

template <>
struct helper<0> {
  template <typename T, int NumElements>
  static void compare(const vec<T, NumElements> &x,
                      const vec<T, NumElements> &y) {
    const T xs = x.template swizzle<0>();
    const T ys = y.template swizzle<0>();
    assert(xs == ys);
  }
};

template <int N>
struct helper {
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

    cl::sycl::device D = Queue.get_device();
    if (!D.has_extension("cl_khr_fp16"))
      exit(0);

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
