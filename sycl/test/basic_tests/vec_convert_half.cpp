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
#include <iostream>
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
      std::cerr << "sometihng failed " << std::setprecision(30) << xs << " || "<< ys << "\n";
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
      std::cerr << "sometihng failed " << std::setprecision(30) << xs << " || "<< ys << "\n";  
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

int main(){
  //automatic
  test<double, half, 8, rounding_mode::automatic>(
      double8{+2.3, +2.5, +2.7, -2.3, -2.5, -2.7, 0., 0.},
      half8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f});
  /*test<half, double, 8, rounding_mode::automatic>(
      double8{+2.3, +2.5, +2.7, -2.3, -2.5, -2.7, 0., 0.},
      half8{+2.3f, +2.5f, +2.7f, -2.3f, -2.5f, -2.7f, 0.f, 0.f}); */
}