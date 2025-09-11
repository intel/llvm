// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s
// This test checks that the correct kernel operator call is invoked when there are multiple definitions of the 
// 'operator()()' call.

#include "sycl.hpp"

sycl::queue Q;

// Check if functor with multiple call operator works.
class Functor1 {
public:
 Functor1(){}

 [[sycl::reqd_sub_group_size(4)]] void operator()(sycl::id<1> id) const {}

 [[sycl::work_group_size_hint(1, 2, 3)]] void operator()(sycl::id<2> id) const {}
};

class ESIMDFunctor {
public:
  ESIMDFunctor(){}

  [[intel::sycl_explicit_simd]] void operator()(sycl::id<2> id) const {}

  [[sycl::work_group_size_hint(1, 2, 3)]] [[intel::sycl_explicit_simd]]
  void operator()(sycl::id<1> id) const {}
};

// Check templated 'operator()()' call works.
class kernels {
public:
  kernels(){}

  template<int Dimensions = 1>
  [[sycl::work_group_size_hint(1, 2, 3)]]
  void operator()(sycl::id<Dimensions> item) const {}
};

int main() {

  Q.submit([&](sycl::handler& cgh) {
      Functor1 F;
      // CHECK: define weak_odr spir_kernel void @_ZTS8Functor1() {{.*}} !intel_reqd_sub_group_size
      cgh.parallel_for(sycl::range<1>(10), F);
    });

  Q.submit([&](sycl::handler& cgh) {
      kernels K;
      // CHECK: define weak_odr spir_kernel void @_ZTS7kernels() {{.*}} !work_group_size_hint !{{[0-9]+}}
      cgh.parallel_for(sycl::range<1>(10), K);
    });

  Q.submit([&](sycl::handler& cgh) {
      ESIMDFunctor EF;
      // CHECK: define weak_odr spir_kernel void @_ZTS12ESIMDFunctor() {{.*}} !work_group_size_hint
      // CHECK-SAME: !sycl_explicit_simd
      cgh.parallel_for(sycl::range<1>(10), EF);
    });

  return 0;
}

