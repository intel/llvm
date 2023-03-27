// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -o %t.out %s
// RUN: cd %T
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==--- kernel_functor.cpp -
// This test illustrates defining kernels as named function objects (functors)
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cassert>

constexpr auto sycl_read_write = sycl::access::mode::read_write;
constexpr auto sycl_device = sycl::access::target::device;

// Case 1:
// - functor class is defined in an anonymous namespace
// - the '()' operator:
//   * does not have parameters (to be used in 'single_task').
//   * has the 'const' qualifier
namespace {
class Functor1 {
public:
  Functor1(int X_, sycl::accessor<int, 1, sycl_read_write, sycl_device> &Acc_)
      : X(X_), Acc(Acc_) {}

  void operator()() const { Acc[0] += X; }

private:
  int X;
  sycl::accessor<int, 1, sycl_read_write, sycl_device> Acc;
};
} // namespace

// Case 2:
// - functor class is defined in a namespace
// - the '()' operator:
//   * does not have parameters (to be used in 'single_task').
//   * has the 'const' qualifier
namespace ns {
class Functor2 {
public:
  Functor2(int X_, sycl::accessor<int, 1, sycl_read_write, sycl_device> &Acc_)
      : X(X_), Acc(Acc_) {}

  // sycl::accessor's operator [] is const, hence 'const' is possible below
  void operator()() const { Acc[0] += X; }

private:
  int X;
  sycl::accessor<int, 1, sycl_read_write, sycl_device> Acc;
};
} // namespace ns

// Case 2:
// - functor class is templated and defined in the translation unit scope
// - the '()' operator:
//   * has a parameter of type sycl::id<1> (to be used in 'parallel_for').
//   * has the 'const' qualifier
template <typename T> class TmplFunctor {
public:
  TmplFunctor(T X_, sycl::accessor<T, 1, sycl_read_write, sycl_device> &Acc_)
      : X(X_), Acc(Acc_) {}

  void operator()(sycl::id<1> id) const { Acc[id] += X; }

private:
  T X;
  sycl::accessor<T, 1, sycl_read_write, sycl_device> Acc;
};

// Case 3:
// - functor class is templated and defined in the translation unit scope
// - the '()' operator:
//   * has a parameter of type sycl::id<1> (to be used in 'parallel_for').
//   * has the 'const' qualifier
template <typename T> class TmplConstFunctor {
public:
  TmplConstFunctor(T X_,
                   sycl::accessor<T, 1, sycl_read_write, sycl_device> &Acc_)
      : X(X_), Acc(Acc_) {}

  void operator()(sycl::id<1> id) const { Acc[id] += X; }

private:
  T X;
  sycl::accessor<T, 1, sycl_read_write, sycl_device> Acc;
};

// Exercise non-templated functors in 'single_task'.
int foo(int X) {
  int A[] = {10};
  {
    sycl::queue Q;
    sycl::buffer<int, 1> Buf(A, 1);

    Q.submit([&](sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl_read_write, sycl_device>(cgh);
      Functor1 F(X, Acc);

      cgh.single_task(F);
    });

    Q.submit([&](sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl_read_write, sycl_device>(cgh);
      ns::Functor2 F(X, Acc);

      cgh.single_task(F);
    });
    Q.submit([&](sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl_read_write, sycl_device>(cgh);
      ns::Functor2 F(X, Acc);

      cgh.single_task(F);
    });
  }
  return A[0];
}

#define ARR_LEN(x) sizeof(x) / sizeof(x[0])

// Exercise templated functors in 'parallel_for'.
template <typename T> T bar(T X) {
  T A[] = {(T)10, (T)10};
  {
    sycl::queue Q;
    sycl::buffer<T, 1> Buf(A, ARR_LEN(A));

    Q.submit([&](sycl::handler &cgh) {
      auto Acc = Buf.template get_access<sycl_read_write, sycl_device>(cgh);
      TmplFunctor<T> F(X, Acc);

      cgh.parallel_for(sycl::range<1>(ARR_LEN(A)), F);
    });
    // Spice with lambdas to make sure functors and lambdas work together.
    Q.submit([&](sycl::handler &cgh) {
      auto Acc = Buf.template get_access<sycl_read_write, sycl_device>(cgh);
      cgh.parallel_for<class LambdaKernel>(
          sycl::range<1>(ARR_LEN(A)), [=](sycl::id<1> id) { Acc[id] += X; });
    });
    Q.submit([&](sycl::handler &cgh) {
      auto Acc = Buf.template get_access<sycl_read_write, sycl_device>(cgh);
      TmplConstFunctor<T> F(X, Acc);

      cgh.parallel_for(sycl::range<1>(ARR_LEN(A)), F);
    });
  }
  T res = (T)0;

  for (int i = 0; i < ARR_LEN(A); i++)
    res += A[i];
  return res;
}

int main() {
  const int Res1 = foo(10);
  const int Res2 = bar(10);
  const int Gold1 = 40;
  const int Gold2 = 80;

  assert(Res1 == Gold1);
  assert(Res2 == Gold2);

  return 0;
}
