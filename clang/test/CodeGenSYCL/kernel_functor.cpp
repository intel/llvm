// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-device -fno-sycl-unnamed-lambda -fsycl-int-header=%t.h %s -o /dev/null
// RUN: FileCheck %s --input-file=%t.h --check-prefixes=NUL,CHECK
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-device -fsycl-int-header=%t.h %s -o /dev/null
// RUN: FileCheck %s --input-file=%t.h --check-prefixes=UL,CHECK

// Checks that functors are supported as SYCL kernels.

#include "Inputs/sycl.hpp"

constexpr auto sycl_read_write = sycl::access::mode::read_write;
constexpr auto sycl_global_buffer = sycl::access::target::global_buffer;

// Case 1:
// - functor class is defined in a namespace
// - the '()' operator:
//   * does not have parameters (to be used in 'single_task').
namespace ns {
  class Functor2 {
  public:
    Functor2(int X_, sycl::accessor<int, 1, sycl_read_write, sycl_global_buffer> &Acc_) :
      X(X_), Acc(Acc_)
    {}

    void operator()() const {
      Acc.use(X);
    }

  private:
    int X;
    sycl::accessor<int, 1, sycl_read_write, sycl_global_buffer> Acc;
  };
}

// Case 2:
// - functor class is templated and defined in the translation unit scope
// - the '()' operator:
//   * has a parameter of type sycl::id<1> (to be used in 'parallel_for').
template <typename T> class TmplConstFunctor {
public:
  TmplConstFunctor(T X_, sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> &Acc_) :
    X(X_), Acc(Acc_)
  {}

  void operator()(sycl::id<1> id) const {
    Acc.use(id, X);
  }

private:
  T X;
  sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> Acc;
};

// Exercise non-templated functors in 'single_task'.
int foo(int X) {
  int A[] = { 10 };
  {
    sycl::queue Q;
    sycl::buffer<int, 1> Buf(A, 1);

    Q.submit([&](sycl::handler& cgh) {
      auto Acc = Buf.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      ns::Functor2 F(X, Acc);

      cgh.single_task(F);
    });
    Q.submit([&](sycl::handler& cgh) {
      auto Acc = Buf.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      ns::Functor2 F(X, Acc);

      cgh.single_task(F);
    });
  }
  return A[0];
}

#define ARR_LEN(x) sizeof(x)/sizeof(x[0])

// Exercise templated functors in 'parallel_for'.
template <typename T> T bar(T X) {
  T A[] = { (T)10, (T)10 };
  {
    sycl::queue Q;
    sycl::buffer<T, 1> Buf(A, ARR_LEN(A));
    // Spice with lambdas to make sure functors and lambdas work together.
    Q.submit([&](sycl::handler& cgh) {
      auto Acc = Buf.template get_access<sycl_read_write, sycl_global_buffer>(cgh);
      cgh.parallel_for<class LambdaKernel>(
        sycl::range<1>(ARR_LEN(A)),
        [=](sycl::id<1> id) { Acc.use(id, X); });
    });
    Q.submit([&](sycl::handler& cgh) {
      auto Acc = Buf.template get_access<sycl_read_write, sycl_global_buffer>(cgh);
      TmplConstFunctor<T> F(X, Acc);

      cgh.parallel_for(sycl::range<1>(ARR_LEN(A)), F);
    });
  }
  T res = (T)0;

  for (int i = 0; i < ARR_LEN(A); i++) {
    res += A[i];
  }
  return res;
}

int main() {
  const int Res1 = foo(10);
  const int Res2 = bar(10);
  const int Gold1 = 40;
  const int Gold2 = 80;

#ifndef __SYCL_DEVICE_ONLY__
  sycl::detail::KernelInfo<ns::Functor2>::getName();
  // NUL: KernelInfo<::ns::Functor2>
  // UL: KernelInfoData<'_', 'Z', 'T', 'S', 'N', '2', 'n', 's', '8', 'F', 'u', 'n', 'c', 't', 'o', 'r', '2', 'E'>
  // CHECK: getName() { return "_ZTSN2ns8Functor2E"; }
  sycl::detail::KernelInfo<TmplConstFunctor<int>>::getName();
  // NUL: KernelInfo<::TmplConstFunctor<int>>
  // UL: KernelInfoData<'_', 'Z', 'T', 'S', '1', '6', 'T', 'm', 'p', 'l', 'C', 'o', 'n', 's', 't', 'F', 'u', 'n', 'c', 't', 'o', 'r', 'I', 'i', 'E'>
  // CHECK: getName() { return "_ZTS16TmplConstFunctorIiE"; }
#endif // __SYCL_DEVICE_ONLY__

  return 0;
}

