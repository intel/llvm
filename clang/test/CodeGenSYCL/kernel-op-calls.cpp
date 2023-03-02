// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-device %s -o /dev/null
// RUN: FileCheck %s --input-file=%t.h --check-prefixes=UL,CHECK
   
// Checks that functors are supported as SYCL kernels.

#include "Inputs/sycl.hpp"

constexpr auto sycl_read_write = sycl::access::mode::read_write;
constexpr auto sycl_global_buffer = sycl::access::target::global_buffer;

// Case 2:
// - functor class is templated and defined in the translation unit scope
// - the '()' operator:
//   * has a parameter of type sycl::id<1> (to be used in 'parallel_for').
template <typename T> class TmplConstFunctor {
public:
  TmplConstFunctor(T X_, sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> &Acc_) :
    X(X_), Acc(Acc_)
  {}

  [[intel::reqd_sub_group_size(4)]] void operator()(sycl::id<1> id) const {
    Acc.use(id, X);
  }

 void operator()() const {
    Acc.use();
  }

  

private:
  T X;
  sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> Acc;
};



#define ARR_LEN(x) sizeof(x)/sizeof(x[0])

// Exercise templated functors in 'parallel_for'.
template <typename T> T bar(T X) {
  T A[] = { (T)10, (T)10 };
  {
    sycl::queue Q;
    sycl::buffer<T, 1> Buf(A, ARR_LEN(A));
   
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
  const int Res2 = bar(10);
  const int Gold1 = 40;
  const int Gold2 = 80;

#ifndef __SYCL_DEVICE_ONLY__
  
  sycl::detail::KernelInfo<TmplConstFunctor<int>>::getName();
  // NUL: KernelInfo<::TmplConstFunctor<int>>
  // UL: KernelInfoData<'_', 'Z', 'T', 'S', '1', '6', 'T', 'm', 'p', 'l', 'C', 'o', 'n', 's', 't', 'F', 'u', 'n', 'c', 't', 'o', 'r', 'I', 'i', 'E'>
  // CHECK: getName() { return "_ZTS16TmplConstFunctorIiE"; }
#endif // __SYCL_DEVICE_ONLY__

  return 0;
}

