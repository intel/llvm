// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s


#include "Inputs/sycl.hpp"

constexpr auto sycl_read_write = sycl::access::mode::read_write;
constexpr auto sycl_global_buffer = sycl::access::target::global_buffer;

template<bool B, typename V = void>
struct enable_if { };
template<typename V>
struct enable_if<true, V> {
  using type = V;
};
template<bool B, typename V = void>
using enable_if_t = typename enable_if<B, V>::type;

template <typename T> class Functor1 {
public:
  Functor1(T X_, sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> &Acc_) :
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


template <typename T> class Functor2 {
public:
  Functor2(T X_, sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> &Acc_) :
    X(X_), Acc(Acc_)
  {}

  [[intel::reqd_sub_group_size(4)]] void operator()(sycl::id<1> id) const {
    Acc.use(id, X);
  }

 [[sycl::work_group_size_hint(1, 2, 3)]] void operator()(sycl::id<2> id) const {
    Acc.use(id, X);
  }

private:
  T X;
  sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> Acc;
};

#define ARR_LEN(x) sizeof(x)/sizeof(x[0])


template <typename T> T bar(T X) {
  T A[] = { (T)10, (T)10 };
  {
    sycl::queue Q;
    sycl::buffer<T, 1> Buf(A, ARR_LEN(A));
   
    Q.submit([&](sycl::handler& cgh) {
      auto Acc = Buf.template get_access<sycl_read_write, sycl_global_buffer>(cgh);
      Functor1<T> F(X, Acc);
      // CHECK: define {{.*}}spir_kernel void @{{.*}}_ZTS8Functor1IiE(i32 noundef %_arg_X, ptr addrspace(1) noundef align 4 %_arg_Acc, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %_arg_Acc1, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %_arg_Acc2, ptr noundef byval(%"struct.sycl::_V1::id") align 4 %_arg_Acc3) #0 !srcloc !11 !kernel_arg_buffer_location !12 !kernel_arg_runtime_aligned !13 !kernel_arg_exclusive_ptr !13 !intel_reqd_sub_group_size !14 {
      cgh.parallel_for(sycl::range<1>(ARR_LEN(A)), F);
      //cgh.parallel_for<class name>(F);
    });

    Q.submit([&](sycl::handler& cgh) {
      auto Acc = Buf.template get_access<sycl_read_write, sycl_global_buffer>(cgh);
      Functor2<T> FuncOp2(X, Acc);
      // CHECK: define {{.*}}spir_kernel void @{{.*}}_ZTS8Functor2IiE(i32 noundef %_arg_X, ptr addrspace(1) noundef align 4 %_arg_Acc, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %_arg_Acc1, ptr noundef byval(%"struct.sycl::_V1::range") align 4 %_arg_Acc2, ptr noundef byval(%"struct.sycl::_V1::id") align 4 %_arg_Acc3) #0 !srcloc !22 !kernel_arg_buffer_location !12 !kernel_arg_runtime_aligned !13 !kernel_arg_exclusive_ptr !13 !work_group_size_hint !23 {
      cgh.parallel_for(sycl::range<2>(ARR_LEN(A)), FuncOp2);
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
  return 0;
}

