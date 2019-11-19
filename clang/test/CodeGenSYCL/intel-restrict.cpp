// RUN: %clang %s -S -emit-llvm --sycl -o - | FileCheck %s

#include "CL/sycl.hpp"

constexpr auto sycl_read_write = cl::sycl::access::mode::read_write;
constexpr auto sycl_global_buffer = cl::sycl::access::target::global_buffer;

template <typename Acc1Ty, typename Acc2Ty>
struct foostr {
  Acc1Ty A;
  Acc2Ty B;
  foostr(Acc1Ty A, Acc2Ty B): A(A), B(B) {}
  [[intel::kernel_args_restrict]]
  void operator()() {
    A[0] = B[0];
  }
};

int foo(int X) {
  int A[] = { 42 };
  int B[] = { 0 };
  {
    cl::sycl::queue Q;
    cl::sycl::buffer<int, 1> BufA(A, 1);
    cl::sycl::buffer<int, 1> BufB(B, 1);

    // CHECK: define {{.*}} spir_kernel {{.*}}kernel_norestrict{{.*}}(i32 addrspace(1)* %{{.*}} i32 addrspace(1)* %{{.*}}

    Q.submit([&](cl::sycl::handler& cgh) {
      auto AccA = BufA.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      auto AccB = BufB.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      cgh.single_task<class kernel_norestrict>(
          [=]() {
            AccB[0] = AccA[0];
          });
    });

    // CHECK: define {{.*}} spir_kernel {{.*}}kernel_restrict{{.*}}(i32 addrspace(1)* noalias %{{.*}} i32 addrspace(1)* noalias %{{.*}}
    Q.submit([&](cl::sycl::handler& cgh) {
      auto AccA = BufA.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      auto AccB = BufB.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      cgh.single_task<class kernel_restrict>(
          [=]() [[intel::kernel_args_restrict]] {
            AccB[0] = AccA[0];
          });
    });

    // CHECK: define {{.*}} spir_kernel {{.*}}kernel_restrict_struct{{.*}}(i32 addrspace(1)* noalias %{{.*}} i32 addrspace(1)* noalias %{{.*}}
    Q.submit([&](cl::sycl::handler& cgh) {
      auto AccA = BufA.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      auto AccB = BufB.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      foostr<decltype(AccA), decltype(AccB)> f(AccA, AccB);
      cgh.single_task<class kernel_restrict_struct>(f);
    });

    // CHECK: define {{.*}} spir_kernel {{.*}}kernel_restrict_other_params{{.*}}(i32 addrspace(1)* noalias %{{.*}} i32 addrspace(1)* noalias %{{.*}}, i32 %{{[^,]*}})
    int num = 42;
    Q.submit([&](cl::sycl::handler& cgh) {
      auto AccA = BufA.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      auto AccB = BufB.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      cgh.single_task<class kernel_restrict_other_params>(
          [=]() [[intel::kernel_args_restrict]] {
            AccB[0] = AccA[0] = num;
          });
    });
  }
  return B[0];
}
