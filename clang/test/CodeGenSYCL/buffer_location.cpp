// RUN: %clang_cc1 -fsycl -fsycl-is-device -I %S/Inputs -triple spir64-unknown-unknown-sycldevice -emit-llvm %s -o - | FileCheck %s

// CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE15kernel_function{{.*}} !kernel_arg_buffer_location ![[MDBL:[0-9]+]]
// CHECK: ![[MDBL]] = !{i32 3, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 2, i32 -1, i32 -1, i32 -1, i32 2, i32 -1, i32 -1, i32 -1, i32 -1}

#include "sycl.hpp"

struct Base {
  int A, B;
  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     cl::sycl::property_list<
                         cl::sycl::property::buffer_location<2>>>
      AccField;
};

struct Captured : Base,
                  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read,
                                     cl::sycl::access::target::global_buffer,
                                     cl::sycl::access::placeholder::false_t,
                                     cl::sycl::property_list<
                                         cl::sycl::property::buffer_location<2>>> {
  int C;
};

int main() {
  Captured Obj;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     cl::sycl::property_list<
                         cl::sycl::property::buffer_location<3>>>
      accessorA;
  cl::sycl::kernel_single_task<class kernel_function>(
      [=]() {
        accessorA.use();
        Obj.use();
      });
  return 0;
}
