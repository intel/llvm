// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE15kernel_function{{.*}} !kernel_arg_buffer_location ![[MDBL:[0-9]+]]
// CHECK: ![[MDBL]] = !{i32 3, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 2, i32 -1, i32 -1, i32 -1, i32 2, i32 -1, i32 -1, i32 -1, i32 -1}

#include "Inputs/sycl.hpp"

struct Base {
  int A, B;
  sycl::accessor<char, 1, sycl::access::mode::read,
                     sycl::access::target::global_buffer,
                     sycl::access::placeholder::false_t,
                     sycl::ext::oneapi::accessor_property_list<
                         sycl::ext::intel::property::buffer_location::instance<2>>>
      AccField;
};

struct Captured : Base,
                  sycl::accessor<char, 1, sycl::access::mode::read,
                                     sycl::access::target::global_buffer,
                                     sycl::access::placeholder::false_t,
                                     sycl::ext::oneapi::accessor_property_list<
                                         sycl::ext::intel::property::buffer_location::instance<2>>> {
  int C;
};

int main() {
  Captured Obj;
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::global_buffer,
                     sycl::access::placeholder::false_t,
                     sycl::ext::oneapi::accessor_property_list<
                         sycl::ext::intel::property::buffer_location::instance<3>>>
      accessorA;
  sycl::kernel_single_task<class kernel_function>(
      [=]() {
        accessorA.use();
        Obj.use();
      });
  return 0;
}
