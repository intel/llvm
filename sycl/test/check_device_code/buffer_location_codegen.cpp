// RUN: %clangxx -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// CHECK: define {{.*}}spir_kernel void @_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E15kernel_function{{.*}} !kernel_arg_buffer_location ![[MDBL:[0-9]+]]
// CHECK: ![[MDBL]] = !{i32 3, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 2, i32 -1, i32 -1, i32 -1, i32 2, i32 -1, i32 -1, i32 -1, i32 -1}

#include <CL/sycl.hpp>

struct Base {
  int A, B;
  cl::sycl::accessor<
      char, 1, cl::sycl::access::mode::read,
      cl::sycl::access::target::global_buffer,
      cl::sycl::access::placeholder::false_t,
      cl::sycl::ext::oneapi::accessor_property_list<
          cl::sycl::ext::intel::property::buffer_location::instance<2>>>
      AccField;
};

struct Captured
    : Base,
      cl::sycl::accessor<
          char, 1, cl::sycl::access::mode::read,
          cl::sycl::access::target::global_buffer,
          cl::sycl::access::placeholder::false_t,
          cl::sycl::ext::oneapi::accessor_property_list<
              cl::sycl::ext::intel::property::buffer_location::instance<2>>> {
  int C;
};

int main() {
  Captured Obj;
  cl::sycl::accessor<
      int, 1, cl::sycl::access::mode::read_write,
      cl::sycl::access::target::global_buffer,
      cl::sycl::access::placeholder::false_t,
      cl::sycl::ext::oneapi::accessor_property_list<
          cl::sycl::ext::intel::property::buffer_location::instance<3>>>
      accessorA;
  sycl::queue Queue;
  Queue.submit([&](sycl::handler &CGH) {
    CGH.single_task<class kernel_function>([=]() {
      (int)accessorA[0];
      (int)Obj[0];
    });
  });
  return 0;
}
