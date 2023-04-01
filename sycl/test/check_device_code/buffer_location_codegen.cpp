// RUN: %clangxx -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// CHECK: define {{.*}}spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E15kernel_function{{.*}} !kernel_arg_buffer_location ![[MDBL:[0-9]+]]
// CHECK: ![[MDBL]] = !{i32 3, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 2, i32 -1, i32 -1, i32 -1, i32 2, i32 -1, i32 -1, i32 -1, i32 -1}

#include <sycl/sycl.hpp>

struct Base {
  int A, B;
  sycl::accessor<char, 1, sycl::access::mode::read,
                 sycl::access::target::device,
                 sycl::access::placeholder::false_t,
                 sycl::ext::oneapi::accessor_property_list<
                     sycl::ext::intel::property::buffer_location::instance<2>>>
      AccField;
};

struct Captured
    : Base,
      sycl::accessor<
          char, 1, sycl::access::mode::read, sycl::access::target::device,
          sycl::access::placeholder::false_t,
          sycl::ext::oneapi::accessor_property_list<
              sycl::ext::intel::property::buffer_location::instance<2>>> {
  int C;
};

int main() {
  Captured Obj;
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                 sycl::access::target::device,
                 sycl::access::placeholder::false_t,
                 sycl::ext::oneapi::accessor_property_list<
                     sycl::ext::intel::property::buffer_location::instance<3>>>
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
