// RUN: %clang -I %S/Inputs -fsycl -fsycl-explicit-simd -fsycl-device-only \
// RUN:  -Xclang -fsycl-int-header=%t.h %s -c
// RUN: FileCheck -input-file=%t.h %s

// This test checks that compiler generates correct integration header elements
// for an accessor in ESIMD mode.

#include <sycl.hpp>

template <typename Acc>
struct AccWrapper { Acc accessor; };

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> acc;
  kernel<class kernel_wrapper>(
      [=]() {
        acc.use();
      });
}
// CHECK: static constexpr
// CHECK: static constexpr
// CHECK-NEXT: const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:  //--- {{[_a-zA-Z0-9]+}}kernel_wrapper
// CHECK-NEXT:  { kernel_param_kind_t::kind_esimd_buffer_accessor, {{[0-9]+}}, {{[0-9]+}} },
