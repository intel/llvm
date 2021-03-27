// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-unknown-unknown -fsycl-int-header=%t.h %s -o %t.out %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s --check-prefix=NONATIVESUPPORT --check-prefix=ALL
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -o %t.out %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s --check-prefix=NATIVESUPPORT --check-prefix=ALL

// This test checks that the compiler generates required information
// in integration header for kernel_handler type (SYCL 2020 specialization
// constants).

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

int main() {
  q.submit([&](handler &h) {
    int a;
    kernel_handler kh;

    h.single_task<class test_kernel_handler>(
        [=](auto) {
          int local = a;
        },
        kh);
  });
}
// ALL: const kernel_param_desc_t kernel_signatures[] = {
// NONATIVESUPPORT: { kernel_param_kind_t::kind_specialization_constants_buffer, 8, 0 }
// NATIVESUPPORT-NOT: { kernel_param_kind_t::kind_specialization_constants_buffer, 8, 0 }
