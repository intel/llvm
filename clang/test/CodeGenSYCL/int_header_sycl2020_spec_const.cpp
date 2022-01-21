// Generic SPIR-V target
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -fsycl-int-header=%t.h %s -o %t.out %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s
//
// SPIR-V AOT targets
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64_gen-unknown-unknown -fsycl-int-header=%t.h %s -o %t.out %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64_x86_64-unknown-unknown -fsycl-int-header=%t.h %s -o %t.out %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64_fpga-unknown-unknown -fsycl-int-header=%t.h %s -o %t.out %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s
//
// Non-SPIR target
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-unknown-unknown -fsycl-int-header=%t.h %s -o %t.out %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

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
// CHECK: const kernel_param_desc_t kernel_signatures[] = {
// CHECK: { kernel_param_kind_t::kind_specialization_constants_buffer, 8, 0 }
