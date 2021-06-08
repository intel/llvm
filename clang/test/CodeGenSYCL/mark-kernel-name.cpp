// RUN: %clang_cc1 -triple x86_64-linux-pc -fsycl-is-host -disable-llvm-passes -fdeclare-spirv-builtins -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple spir64 -aux-triple x86_64-linux-pc  -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

#include "Inputs/sycl.hpp"

// This test validates that the use of __builtin_sycl_mark_kernel_name alters
// the code-gen'ed value of __builtin_unique_stable_name. In this case, lambda1
// emits the unmodified version like we do typically, while lambda2 is 'marked',
// so it should follow kernel naming (that is, using the E10000 naming).  Note
// that the top level kernel lambda (the E10000 in common) is automatically part
// of a kernel name, since it is passed to the kernel function (which is
// necessary so that the 'device' build actually emits the builtins.

int main() {

  cl::sycl::kernel_single_task<class K>([]() {
    auto lambda1 = []() {};
    auto lambda2 = []() {};

    (void)__builtin_sycl_unique_stable_name(decltype(lambda1));
    // CHECK: [35 x i8] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE_\00"

    // Should change the unique-stable-name of the lambda.
    (void)__builtin_sycl_mark_kernel_name(decltype(lambda2));
    (void)__builtin_sycl_unique_stable_name(decltype(lambda2));
    // CHECK: [40 x i8] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE10000_\00"
  });
}
