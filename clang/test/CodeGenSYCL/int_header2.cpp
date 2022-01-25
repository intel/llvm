// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

// CHECK: template <> struct KernelInfo<KernelName> {
// CHECK: static constexpr const char* getFileName() { return clang/test/CodeGenSYCL/int_header2.cpp; }
// CHECK: static constexpr const char* getFunctionName() { return test; }
// CHECK: static constexpr unsigned getLineNumber() { return 18; }
// CHECK: static constexpr unsigned getColumnNumber() { return 52; }

#include "/iusers2/schittir/workspaces/publicsycl/llvm-1/clang/test/CodeGenSYCL/Inputs/sycl.hpp"

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void kernel_single_task(const KernelType &kernelFunc) {
  kernelFunc();
}

int test() {
    cl::sycl::kernel_single_task<class KernelName>([](){});
  return 0;
}

int main() {
  test();
}
