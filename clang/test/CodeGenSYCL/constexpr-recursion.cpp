// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks that SYCL compiler allows compile-time evaluated recursion
// and that the evaluated function is really gone from the generated IR.

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc(); //#call_kernelFunc // expected-note 3{{called by 'kernel_single_task<fake_kernel, (lambda at}}
}

// Compiler must evaluate recursion, no errors expected.
static constexpr unsigned int getNextPowerOf2(unsigned int n,
                                              unsigned int k = 1) {
  return (k >= n) ? k : getNextPowerOf2(n, k * 2);
}
// CHECK-NOT: getNextPowerOf2

unsigned test_constexpr_recursion(unsigned int val) {
  unsigned int res = val;
  unsigned int *addr = &res;

  kernel_single_task<class ConstexprRecursionKernel>([=]() {
    constexpr unsigned int x = getNextPowerOf2(3);
    *addr += x;
  });
  return res;
}
