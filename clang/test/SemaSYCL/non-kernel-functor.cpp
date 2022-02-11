// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown  -verify %s

// Test to verify that non-kernel functors are not processed as SYCL kernel
// functors

// expected-no-diagnostics
class First {
public:
  void operator()() { return; }
};

class Second {
public:
  First operator()() { return First(); }
};

SYCL_EXTERNAL
void foo() {
  Second NonKernelFunctorObj;
  NonKernelFunctorObj()();
}
