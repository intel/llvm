// RUN: %clang_cc1 -Wno-return-type -fsycl-is-device -fcxx-exceptions -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

struct FuncObj {
  [[intelfpga::uses_global_work_offset(1)]] void operator()() {}
};

template <typename name, typename Func>
void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  // CHECK: SYCLIntelUsesGlobalWorkOffsetAttr{{.*}}Enabled
  kernel<class test_kernel1>([]() {
    FuncObj();
  });

  // CHECK: SYCLIntelUsesGlobalWorkOffsetAttr
  // CHECK-NOT: Enabled
  kernel<class test_kernel2>(
      []() [[intelfpga::uses_global_work_offset(0)]]{});

  // CHECK: SYCLIntelUsesGlobalWorkOffsetAttr{{.*}}Enabled
  // expected-warning@+2{{attribute should be 0 or 1. Adjusted to 1}}
  kernel<class test_kernel3>(
      []() [[intelfpga::uses_global_work_offset(42)]]{});

  // expected-error@+2{{attribute requires a non-negative integral compile time constant expression}}
  kernel<class test_kernel4>(
      []() [[intelfpga::uses_global_work_offset(-1)]]{});

  // expected-error@+2{{attribute requires parameter 0 to be an integer constant}}
  kernel<class test_kernel5>(
      []() [[intelfpga::uses_global_work_offset("foo")]]{});

  kernel<class test_kernel6>([]() {
    // expected-error@+1{{attribute only applies to functions}}
    [[intelfpga::uses_global_work_offset(1)]] int a;
  });

  // CHECK: SYCLIntelUsesGlobalWorkOffsetAttr{{.*}}
  // CHECK-NOT: Enabled
  // CHECK: SYCLIntelUsesGlobalWorkOffsetAttr{{.*}}Enabled
  // expected-warning@+2{{attribute 'uses_global_work_offset' is already applied}}
  kernel<class test_kernel7>(
      []() [[intelfpga::uses_global_work_offset(0), intelfpga::uses_global_work_offset(1)]]{});

  return 0;
}
