// RUN: %clang_cc1 -Wno-return-type -fsycl-is-device -fcxx-exceptions -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

struct FuncObj {
  [[intelfpga::no_global_work_offset]] void operator()() {}
};

template <typename name, typename Func>
void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  // CHECK: SYCLIntelNoGlobalWorkOffsetAttr{{.*}}Enabled
  kernel<class test_kernel1>([]() {
    FuncObj();
  });

  // CHECK: SYCLIntelNoGlobalWorkOffsetAttr
  // CHECK-NOT: Enabled
  kernel<class test_kernel2>(
      []() [[intelfpga::no_global_work_offset(0)]]{});

  // CHECK: SYCLIntelNoGlobalWorkOffsetAttr{{.*}}Enabled
  // expected-warning@+2{{'no_global_work_offset' attribute should be 0 or 1. Adjusted to 1}}
  kernel<class test_kernel3>(
      []() [[intelfpga::no_global_work_offset(42)]]{});

  // expected-error@+2{{'no_global_work_offset' attribute requires a non-negative integral compile time constant expression}}
  kernel<class test_kernel4>(
      []() [[intelfpga::no_global_work_offset(-1)]]{});

  // expected-error@+2{{'no_global_work_offset' attribute requires parameter 0 to be an integer constant}}
  kernel<class test_kernel5>(
      []() [[intelfpga::no_global_work_offset("foo")]]{});

  kernel<class test_kernel6>([]() {
    // expected-error@+1{{'no_global_work_offset' attribute only applies to functions}}
    [[intelfpga::no_global_work_offset(1)]] int a;
  });

  // CHECK: SYCLIntelNoGlobalWorkOffsetAttr{{.*}}
  // CHECK-NOT: Enabled
  // CHECK: SYCLIntelNoGlobalWorkOffsetAttr{{.*}}Enabled
  // expected-warning@+2{{attribute 'no_global_work_offset' is already applied}}
  kernel<class test_kernel7>(
      []() [[intelfpga::no_global_work_offset(0), intelfpga::no_global_work_offset(1)]]{});

  return 0;
}
