// RUN: %clang_cc1 -fsycl -fsycl-is-device -Wno-return-type -fcxx-exceptions -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

struct FuncObj {
  //expected-warning@+2 {{attribute 'intelfpga::no_global_work_offset' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::no_global_work_offset' instead?}}
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
      []() [[intel::no_global_work_offset(0)]]{});

  // CHECK: SYCLIntelNoGlobalWorkOffsetAttr{{.*}}Enabled
  // expected-warning@+2{{'no_global_work_offset' attribute should be 0 or 1. Adjusted to 1}}
  kernel<class test_kernel3>(
      []() [[intel::no_global_work_offset(42)]]{});

  // expected-error@+2{{'no_global_work_offset' attribute requires a non-negative integral compile time constant expression}}
  kernel<class test_kernel4>(
      []() [[intel::no_global_work_offset(-1)]]{});

  // expected-error@+2{{'no_global_work_offset' attribute requires parameter 0 to be an integer constant}}
  kernel<class test_kernel5>(
      []() [[intel::no_global_work_offset("foo")]]{});

  kernel<class test_kernel6>([]() {
    // expected-error@+1{{'no_global_work_offset' attribute only applies to functions}}
    [[intel::no_global_work_offset(1)]] int a;
  });

  // CHECK: SYCLIntelNoGlobalWorkOffsetAttr{{.*}}
  // CHECK-NOT: Enabled
  // CHECK: SYCLIntelNoGlobalWorkOffsetAttr{{.*}}Enabled
  // expected-warning@+2{{attribute 'no_global_work_offset' is already applied}}
  kernel<class test_kernel7>(
      []() [[intel::no_global_work_offset(0), intel::no_global_work_offset(1)]]{});

  return 0;
}
