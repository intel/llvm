// RUN: %clang_cc1 %s -fsyntax-only -fsycl -fsycl-is-device -triple spir64 -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl -fsycl-is-device -triple spir64 | FileCheck %s
// RUN: %clang_cc1 -fsycl -fsycl-is-host -fsyntax-only -verify %s

#ifndef __SYCL_DEVICE_ONLY__
struct FuncObj {
  [[intelfpga::max_work_group_size(1, 1, 1)]] // expected-no-diagnostics
  void operator()() {}
};

template <typename name, typename Func>
void kernel(Func kernelFunc) {
  kernelFunc();
}

void foo() {
  kernel<class test_kernel1>(
      FuncObj());
}

#else // __SYCL_DEVICE_ONLY__

[[intelfpga::max_work_group_size(2, 2, 2)]]
void func_do_not_ignore() {}

struct FuncObj {
  [[intelfpga::max_work_group_size(4, 4, 4)]]
  void operator()() {}
};

#ifdef TRIGGER_ERROR
struct DAFuncObj {
  [[intelfpga::max_work_group_size(4, 4, 4)]]
  [[cl::reqd_work_group_size(8, 8, 4)]] // expected-error{{'reqd_work_group_size' attribute conflicts with 'max_work_group_size' attribute}}
  void operator()() {}
};
#endif // TRIGGER_ERROR

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
  // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}} 4 4 4
  kernel<class test_kernel1>(
      FuncObj());

  // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
  // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}} 8 8 8
  kernel<class test_kernel2>(
      []() [[intelfpga::max_work_group_size(8, 8, 8)]] {});

  // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
  // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}}
  kernel<class test_kernel3>(
      []() {func_do_not_ignore();});

#ifdef TRIGGER_ERROR
  [[intelfpga::max_work_group_size(1, 1, 1)]] int Var = 0; // expected-error{{'max_work_group_size' attribute only applies to functions}}

  kernel<class test_kernel4>(
      []() [[intelfpga::max_work_group_size(0, 1, 3)]] {}); // expected-error{{'max_work_group_size' attribute must be greater than 0}}

  kernel<class test_kernel5>(
      []() [[intelfpga::max_work_group_size(-8, 8, 1)]] {}); // expected-error{{'max_work_group_size' attribute requires a non-negative integral compile time constant expression}}

  kernel<class test_kernel6>(
      []() [[intelfpga::max_work_group_size(16, 16, 16),
             intelfpga::max_work_group_size(2, 2, 2)]] {}); // expected-warning{{attribute 'max_work_group_size' is already applied with different parameters}}

  kernel<class test_kernel7>(
      DAFuncObj());

#endif // TRIGGER_ERROR
}
#endif // __SYCL_DEVICE_ONLY__
