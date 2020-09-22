// RUN: %clang_cc1 %s -fsycl -fsycl-is-device -triple spir64 -fsyntax-only -Wno-sycl-2017-compat -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 %s -fsycl -fsycl-is-device -triple spir64 -fsyntax-only -Wno-sycl-2017-compat -ast-dump | FileCheck %s

#ifndef __SYCL_DEVICE_ONLY__
struct FuncObj {
  [[intelfpga::num_simd_work_items(42)]] // expected-no-diagnostics
  void
  operator()() const {}
};

template <typename name, typename Func>
void kernel(const Func &kernelFunc) {
  kernelFunc();
}

void foo() {
  kernel<class test_kernel1>(
      FuncObj());
}

#else // __SYCL_DEVICE_ONLY__
[[intelfpga::num_simd_work_items(2)]] void func_do_not_ignore() {}

struct FuncObj {
  [[intelfpga::num_simd_work_items(42)]] void operator()() const {}
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
  // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
  // CHECK-NEXT:  IntegerLiteral{{.*}}42{{$}}
  kernel<class test_kernel1>(
      FuncObj());

  // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
  // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
  // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
  kernel<class test_kernel2>(
      []() [[intelfpga::num_simd_work_items(8)]] {});

  // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
  // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
  // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
  kernel<class test_kernel3>(
      []() { func_do_not_ignore(); });

#ifdef TRIGGER_ERROR
  [[intelfpga::num_simd_work_items(0)]] int Var = 0; // expected-error{{'num_simd_work_items' attribute only applies to functions}}

  kernel<class test_kernel4>(
      []() [[intelfpga::num_simd_work_items(0)]] {}); // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}

  kernel<class test_kernel5>(
      []() [[intelfpga::num_simd_work_items(-42)]] {}); // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}

  kernel<class test_kernel6>(
      []() [[intelfpga::num_simd_work_items(1), intelfpga::num_simd_work_items(2)]] {}); // expected-warning{{attribute 'num_simd_work_items' is already applied with different parameters}}
#endif // TRIGGER_ERROR
}
#endif // __SYCL_DEVICE_ONLY__
