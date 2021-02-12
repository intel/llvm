// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl -fsycl-is-device -triple spir64 -Wno-sycl-2017-compat -verify | FileCheck %s

#include "Inputs/sycl.hpp"
// expected-warning@+2 {{attribute 'intelfpga::scheduler_target_fmax_mhz' is deprecated}}
// expected-note@+1 {{did you mean to use 'intel::scheduler_target_fmax_mhz' instead?}}
[[intelfpga::scheduler_target_fmax_mhz(2)]] void
func() {}

template <int N>
[[intel::scheduler_target_fmax_mhz(N)]] void zoo() {}

int main() {
  // CHECK-LABEL:  FunctionDecl {{.*}}test_kernel1 'void ()'
  // CHECK:        SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
  // CHECK-NEXT:   ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:   value: Int 5
  // CHECK-NEXT:   IntegerLiteral {{.*}} 'int' 5
  // expected-warning@+3 {{attribute 'intelfpga::scheduler_target_fmax_mhz' is deprecated}}
  // expected-note@+2 {{did you mean to use 'intel::scheduler_target_fmax_mhz' instead?}}
  cl::sycl::kernel_single_task<class test_kernel1>(
      []() [[intelfpga::scheduler_target_fmax_mhz(5)]]{});

  // CHECK-LABEL:  FunctionDecl {{.*}}test_kernel2 'void ()'
  // CHECK:        SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
  // CHECK-NEXT:   ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:   value: Int 2
  // CHECK-NEXT:   IntegerLiteral {{.*}} 'int' 2
  cl::sycl::kernel_single_task<class test_kernel2>(
      []() { func(); });

  // CHECK-LABEL:  FunctionDecl {{.*}}test_kernel3 'void ()'
  // CHECK:        SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
  // CHECK-NEXT:   ConstantExpr {{.*}} 'int'
  // CHECK-NEXT:   value: Int 75
  // CHECK-NEXT:   SubstNonTypeTemplateParmExpr {{.*}} 'int'
  // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 N
  // CHECK-NEXT:   IntegerLiteral {{.*}} 'int' 75
  cl::sycl::kernel_single_task<class test_kernel3>(
      []() { zoo<75>(); });

  [[intel::scheduler_target_fmax_mhz(0)]] int Var = 0; // expected-error{{'scheduler_target_fmax_mhz' attribute only applies to functions}}

  cl::sycl::kernel_single_task<class test_kernel4>(
      []() [[intel::scheduler_target_fmax_mhz(1048577)]]{}); // expected-error{{'scheduler_target_fmax_mhz' attribute requires integer constant between 0 and 1048576 inclusive}}

  cl::sycl::kernel_single_task<class test_kernel5>(
      []() [[intel::scheduler_target_fmax_mhz(-4)]]{}); // expected-error{{'scheduler_target_fmax_mhz' attribute requires integer constant between 0 and 1048576 inclusive}}

  cl::sycl::kernel_single_task<class test_kernel6>(
      []() [[intel::scheduler_target_fmax_mhz(1), intel::scheduler_target_fmax_mhz(2)]]{}); // expected-warning{{attribute 'scheduler_target_fmax_mhz' is already applied with different parameters}}
}
