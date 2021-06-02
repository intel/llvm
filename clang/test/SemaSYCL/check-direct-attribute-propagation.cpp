// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -sycl-std=2020 -ast-dump %s | FileCheck %s

// Tests to validate the SYCL 2020 requirement mandating the avoidance of the propagation of all kernel attributes to the caller when used on a function.

#include "sycl.hpp"

sycl::queue deviceQueue;

struct FuncObj {
  [[intel::sycl_explicit_simd]] void operator()() const {}
};

struct FuncObj1 {
  [[intel::no_global_work_offset(1)]] void operator()() const {}
};

struct FuncObj2 {
  [[intel::scheduler_target_fmax_mhz(10)]] void operator()() const {}
};

struct FuncObj3 {
  [[intel::max_work_group_size(2, 2, 2)]] void operator()() const {}
};

struct FuncObj4 {
  [[intel::reqd_work_group_size(2, 2, 2)]] void operator()() const {}
};

struct FuncObj5 {
  [[intel::num_simd_work_items(8)]] void operator()() const {}
};

struct FuncObj6 {
  [[intel::kernel_args_restrict]] void operator()() const {}
};

[[intel::sycl_explicit_simd]] void func() {}

[[intel::no_global_work_offset(1)]] void func1() {}

[[intel::scheduler_target_fmax_mhz(2)]] void func2() {}

[[intel::max_work_group_size(1, 1, 1)]] void func3() {}

[[intel::reqd_work_group_size(1, 1, 1)]] void func4() {}

[[intel::num_simd_work_items(5)]] void func5() {}

[[intel::kernel_args_restrict]] void func6() {}

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    // Test attribute directly applies on kernel functor.
    // CHECK:       FunctionDecl {{.*}}test_kernel1
    // CHECK:       SYCLSimdAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLKernelAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLSimdAttr {{.*}}
    h.single_task<class test_kernel1>(
        FuncObj());
    // Test attribute directly applies on kernel lambda.
    // CHECK:       FunctionDecl {{.*}}test_kernel2
    // CHECK:       SYCLSimdAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLKernelAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLSimdAttr {{.*}}
    h.single_task<class test_kernel2>(
        []() [[intel::sycl_explicit_simd]]{});

    // Test attribute is not propagated from function.
    // CHECK:      FunctionDecl {{.*}}test_kernel3
    // CHECK:      SYCLSimdAttr {{.*}} Implicit
    // CHECK-NEXT: SYCLKernelAttr {{.*}} Implicit
    // CHECK-NEXT: SYCLSimdAttr {{.*}}
    // CHECK-NOT:  SYCLSimdAttr {{.*}}
    h.single_task<class test_kernel3>(
        []() [[intel::sycl_explicit_simd]] { func(); });

    // Test attribute is not propagated from function.
    // CHECK:      FunctionDecl {{.*}}test_kernel4
    // CHECK-NOT:  SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
    h.single_task<class test_kernel4>(
        []() { func1(); });

    // Test attribute directly applies on kernel functor.
    // CHECK:      FunctionDecl {{.*}}test_kernel5
    // CHECK:      SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
    // CHECK-NEXT: ConstantExpr {{.*}} 'int'
    // CHECK-NEXT: value: Int 1
    // CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
    h.single_task<class test_kernel5>(
        FuncObj1());

    // Test attribute directly applies on kernel lambda.
    // CHECK:       FunctionDecl {{.*}}test_kerne6
    // CHECK:       SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    h.single_task<class test_kerne6>(
        []() [[intel::no_global_work_offset]]{});

    // Test attribute directly applies on kernel functor.
    // CHECK:       FunctionDecl {{.*}}test_kernel7
    // CHECK:       SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 10
    // CHECK-NEXT:  IntegerLiteral{{.*}}10{{$}}
    h.single_task<class test_kernel7>(
        FuncObj2());

    // Test attribute directly applies on kernel lambda.
    // CHECK:       FunctionDecl {{.*}}test_kernel8
    // CHECK:       SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 20
    // CHECK-NEXT:  IntegerLiteral{{.*}}20{{$}}
    h.single_task<class test_kernel8>(
        []() [[intel::scheduler_target_fmax_mhz(20)]]{});

    // Test attribute is not propagated from function.
    // CHECK:      FunctionDecl {{.*}}test_kernel9
    // CHECK-NOT:  SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
    h.single_task<class test_kernel9>(
        []() { func2(); });

    // Test attribute directly applies on kernel functor.
    // CHECK:       FunctionDecl {{.*}}test_kernel10
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    h.single_task<class test_kernel10>(
        FuncObj3());

    // Test attribute is not propagated from function.
    // CHECK:     FunctionDecl {{.*}}test_kernel11
    // CHECK-NOT: SYCLIntelMaxWorkGroupSizeAttr {{.*}}
    h.single_task<class test_kernel11>(
        []() { func3(); });

    // Test attribute directly applies on kernel lambda.
    // CHECK:       FunctionDecl {{.*}}test_kernel12
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    h.single_task<class test_kernel12>(
        []() [[intel::max_work_group_size(8, 8, 8)]]{});

    // Test attribute directly applies on kernel functor.
    // CHECK:       FunctionDecl {{.*}}test_kernel13
    // CHECK:       ReqdWorkGroupSizeAttr{{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    h.single_task<class test_kernel13>(
        FuncObj4());

    // Test attribute is not propagated from function.
    // CHECK:      FunctionDecl {{.*}}test_kernel14
    // CHECK-NOT:  ReqdWorkGroupSizeAttr {{.*}}
    h.single_task<class test_kernel14>(
        []() { func4(); });

    // Test attribute directly applies on kernel lambda.
    // CHECK:       FunctionDecl {{.*}}test_kernel15
    // CHECK:       ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    h.single_task<class test_kernel15>(
        []() [[intel::reqd_work_group_size(8, 8, 8)]]{});

    // Test attribute directly applies on kernel functor.
    // CHECK:       FunctionDecl {{.*}}test_kernel16
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr  {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    h.single_task<class test_kernel16>(
        FuncObj5());

    // Test attribute directly applies on kernel lambda.
    // CHECK:       FunctionDecl {{.*}}test_kernel17
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 20
    // CHECK-NEXT:  IntegerLiteral{{.*}}20{{$}}
    h.single_task<class test_kernel17>(
        []() [[intel::num_simd_work_items(20)]]{});

    // Test attribute is not propagated from function.
    // CHECK:     FunctionDecl {{.*}}test_kernel18
    // CHECK-NOT: SYCLIntelNumSimdWorkItemsAttr {{.*}}
    h.single_task<class test_kernel18>(
        []() { func5(); });

    // Test attribute directly applies on kernel functor.
    // CHECK: FunctionDecl {{.*}}test_kernel19
    // CHECK: SYCLIntelKernelArgsRestrictAttr {{.*}}
    h.single_task<class test_kernel19>(
        FuncObj6());

    // Test attribute directly applies on kernel lambda.
    // CHECK: FunctionDecl {{.*}}test_kernel20
    // CHECK: SYCLIntelKernelArgsRestrictAttr {{.*}}
    h.single_task<class test_kernel20>(
        []() [[intel::kernel_args_restrict]]{});

    // Test attribute is not propagated from functiom.
    // CHECK:     FunctionDecl {{.*}}test_kernel21
    // CHECK-NOT: SYCLIntelKernelArgsRestrictAttr {{.*}}
    h.single_task<class test_kernel21>(
        []() { func6(); });

  });
  return 0;
}
