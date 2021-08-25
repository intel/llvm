// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -ast-dump %s | FileCheck %s

// Tests for AST of [[intel::scheduler_target_fmax_mhz()]], [[intel::num_simd_work_items()]],
// [[intel::no_global_work_offset()]], [[intel::max_global_work_dim()]], [[intel::sycl_explicit_simd]],
// [[sycl::reqd_sub_group_size()]], [[sycl::reqd_work_group_size()]], [[intel::kernel_args_restrict]], and
// [[intel::max_work_group_size()]] function attributes in SYCL 2020.

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
  [[sycl::reqd_work_group_size(2, 2, 2)]] void operator()() const {}
};

struct FuncObj5 {
  [[intel::num_simd_work_items(8)]] void operator()() const {}
};

struct FuncObj6 {
  [[intel::kernel_args_restrict]] void operator()() const {}
};

struct FuncObj7 {
  [[intel::max_global_work_dim(1)]] void operator()() const {}
};

[[intel::sycl_explicit_simd]] void func() {}

[[intel::no_global_work_offset(1)]] void func1() {}

[[intel::scheduler_target_fmax_mhz(2)]] void func2() {}

[[intel::max_work_group_size(1, 1, 1)]] void func3() {}

[[sycl::reqd_work_group_size(1, 1, 1)]] void func4() {}

[[intel::num_simd_work_items(5)]] void func5() {}

[[intel::kernel_args_restrict]] void func6() {}

[[intel::max_global_work_dim(0)]] void func7() {}

[[sycl::reqd_sub_group_size(4)]] void func8() {}

class Functor {
public:
  void operator()() const {
    func8();
  }
};

class Functor1 {
public:
  [[sycl::reqd_sub_group_size(12)]] void operator()() const {}
};

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    // CHECK:       FunctionDecl {{.*}}test_kernel1
    // CHECK:       SYCLSimdAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLKernelAttr {{.*}} Implicit
    // CHECK-NEXT:  AsmLabelAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLSimdAttr
    h.single_task<class test_kernel1>(
        FuncObj());
    // CHECK:       FunctionDecl {{.*}}test_kernel2
    // CHECK:       SYCLSimdAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLKernelAttr {{.*}} Implicit
    // CHECK-NEXT:  AsmLabelAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLSimdAttr
    h.single_task<class test_kernel2>(
        []() [[intel::sycl_explicit_simd]]{});

    // Test attribute is not propagated.
    // CHECK:      FunctionDecl {{.*}}test_kernel3
    // CHECK:      SYCLSimdAttr {{.*}} Implicit
    // CHECK-NEXT: SYCLKernelAttr {{.*}} Implicit
    // CHECK-NEXT:  AsmLabelAttr {{.*}} Implicit
    // CHECK-NEXT: SYCLSimdAttr
    // CHECK-NOT:  SYCLSimdAttr
    h.single_task<class test_kernel3>(
        []() [[intel::sycl_explicit_simd]] { func(); });

    // Test attribute is not propagated.
    // CHECK:      FunctionDecl {{.*}}test_kernel4
    // CHECK-NOT:  SYCLIntelNoGlobalWorkOffsetAttr
    h.single_task<class test_kernel4>(
        []() { func1(); });

    // CHECK:      FunctionDecl {{.*}}test_kernel5
    // CHECK:      SYCLIntelNoGlobalWorkOffsetAttr
    // CHECK-NEXT: ConstantExpr {{.*}} 'int'
    // CHECK-NEXT: value: Int 1
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
    h.single_task<class test_kernel5>(
        FuncObj1());

    // CHECK:       FunctionDecl {{.*}}test_kernel6
    // CHECK:       SYCLIntelNoGlobalWorkOffsetAttr
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 1
    h.single_task<class test_kernel6>(
        []() [[intel::no_global_work_offset]]{});

    // CHECK:       FunctionDecl {{.*}}test_kernel7
    // CHECK:       SYCLIntelSchedulerTargetFmaxMhzAttr
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 10
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 10
    h.single_task<class test_kernel7>(
        FuncObj2());

    // CHECK:       FunctionDecl {{.*}}test_kernel8
    // CHECK:       SYCLIntelSchedulerTargetFmaxMhzAttr
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 20
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 20
    h.single_task<class test_kernel8>(
        []() [[intel::scheduler_target_fmax_mhz(20)]]{});

    // Test attribute is not propagated.
    // CHECK:      FunctionDecl {{.*}}test_kernel9
    // CHECK-NOT:  SYCLIntelSchedulerTargetFmaxMhzAttr
    h.single_task<class test_kernel9>(
        []() { func2(); });

    // CHECK:       FunctionDecl {{.*}}test_kernel10
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 2
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 2
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 2
    h.single_task<class test_kernel10>(
        FuncObj3());

    // Test attribute is not propagated.
    // CHECK:     FunctionDecl {{.*}}test_kernel11
    // CHECK-NOT: SYCLIntelMaxWorkGroupSizeAttr
    h.single_task<class test_kernel11>(
        []() { func3(); });

    // CHECK:       FunctionDecl {{.*}}test_kernel12
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 8
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 8
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 8
    h.single_task<class test_kernel12>(
        []() [[intel::max_work_group_size(8, 8, 8)]]{});

    // CHECK:       FunctionDecl {{.*}}test_kernel13
    // CHECK:       ReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 2
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 2
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 2
    h.single_task<class test_kernel13>(
        FuncObj4());

    // Test attribute is not propagated.
    // CHECK:      FunctionDecl {{.*}}test_kernel14
    // CHECK-NOT:  ReqdWorkGroupSizeAttr
    h.single_task<class test_kernel14>(
        []() { func4(); });

    // CHECK:       FunctionDecl {{.*}}test_kernel15
    // CHECK:       ReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 8
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 8
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 8
    h.single_task<class test_kernel15>(
        []() [[sycl::reqd_work_group_size(8, 8, 8)]]{});

    // CHECK:       FunctionDecl {{.*}}test_kernel16
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 8
    h.single_task<class test_kernel16>(
        FuncObj5());

    // CHECK:       FunctionDecl {{.*}}test_kernel17
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 20
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 20
    h.single_task<class test_kernel17>(
        []() [[intel::num_simd_work_items(20)]]{});

    // Test attribute is not propagated.
    // CHECK:     FunctionDecl {{.*}}test_kernel18
    // CHECK-NOT: SYCLIntelNumSimdWorkItemsAttr
    h.single_task<class test_kernel18>(
        []() { func5(); });

    // CHECK: FunctionDecl {{.*}}test_kernel19
    // CHECK: SYCLIntelKernelArgsRestrictAttr
    h.single_task<class test_kernel19>(
        FuncObj6());

    // CHECK: FunctionDecl {{.*}}test_kernel20
    // CHECK: SYCLIntelKernelArgsRestrictAttr
    h.single_task<class test_kernel20>(
        []() [[intel::kernel_args_restrict]]{});

    // Test attribute is not propagated.
    // CHECK:     FunctionDecl {{.*}}test_kernel21
    // CHECK-NOT: SYCLIntelKernelArgsRestrictAttr
    h.single_task<class test_kernel21>(
        []() { func6(); });

    // CHECK: FunctionDecl {{.*}}test_kernel22
    // CHECK: SYCLIntelMaxGlobalWorkDimAttr
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 1
    h.single_task<class test_kernel22>(
        FuncObj7());

    // CHECK: FunctionDecl {{.*}}test_kernel23
    // CHECK: SYCLIntelMaxGlobalWorkDimAttr
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 0
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 0
    h.single_task<class test_kernel23>(
        []() [[intel::max_global_work_dim(0)]]{});

    // Test attribute is not propagated.
    // CHECK:     FunctionDecl {{.*}}test_kernel24
    // CHECK-NOT: SYCLIntelMaxGlobalWorkDimAttr
    h.single_task<class test_kernel24>(
        []() { func7(); });

    // Test attribute is not propagated.
    // CHECK:     FunctionDecl {{.*}}test_kernel25
    // CHECK-NOT: IntelReqdSubGroupSizeAttr
    Functor f;
    h.single_task<class test_kernel25>(f);

    // CHECK:     FunctionDecl {{.*}}test_kernel26
    // CHECK:     IntelReqdSubGroupSizeAttr
    Functor1 f1;
    h.single_task<class test_kernel26>(f1);

    // CHECK:       FunctionDecl {{.*}}test_kernel27
    // CHECK:       IntelReqdSubGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 8
    h.single_task<class test_kernel27>(
        []() [[sycl::reqd_sub_group_size(8)]]{});
  });
  return 0;
}
