// RUN: %clang_cc1 %s -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -fsyntax-only -sycl-std=2017 -ast-dump | FileCheck %s

// The test checks AST of [[intel::num_simd_work_items()]] attribute.

#include "sycl.hpp"

using namespace sycl;
queue q;

// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
// CHECK: FunctionDecl {{.*}} {{.*}} funccc 'void ()'
// CHECK: SYCLIntelNumSimdWorkItemsAttr
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 2
// CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
// CHECK-NOT: SYCLIntelNumSimdWorkItemsAttr
[[intel::num_simd_work_items(2)]] [[intel::num_simd_work_items(2)]] void funccc() {}

// Test that checks template parameter support on member function of class template.
template <int SIZE>
class KernelFunctor {
public:
  [[intel::num_simd_work_items(SIZE)]] void operator()() {}
};

int kernel() {
  // no error expected
  KernelFunctor<10>();
  return 0;
}

// CHECK: ClassTemplateDecl {{.*}} {{.*}} KernelFunctor
// CHECK: ClassTemplateSpecializationDecl {{.*}} {{.*}} class KernelFunctor definition
// CHECK: CXXRecordDecl {{.*}} {{.*}} implicit class KernelFunctor
// CHECK: SYCLIntelNumSimdWorkItemsAttr
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}10{{$}}

// Test that checks template parameter support on function.
template <int N>
// CHECK: FunctionTemplateDecl {{.*}} {{.*}} funcc
// CHECK: NonTypeTemplateParmDecl {{.*}} {{.*}} referenced 'int' depth 0 index 0 N
// CHECK: FunctionDecl {{.*}} {{.*}} funcc 'void ()'
// CHECK: SYCLIntelNumSimdWorkItemsAttr
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
[[intel::num_simd_work_items(N)]] void funcc() {}

int ver() {
  // no error expected.
  funcc<8>();
  return 0;
}

[[intel::num_simd_work_items(2)]] void func_do_not_ignore() {}

struct FuncObj {
  [[intel::num_simd_work_items(42)]] void operator()() const {}
};

// If the declaration has a [[sycl::reqd_work_group_size()]]
// or [[cl::reqd_work_group_size()]] or
// __attribute__((reqd_work_group_size)) attribute, check to see
// if the last argument can be evenly divided by the
// [[intel::num_simd_work_items()]] attribute.
struct TRIFuncObjGood1 {
  [[intel::num_simd_work_items(4)]] [[sycl::reqd_work_group_size(3, 64, 4)]] void
  operator()() const {}
};

struct TRIFuncObjGood2 {
  [[sycl::reqd_work_group_size(3, 64, 4)]] [[intel::num_simd_work_items(4)]] void
  operator()() const {}
};

struct TRIFuncObjGood3 {
  [[intel::num_simd_work_items(4)]] [[sycl::reqd_work_group_size(3, 64, 4)]] void
  operator()() const {}
};

struct TRIFuncObjGood4 {
  [[sycl::reqd_work_group_size(3, 64, 4)]] [[intel::num_simd_work_items(4)]] void
  operator()() const {}
};

struct TRIFuncObjGood5 {
  [[intel::num_simd_work_items(5)]] void
  operator()() const;
};

[[sycl::reqd_work_group_size(3, 10, 5)]] void TRIFuncObjGood5::operator()() const {}

struct TRIFuncObjGood6 {
  [[sycl::reqd_work_group_size(3, 10, 5)]] void
  operator()() const;
};

[[intel::num_simd_work_items(5)]] void TRIFuncObjGood6::operator()() const {}

int main() {
  q.submit([&](handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 42
    // CHECK-NEXT:  IntegerLiteral{{.*}}42{{$}}
    h.single_task<class test_kernel1>(FuncObj());

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    h.single_task<class test_kernel2>(
        []() [[intel::num_simd_work_items(8)]] {});

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    h.single_task<class test_kernel3>(
        []() { func_do_not_ignore(); });

    h.single_task<class test_kernel4>(TRIFuncObjGood1());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel4
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK:       SYCLReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel5>(TRIFuncObjGood2());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel5
    // CHECK:       SYCLReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel6>(TRIFuncObjGood3());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel6
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK:       SYCLReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel7>(TRIFuncObjGood4());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel7
    // CHECK:       SYCLReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 64
    // CHECK-NEXT:  IntegerLiteral{{.*}}64{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class test_kernel8>(TRIFuncObjGood5());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel8
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 5
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}
    // CHECK:       SYCLReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 10
    // CHECK-NEXT:  IntegerLiteral{{.*}}10{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 5
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}

    h.single_task<class test_kernel9>(TRIFuncObjGood6());
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel9
    // CHECK:       SYCLReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 3
    // CHECK-NEXT:  IntegerLiteral{{.*}}3{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 10
    // CHECK-NEXT:  IntegerLiteral{{.*}}10{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 5
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}
    // CHECK:       SYCLIntelNumSimdWorkItemsAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 5
    // CHECK-NEXT:  IntegerLiteral{{.*}}5{{$}}

    // Ignore duplicate attribute.
    h.single_task<class test_kernel10>(
        // CHECK-LABEL: FunctionDecl {{.*}}test_kernel10
        // CHECK: SYCLIntelNumSimdWorkItemsAttr
        // CHECK-NEXT: ConstantExpr {{.*}} 'int'
        // CHECK-NEXT: value: Int 8
        // CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
        // CHECK-NOT: SYCLIntelNumSimdWorkItemsAttr
        []() [[intel::num_simd_work_items(8),
               intel::num_simd_work_items(8)]] {});
  });
  return 0;
}
