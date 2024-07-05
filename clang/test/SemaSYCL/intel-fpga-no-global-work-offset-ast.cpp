// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump %s | FileCheck %s

// Tests for AST of Intel FPGA no_global_work_offset function attribute.

#include "sycl.hpp"

using namespace sycl;
queue q;

struct FuncObj {
  [[intel::no_global_work_offset]] void operator()() const {}
};

// CHECK: FunctionDecl {{.*}} func 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelNoGlobalWorkOffsetAttr
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
[[intel::no_global_work_offset(1)]] void func() {}

class KernelFunctor {
public:
  void operator()() const {
    func();
  }
};

// CHECK: FunctionTemplateDecl {{.*}} func1
// CHECK: FunctionDecl {{.*}} func1 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelNoGlobalWorkOffsetAttr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'
// CHECK: FunctionDecl {{.*}} func1 'void ()'
// CHECK-NEXT: TemplateArgument integral 1
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelNoGlobalWorkOffsetAttr
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1

// Test that checks template parameter support on function.
template <int N>
[[intel::no_global_work_offset(N)]] void func1() {}

// Test that checks template parameter support on member function of class template.
template <int SIZE>
class KernelFunctor2 {
public:
  [[intel::no_global_work_offset(SIZE)]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    // CHECK: FunctionDecl {{.*}}test_kernel1
    // CHECK:       SYCLIntelNoGlobalWorkOffsetAttr
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    h.single_task<class test_kernel1>(FuncObj());

    // CHECK: FunctionDecl {{.*}}test_kernel2
    // CHECK: SYCLIntelNoGlobalWorkOffsetAttr
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 0
    // CHECK-NEXT:  IntegerLiteral{{.*}}0{{$}}
    h.single_task<class test_kernel2>(
        []() [[intel::no_global_work_offset(0)]] {});

    // CHECK: FunctionDecl {{.*}}test_kernel3
    // CHECK: SYCLIntelNoGlobalWorkOffsetAttr
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 42
    // CHECK-NEXT:  IntegerLiteral{{.*}}42{{$}}
    h.single_task<class test_kernel3>(
        []() [[intel::no_global_work_offset(42)]] {});

    // CHECK: FunctionDecl {{.*}}test_kernel4
    // CHECK: SYCLIntelNoGlobalWorkOffsetAttr
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int -1
    // CHECK-NEXT: UnaryOperator{{.*}} 'int' prefix '-'
    // CHECK-NEXT-NEXT: IntegerLiteral{{.*}}1{{$}}
    h.single_task<class test_kernel4>(
        []() [[intel::no_global_work_offset(-1)]] {});

    // Ignore duplicate attribute.
    h.single_task<class test_kernel5>(
        // CHECK: FunctionDecl {{.*}}test_kernel5
        // CHECK:       SYCLIntelNoGlobalWorkOffsetAttr
        // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
        // CHECK-NEXT:  value: Int 1
        // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
        // CHECK-NOT:   SYCLIntelNoGlobalWorkOffsetAttr
        []() [[intel::no_global_work_offset,
               intel::no_global_work_offset]] {}); // OK

    // Test attribute does not get propagated.
    // CHECK: FunctionDecl {{.*}}test_kernel6
    // CHECK-NOT: SYCLIntelLoopFuseAttr
    KernelFunctor f1;
    h.single_task<class test_kernel6>(f1);

    // CHECK: FunctionDecl {{.*}}test_kernel7
    // CHECK: SYCLIntelNoGlobalWorkOffsetAttr
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 1
    // CHECK-NEXT: SubstNonTypeTemplateParmExpr
    // CHECK-NEXT: NonTypeTemplateParmDecl
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
    KernelFunctor2<1> f2;
    h.single_task<class test_kernel7>(f2);
  });
  func1<1>();
  return 0;
}
