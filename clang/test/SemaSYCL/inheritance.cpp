// RUN: %clang_cc1 -fsycl-is-device -ast-dump %s | FileCheck %s

#include "Inputs/sycl.hpp"

class second_base {
public:
  int *e;
};

class InnerFieldBase {
public:
  int d;
};
class InnerField : public InnerFieldBase {
  int c;
};

struct base {
public:
  int b;
  InnerField obj;
};

struct derived : base, second_base {
  int a;

  void operator()() const {
  }
};

int main() {
  cl::sycl::queue q;

  q.submit([&](cl::sycl::handler &cgh) {
    derived f{};
    cgh.single_task(f);
  });

  return 0;
}

// Check declaration of the kernel
// CHECK: derived{{.*}} 'void (base, __wrapper_class, int)

// Check parameters of the kernel
// CHECK: ParmVarDecl {{.*}} used _arg__base 'base'
// CHECK: ParmVarDecl {{.*}} used _arg_e '__wrapper_class'
// CHECK: ParmVarDecl {{.*}} used _arg_a 'int'

// Check initializers for derived and base classes.
// Each class has it's own initializer list
// Base classes should be initialized first.
// CHECK: VarDecl {{.*}} derived 'derived' cinit
// CHECK-NEXT: InitListExpr {{.*}} 'derived'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'base' 'void (const base &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const base' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue ParmVar {{.*}} '_arg__base' 'base'
// CHECK-NEXT: InitListExpr {{.*}} 'second_base'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue ParmVar {{.*}} '_arg_e' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue ParmVar {{.*}} '_arg_a' 'int'
