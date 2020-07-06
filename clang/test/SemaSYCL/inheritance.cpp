// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s

#include <sycl.hpp>

class second_base {
public:
  int e;
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

  void operator()() {
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
// CHECK: derived{{.*}} 'void (int, int, int, int, int)'

// Check parameters of the kernel
// CHECK: ParmVarDecl {{.*}} used _arg_b 'int'
// CHECK: ParmVarDecl {{.*}} used _arg_d 'int'
// CHECK: ParmVarDecl {{.*}} used _arg_c 'int'
// CHECK: ParmVarDecl {{.*}} used _arg_e 'int'
// CHECK: ParmVarDecl {{.*}} used _arg_a 'int'

// Check initializers for derived and base classes.
// Each class has it's own initializer list
// Base classes should be initialized first.
// CHECK: VarDecl {{.*}} derived 'derived' cinit
// CHECK-NEXT: InitListExpr {{.*}} 'derived'
// CHECK-NEXT: InitListExpr {{.*}} 'base'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue ParmVar {{.*}} '_arg_b' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'InnerField'
// CHECK-NEXT: InitListExpr {{.*}} 'InnerFieldBase'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue ParmVar {{.*}} '_arg_d' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue ParmVar {{.*}} '_arg_c' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'second_base'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue ParmVar {{.*}} '_arg_e' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue ParmVar {{.*}} '_arg_a' 'int'
