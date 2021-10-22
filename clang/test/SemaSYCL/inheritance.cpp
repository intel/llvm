// RUN: %clang_cc1 -fsycl-is-device -ast-dump %s | FileCheck %s

#include "Inputs/sycl.hpp"

class third_base {
public:
  int *d;
  sycl::accessor<char, 1, sycl::access::mode::read> AccField;
};

class second_base {
public:
  int *e;
  second_base(int *E) : e(E) {}
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

struct derived : base, second_base, third_base{
  int a;
  derived() : second_base(nullptr) {}
  void operator()() const {
  }
};

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    derived f{};
    cgh.single_task(f);
  });

  return 0;
}

// Check declaration of the kernel
// CHECK: derived{{.*}} 'void (base, __generated_second_base, __wrapper_class,
// CHECK-SAME: __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int)

// Check parameters of the kernel
// CHECK: ParmVarDecl {{.*}} used _arg__base 'base'
// CHECK: ParmVarDecl {{.*}} used _arg__base '__generated_second_base'
// CHECK: ParmVarDecl {{.*}} used _arg_d '__wrapper_class'
// CHECK: ParmVarDecl {{.*}} used _arg_AccField '__global char *'
// CHECK: ParmVarDecl {{.*}} used _arg_AccField 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}} used _arg_AccField 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}} used _arg_AccField 'sycl::id<1>'
// CHECK: ParmVarDecl {{.*}} used _arg_a 'int'

// Check initializers for derived and base classes.
// Each class has it's own initializer list
// Base classes should be initialized first.
// CHECK: VarDecl {{.*}} __wrapper_union

// base is a simple class with no corresponding generated type. Therefore
// copy from ParamVar
// CHECK:      CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    ImplicitCastExpr {{.*}} <DerivedToBase (base)>
// CHECK-NEXT:     MemberExpr {{.*}} 'derived'
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} '_arg__base'
// CHECK-NEXT:  IntegerLiteral {{.*}} 12

// second_base contains pointers and therefore the ParamVar is a new generated
// type. Perform a copy of the corresponding kernel parameter via
// reinterpret_cast.
// CHECK-NEXT: CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    ImplicitCastExpr {{.*}} <DerivedToBase (second_base)>
// CHECK-NEXT:     MemberExpr {{.*}} 'derived'
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} '_arg__base' '__generated_second_base'
// CHECK-NEXT:  IntegerLiteral {{.*}} 8

// third_base contains special type accessor. Therefore it is decomposed and it's
// data members are copied from corresponding ParamVar
// CHECK-NEXT:  BinaryOperator {{.*}} '='
// CHECK-NEXT:   MemberExpr {{.*}} .d
// CHECK-NEXT:    ImplicitCastExpr {{.*}} <DerivedToBase (third_base)>
// CHECK-NEXT:     MemberExpr {{.*}} 'derived'
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:   ImplicitCastExpr {{.*}} <AddressSpaceConversion>
// CHECK-NEXT:    ImplicitCastExpr
// CHECK-NEXT:     MemberExpr {{.*}} lvalue .
// CHECK-NEXT:      DeclRefExpr {{.*}} '_arg_d' '__wrapper_class'
// CHECK-NEXT: CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .AccField
// CHECK-NEXT:     ImplicitCastExpr {{.*}} <DerivedToBase (third_base)>
// CHECK-NEXT:      MemberExpr {{.*}} 'derived'
// CHECK-NEXT:       DeclRefExpr {{.*}} '__wrapper_union'
// Check kernel body for call to __init function of accessor
// CHECK: CXXMemberCallExpr
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} lvalue .AccField
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'third_base':'third_base' lvalue <DerivedToBase (third_base)>
// CHECK-NEXT:  MemberExpr {{.*}} 'derived'
// CHECK-NEXT:   DeclRefExpr {{.*}} '__wrapper_union'

// Initialize fields of 'derived'
// CHECK:      BinaryOperator {{.*}} '='
// CHECK-NEXT:  MemberExpr {{.*}} .a
// CHECK-NEXT:   MemberExpr {{.*}} 'derived'
// CHECK-NEXT:    DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  DeclRefExpr {{.*}} '_arg_a'
