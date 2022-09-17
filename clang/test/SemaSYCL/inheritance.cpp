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
// CHECK: derived{{.*}} 'void (base, _generated_second_base, __wrapper_class,
// CHECK-SAME: __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int)

// Check parameters of the kernel
// CHECK: ParmVarDecl {{.*}} used _arg__base 'base'
// CHECK: ParmVarDecl {{.*}} used _arg__base '_generated_second_base'
// CHECK: ParmVarDecl {{.*}} used _arg_d '__wrapper_class'
// CHECK: ParmVarDecl {{.*}} used _arg_AccField '__global char *'
// CHECK: ParmVarDecl {{.*}} used _arg_AccField 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}} used _arg_AccField 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}} used _arg_AccField 'sycl::id<1>'
// CHECK: ParmVarDecl {{.*}} used _arg_a 'int'

// Check initializers for derived and base classes.
// Each class has it's own initializer list
// Base classes should be initialized first.
// CHECK: VarDecl {{.*}} used derived 'derived' cinit
// CHECK-NEXT: InitListExpr {{.*}} 'derived'

// base is a simple class with no corresponding generated type. Therefore
// copy from ParamVar
// CHECK-NEXT: CXXConstructExpr {{.*}} 'base':'base' 'void (const base &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const base' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue ParmVar {{.*}} '_arg__base' 'base'

// second_base contains pointers and therefore the ParamVar is a new generated
// type. Default construct this class and initialize second_base via memcpy in
// body statements.
// CHECK-NEXT: CXXConstructExpr {{.*}} 'second_base':'second_base' 'void () noexcept'

// third_base contains special type accessor. Therefore it is decomposed and it's
// data members are copied from corresponding ParamVar
// CHECK-NEXT: InitListExpr {{.*}} 'third_base'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue ParmVar {{.*}} '_arg_d' '__wrapper_class'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::accessor<char, 1, sycl::access::mode::read>'

// Initialize fields of 'derived'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue ParmVar {{.*}} '_arg_a' 'int'

// Check kernel body for call to __builtin_memcpy to initialize second_base 
// CHECK: CallExpr {{.*}} 'void *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void *(*)(void *, const void *, unsigned long) noexcept' <BuiltinFnToFnPtr>
// CHECK-NEXT: DeclRefExpr {{.*}} Function {{.*}} '__builtin_memcpy' 'void *(void *, const void *, unsigned long) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void *' <BitCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'second_base *' <LValueToRValue>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'second_base *' lvalue <DerivedToBase (second_base)>
// CHECK-NEXT: UnaryOperator {{.*}} 'derived *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr {{.*}} 'derived' lvalue Var {{.*}} 'derived' 'derived'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const void *' <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '_generated_second_base *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr {{.*}} '_generated_second_base' lvalue ParmVar {{.*}} '_arg__base' '_generated_second_base'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 8

// Check kernel body for call to __init function of accessor
// CHECK: CXXMemberCallExpr
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} lvalue .AccField
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'third_base':'third_base' lvalue <DerivedToBase (third_base)>
// CHECK-NEXT: DeclRefExpr {{.*}} 'derived' lvalue Var {{.*}} 'derived' 'derived'
