// RUN: %clang_cc1 -sycl-std=2020 -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel arguments for
// VLAs.

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

class Canonical;
class Canon2;
class FLA;
class SD;
class MD3;
class MDVLAFLA;
class MDFLAVLA;

void foo(int *i, int x, int y) {
// CHECK: FunctionDecl {{.*}} foo 'void (int *, int, int)'
// CHECK-NEXT: ParmVarDecl {{.*}} used i 'int *'
// CHECK-NEXT: ParmVarDecl [[PARM_X:0x[0-9a-f]+]] {{.*}} used x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used y 'int'
  sycl::queue q;

  using intarray = int(*)[x]; // Canonical example for VLA usage
  intarray ia = reinterpret_cast<intarray>(i);

  int fla[4]; // Not a VLA

  using intmdarray = int(*)[x][y]; // Multi-dimensional VLA
  intmdarray imda = reinterpret_cast<intmdarray>(i);

  using intflavlaarray = int(*)[x][4]; // VLA then a fixed-length array
  intflavlaarray ifva = reinterpret_cast<intflavlaarray>(i);

  using intvlaflaarray = int(*)[4][y]; // A fix-length array dimension followed by a VLA
  intvlaflaarray ivfa = reinterpret_cast<intvlaflaarray>(i);

  ++x;
// CHECK: UnaryOperator {{.*}} 'int' lvalue prefix '++'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar [[PARM_X]] 'x' 'int'
  using anotherintarray = int(*)[x]; // Canonical example for VLA usage
  anotherintarray aia = reinterpret_cast<anotherintarray>(i);
// CHECK: DeclStmt
// CHECK-NEXT: TypeAliasDecl {{.*}} referenced anotherintarray 'int (*)[x]'
// CHECK-NEXT: PointerType {{.*}} 'int (*)[x]' variably_modified
// CHECK-NEXT: ParenType {{.*}} 'int [x]' sugar variably_modified
// CHECK-NEXT: VariableArrayType {{.*}} 'int [x]' variably_modified {{.*}}
// CHECK-NEXT: BuiltinType {{.*}} 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar [[PARM_X]] 'x' 'int'

  q.submit([&](cl::sycl::handler &h) {
    h.single_task<Canonical>(
      [=]() {
        ia[1][2] = 9;
      });
// Check Canonical parameters
// CHECK: FunctionDecl {{.*}}Canonical{{.*}} 'void (unsigned long, int *__global *)'
// CHECK: ParmVarDecl {{.*}} used _arg_ 'unsigned long'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'int *__global *'

// Check Canonical inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned long' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned long' lvalue ParmVar {{.*}} '_arg_' 'unsigned long'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'intarray':'int (*)[x]' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *__global *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int *__global *' lvalue ParmVar {{.*}} '_arg_' 'int *__global *'

    h.single_task<Canon2>(
      [=]() {
        aia[3][4] = 7;
      });

    h.single_task<FLA>(
      [=]() {
        int i;
        i = fla[0];
      });
// Check FLA parameters
// CHECK: FunctionDecl {{.*}}FLA{{.*}} 'void (__wrapper_class)'
// CHECK: ParmVarDecl {{.*}} used _arg_ '__wrapper_class'

// Check FLA inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ArrayInitLoopExpr {{.*}} 'int [4]'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int [4]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'int [4]' lvalue . {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int [4]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'int [4]' lvalue . {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_' '__wrapper_class'
// CHECK-NEXT: ArrayInitIndexExpr {{.*}} 'unsigned long'

    h.single_task<SD>(
      [=]() {
        i[0] = 9; // single-level pointer
      });
// CHECK SD parameters
// CHECK: FunctionDecl {{.*}}SD{{.*}} 'void (__global int *)'
// CHECK: ParmVarDecl {{.*}}id sloc> used _arg_ '__global int *'

// Check SD inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_' '__global int *'

    h.single_task<MD3>(
      [=]() {
        imda[1][2][3] = 9; // Multi-dimensional VLA
      });
// CHECK MD3 parameters
// CHECK: FunctionDecl {{.*}}MD3{{.*}} 'void (unsigned long, unsigned long, int **__global *)'
// CHECK: ParmVarDecl {{.*}} used _arg_ 'unsigned long'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'unsigned long'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'int **__global *'

// Check MD3 inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned long' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned long' lvalue ParmVar {{.*}} '_arg_' 'unsigned long'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned long' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned long' lvalue ParmVar {{.*}} '_arg_' 'unsigned long'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'intmdarray':'int (*)[x][y]' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int **__global *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int **__global *' lvalue ParmVar {{.*}} '_arg_' 'int **__global *'

    h.single_task<class MDVLAFLA>(
      [=]() {
        ifva[1][2][3] = 9;
      });
// CHECK MDVLAFLA parameters
// CHECK: FunctionDecl {{.*}}MDVLAFLA{{.*}} 'void (unsigned long, int (*__global *)[4])'
// CHECK: ParmVarDecl {{.*}} used _arg_ 'unsigned long'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'int (*__global *)[4]'

// Check MDVLAFLA inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned long' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned long' lvalue ParmVar {{.*}} '_arg_' 'unsigned long'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'intflavlaarray':'int (*)[x][4]' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*__global *)[4]' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int (*__global *)[4]' lvalue ParmVar {{.*}} '_arg_' 'int (*__global *)[4]'

    h.single_task<class MDFLAVLA>(
      [=]() {
        ivfa[1][2][3] = 9;
      });
// Check MDFLAVLA parameters
// CHECK: FunctionDecl {{.*}}MDFLAVLA{{.*}} 'void (unsigned long, unsigned long, int **__global *)'
// CHECK: ParmVarDecl {{.*}} used _arg_ 'unsigned long'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'unsigned long'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'int **__global *'

// Check MDFLAVLA inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned long' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned long' lvalue ParmVar {{.*}} '_arg_' 'unsigned long'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned long' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned long' lvalue ParmVar {{.*}} '_arg_' 'unsigned long'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'intvlaflaarray':'int (*)[4][y]' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int **__global *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int **__global *' lvalue ParmVar {{.*}} '_arg_' 'int **__global *'
  });
}
