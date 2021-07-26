// RUN: %clang_cc1 -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel arguments for
// VLAs.

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(const Func &kernelFunc) {
  kernelFunc();
}

void foo(int *i, int x, int y) {
  using intarray = int(*)[x]; // Motivating example
  intarray ia = reinterpret_cast<intarray>(i);
  a_kernel<class Canonical>([=]() {
    ia[1][2] = 9;
  });
  a_kernel<class FLA>([=]() { // Not a VLA
    int fla[4];
    fla[0] = 9;
  });
  a_kernel<class SD>([=]() {
    i[0] = 9; // single-level pointer
  });
  using intmdarray = int(*)[x][y];
  intmdarray imda = reinterpret_cast<intmdarray>(i);
  a_kernel<class MD3>([=]() {
    imda[1][2][3] = 9; // Multi-dimensional VLA
  });
  using intflavlaarray = int(*)[x][4]; // VLA then a fixed-length array
  intflavlaarray ifva = reinterpret_cast<intflavlaarray>(i);
  a_kernel<class MDFLA>([=]() {
    ifva[1][2][3] = 9;
  });
  using intvlaflaarray = int(*)[4][y]; // A fix-length array dimension followed by a VLA
  intvlaflaarray ivfa = reinterpret_cast<intvlaflaarray>(i);
  a_kernel<class MDFLAVLA>([=]() {
    ivfa[1][2][3] = 9;
  });
}

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

// Check FLA parameters
// CHECK: FunctionDecl {{.*}}FLA{{.*}} 'void ()'

// Check FLA inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr

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

// CHECK MDFLA parameters
// CHECK: FunctionDecl {{.*}}MDFLA{{.*}} 'void (unsigned long, int (*__global *)[4])'
// CHECK: ParmVarDecl {{.*}} used _arg_ 'unsigned long'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'int (*__global *)[4]'

// Check MDFLA inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned long' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned long' lvalue ParmVar {{.*}} '_arg_' 'unsigned long'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'intflavlaarray':'int (*)[x][4]' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*__global *)[4]' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int (*__global *)[4]' lvalue ParmVar {{.*}} '_arg_' 'int (*__global *)[4]'

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
