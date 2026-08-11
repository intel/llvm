// RUN: %clang_cc1 -std=c++2c -ast-dump %s | FileCheck %s

int f(int);
int f(char);
auto p = declcall(f(0));

// CHECK: CXXDeclcallExpr {{.*}} 'int (*)(int)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int (int)' {{.*}} Function {{.*}} 'f' 'int (int)'
