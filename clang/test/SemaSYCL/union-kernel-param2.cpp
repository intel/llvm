// RUN: %clang_cc1 -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel arguments for
// a struct-with-an-array-of-unions and a array-of-struct-with-a-union.

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {

  union MyUnion {
    struct MyStruct {
      int a[3];
      float b;
      char c;
    } struct_mem;
    int d;
  } union_mem;

  struct MyStruct {
    union MyUnion {
      int a[3];
      float b;
      char c;
    } union_mem;
    int *d;
  } struct_mem;

  a_kernel<class kernel_A>(
      [=]() {
        int local = union_mem.struct_mem.a[2];
      });

  a_kernel<class kernel_B>(
      [=]() {
        int local = struct_mem.union_mem.a[2];
      });
}

// Check kernel_A parameters
// CHECK: FunctionDecl {{.*}}kernel_A{{.*}} 'void (union MyUnion)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_union_mem 'union MyUnion':'MyUnion'

// Check kernel_A inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: CXXConstructExpr {{.*}} 'union MyUnion':'MyUnion' 'void (const MyUnion &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const MyUnion'
// CHECK-NEXT: DeclRefExpr {{.*}} 'union MyUnion':'MyUnion' lvalue ParmVar {{.*}} '_arg_union_mem' 'union MyUnion':'MyUnion'

// Check kernel_B parameters
// CHECK: FunctionDecl {{.*}}kernel_B{{.*}} 'void (union MyUnion, __wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_union_mem 'union MyUnion':'MyStruct::MyUnion'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_d '__wrapper_class'

// Check kernel_B inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: InitListExpr {{.*}} 'MyStruct'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'union MyUnion':'MyStruct::MyUnion' 'void (const MyStruct::MyUnion &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const MyStruct::MyUnion'
// CHECK-NEXT: DeclRefExpr {{.*}} 'union MyUnion':'MyStruct::MyUnion' lvalue ParmVar {{.*}} '_arg_union_mem' 'union MyUnion':'MyStruct::MyUnion'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: MemberExpr
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_d' '__wrapper_class'
