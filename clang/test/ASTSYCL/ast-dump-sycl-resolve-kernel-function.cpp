// Tests without serialization:
// RUN: %clang_cc1 -std=c++17 -triple spirv64-unknown-unknown -fsycl-is-device \
// RUN:   -ast-dump %s \
// RUN:   | FileCheck --match-full-lines %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-host \
// RUN:   -ast-dump %s \
// RUN:   | FileCheck --match-full-lines %s
//
// Tests with serialization (host only: deserializing the kernel declarations
// that the device compilation generates for free function kernels currently
// asserts, independently of this builtin):
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-host \
// RUN:   -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-host \
// RUN:   -include-pch %t -ast-dump-all /dev/null \
// RUN:   | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN:   | FileCheck --match-full-lines %s

// These tests validate the AST produced for calls to
// __builtin_sycl_resolve_kernel_function. The builtin call is replaced by a
// pointer to the selected free function kernel, except in a template where the
// operand depends on a template type parameter; there the call to the builtin
// and the unresolved call it operates on are kept and resolved when the
// template is instantiated.

// CHECK: TranslationUnitDecl {{.*}}

#define ND_KERNEL                                                              \
  [[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]   \
  __attribute__((sycl_device))

template <typename T> ND_KERNEL void axpy(T *y, T a) {}
ND_KERNEL void k2(int *p, int n) {}
// CHECK:      |-FunctionDecl [[K2:0x[0-9a-f]+]] {{.*}} used k2 'void (int *, int)' external-linkage

// The kernel selected for each launch below is an implicit instantiation, an
// explicit specialization, and an explicit instantiation of ffk.
template <typename T> ND_KERNEL void ffk(int *p, T v) {}
template <> ND_KERNEL void ffk(int *p, int v) {}
template void ffk(int *p, long);
// CHECK:      |-FunctionTemplateDecl {{.*}} ffk external-linkage
// CHECK:      | |-FunctionDecl [[FFK_LONG:0x[0-9a-f]+]] {{.*}} used ffk 'void (int *, long)' explicit_instantiation_definition instantiated_from {{.*}}
// CHECK:      | `-FunctionDecl [[FFK_CHAR:0x[0-9a-f]+]] {{.*}} used ffk 'void (int *, char)' implicit_instantiation instantiated_from {{.*}}
// CHECK:      |-FunctionDecl [[FFK_INT:0x[0-9a-f]+]] prev {{.*}} used ffk 'void (int *, int)' explicit_specialization external-linkage

// The operand depends on a template type parameter.
template <typename T> void launch_dep(T *y, T a) {
  (void)__builtin_sycl_resolve_kernel_function(axpy(y, a));
}
// CHECK:      |-FunctionTemplateDecl {{.*}} launch_dep external-linkage
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl {{.*}} launch_dep 'void (T *, T)'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} referenced y 'T *'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} referenced a 'T'
// CHECK-NEXT: | | `-CompoundStmt {{.*}}
// CHECK-NEXT: | |   `-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: | |     `-CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: | |       |-DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_sycl_resolve_kernel_function' 'void *(...) noexcept'
// CHECK-NEXT: | |       `-CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: | |         |-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'axpy' {{.*}}
// CHECK-NEXT: | |         |-DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'y' 'T *'
// CHECK-NEXT: | |         `-DeclRefExpr {{.*}} 'T' lvalue ParmVar {{.*}} 'a' 'T'
// CHECK-NEXT: | `-FunctionDecl {{.*}} used launch_dep 'void (float *, float)' implicit_instantiation instantiated_from {{.*}}
// CHECK-NEXT: |   |-TemplateArgument type 'float'
// CHECK-NEXT: |   | `-BuiltinType {{.*}} 'float'
// CHECK-NEXT: |   |-ParmVarDecl {{.*}} used y 'float *'
// CHECK-NEXT: |   |-ParmVarDecl {{.*}} used a 'float'
// CHECK-NEXT: |   `-CompoundStmt {{.*}}
// CHECK-NEXT: |     `-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: |       `-ImplicitCastExpr {{.*}} 'void (*)(float *, float)' <FunctionToPointerDecay> part_of_explicit_cast
// CHECK-NEXT: |         `-DeclRefExpr {{.*}} 'void (float *, float)' lvalue Function {{.*}} 'axpy' 'void (float *, float)'

// The operand depends only on the value of a template parameter; the kernel is
// already selected in the template.
template <int N> void launch_valdep(int *p) {
  (void)__builtin_sycl_resolve_kernel_function(k2(p, N));
}
// CHECK:      |-FunctionTemplateDecl {{.*}} launch_valdep external-linkage
// CHECK-NEXT: | |-NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 N
// CHECK-NEXT: | |-FunctionDecl {{.*}} launch_valdep 'void (int *)'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} referenced p 'int *'
// CHECK-NEXT: | | `-CompoundStmt {{.*}}
// CHECK-NEXT: | |   `-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: | |     `-ImplicitCastExpr {{.*}} 'void (*)(int *, int)' <FunctionToPointerDecay> part_of_explicit_cast
// CHECK-NEXT: | |       `-DeclRefExpr {{.*}} 'void (int *, int)' lvalue Function [[K2]] 'k2' 'void (int *, int)'
// CHECK-NEXT: | `-FunctionDecl {{.*}} used launch_valdep 'void (int *)' implicit_instantiation instantiated_from {{.*}}
// CHECK-NEXT: |   |-TemplateArgument integral '3'
// CHECK-NEXT: |   |-ParmVarDecl {{.*}} p 'int *'
// CHECK-NEXT: |   `-CompoundStmt {{.*}}
// CHECK-NEXT: |     `-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} 'void (int *, int)' lvalue Function [[K2]] 'k2' 'void (int *, int)'

// The operand does not depend on a template parameter; no two-phase lookup is
// needed and both the template and its instantiation refer to the kernel.
template <int> void launch_nondep(int *p) {
  (void)__builtin_sycl_resolve_kernel_function(k2(p, 0));
}
// CHECK:      |-FunctionTemplateDecl {{.*}} launch_nondep external-linkage
// CHECK-NEXT: | |-NonTypeTemplateParmDecl {{.*}} 'int' depth 0 index 0
// CHECK-NEXT: | |-FunctionDecl {{.*}} launch_nondep 'void (int *)'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} referenced p 'int *'
// CHECK-NEXT: | | `-CompoundStmt {{.*}}
// CHECK-NEXT: | |   `-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: | |     `-ImplicitCastExpr {{.*}} 'void (*)(int *, int)' <FunctionToPointerDecay> part_of_explicit_cast
// CHECK-NEXT: | |       `-DeclRefExpr {{.*}} 'void (int *, int)' lvalue Function [[K2]] 'k2' 'void (int *, int)'
// CHECK-NEXT: | `-FunctionDecl {{.*}} used launch_nondep 'void (int *)' implicit_instantiation instantiated_from {{.*}}
// CHECK-NEXT: |   |-TemplateArgument integral '7'
// CHECK-NEXT: |   |-ParmVarDecl {{.*}} p 'int *'
// CHECK-NEXT: |   `-CompoundStmt {{.*}}
// CHECK-NEXT: |     `-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} 'void (int *, int)' lvalue Function [[K2]] 'k2' 'void (int *, int)'

void launch_specializations(int *p) {
  (void)__builtin_sycl_resolve_kernel_function(ffk(p, '1'));
  (void)__builtin_sycl_resolve_kernel_function(ffk(p, 2));
  (void)__builtin_sycl_resolve_kernel_function(ffk(p, 3L));
}
// CHECK:      |-FunctionDecl {{.*}} launch_specializations 'void (int *)' external-linkage
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used p 'int *'
// CHECK-NEXT: | `-CompoundStmt {{.*}}
// CHECK-NEXT: |   |-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: |   | `-ImplicitCastExpr {{.*}} 'void (*)(int *, char)' <FunctionToPointerDecay> part_of_explicit_cast
// CHECK-NEXT: |   |   `-DeclRefExpr {{.*}} 'void (int *, char)' lvalue Function [[FFK_CHAR]] 'ffk' 'void (int *, char)'
// CHECK-NEXT: |   |-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: |   | `-ImplicitCastExpr {{.*}} 'void (*)(int *, int)' <FunctionToPointerDecay> part_of_explicit_cast
// CHECK-NEXT: |   |   `-DeclRefExpr {{.*}} 'void (int *, int)' lvalue Function [[FFK_INT]] 'ffk' 'void (int *, int)'
// CHECK-NEXT: |   `-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: |     `-ImplicitCastExpr {{.*}} 'void (*)(int *, long)' <FunctionToPointerDecay> part_of_explicit_cast
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} 'void (int *, long)' lvalue Function [[FFK_LONG]] 'ffk' 'void (int *, long)'

void use(float *y, int *p) {
  launch_dep(y, 1.0f);
  launch_valdep<3>(p);
  launch_nondep<7>(p);
}
