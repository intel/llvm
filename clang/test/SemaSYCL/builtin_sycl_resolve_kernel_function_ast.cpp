// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -ast-dump -ast-dump-filter launch_ %s | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -ast-dump -ast-dump-filter launch_ %s | FileCheck %s

// In a template, a call to __builtin_sycl_resolve_kernel_function whose operand
// depends on a template type parameter keeps the unresolved call and is
// replaced by a pointer to the selected kernel when instantiated. An operand
// that is only value-dependent has already selected its callee and is replaced
// in the template itself.

#define ND_KERNEL                                                              \
  [[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]   \
  __attribute__((sycl_device))

template <typename T> ND_KERNEL void axpy(T *y, T a) {}
ND_KERNEL void k2(int *p, int n) {}

template <typename T> void launch_dep(T *y, T a) {
  (void)__builtin_sycl_resolve_kernel_function(axpy(y, a));
}

template <int N> void launch_valdep(int *p) {
  (void)__builtin_sycl_resolve_kernel_function(k2(p, N));
}

void use(float *y, int *p) {
  launch_dep(y, 1.0f);
  launch_valdep<3>(p);
}

// CHECK-LABEL: Dumping launch_dep:
// CHECK:      FunctionDecl {{.*}} launch_dep 'void (T *, T)'
// CHECK:      CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: DeclRefExpr {{.*}} Function {{.*}} '__builtin_sycl_resolve_kernel_function'
// CHECK-NEXT: CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: UnresolvedLookupExpr {{.*}} (ADL) = 'axpy'
// CHECK-NEXT: DeclRefExpr {{.*}} ParmVar {{.*}} 'y' 'T *'
// CHECK-NEXT: DeclRefExpr {{.*}} ParmVar {{.*}} 'a' 'T'
// CHECK:      FunctionDecl {{.*}} launch_dep 'void (float *, float)' implicit_instantiation
// CHECK-NOT:  __builtin_sycl_resolve_kernel_function
// CHECK:      ImplicitCastExpr {{.*}} 'void (*)(float *, float)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} Function {{.*}} 'axpy' 'void (float *, float)'

// CHECK-LABEL: Dumping launch_valdep:
// CHECK:      FunctionDecl {{.*}} launch_valdep 'void (int *)'
// CHECK-NOT:  __builtin_sycl_resolve_kernel_function
// CHECK:      ImplicitCastExpr {{.*}} 'void (*)(int *, int)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} Function {{.*}} 'k2' 'void (int *, int)'
// CHECK:      FunctionDecl {{.*}} launch_valdep 'void (int *)' implicit_instantiation
// CHECK-NOT:  __builtin_sycl_resolve_kernel_function
// CHECK:      DeclRefExpr {{.*}} Function {{.*}} 'k2' 'void (int *, int)'
