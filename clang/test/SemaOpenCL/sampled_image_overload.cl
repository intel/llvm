// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

void __attribute__((overloadable)) foo(__ocl_sampled_image1d_ro_t);
void __attribute__((overloadable)) foo(__ocl_sampled_image2d_ro_t);

// CHECK: FunctionDecl {{.*}} <{{.*}}> line:{{.*}} ker 'void (__private __ocl_sampled_image1d_ro_t, __private __ocl_sampled_image2d_ro_t)'
void kernel ker(__ocl_sampled_image1d_ro_t src1, __ocl_sampled_image2d_ro_t src2) {
  // CHECK: CallExpr
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <{{.*}}> 'void (*)(__private __ocl_sampled_image1d_ro_t)'
  foo(src1);
  // CHECK: CallExpr
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <{{.*}}> 'void (*)(__private __ocl_sampled_image2d_ro_t)'
  foo(src2);
}
