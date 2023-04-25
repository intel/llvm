// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffp-builtin-accuracy=high \
// RUN: -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: "-ffp-builtin-accuracy=high:[sin,cosf] low:[tan] medium:[sincos]" \
// RUN: -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=CHECK-FUNC-1 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: "-ffp-builtin-accuracy=medium high:[tan] cuda:[cos]" \
// RUN: -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=CHECK-FUNC-2 %s

// RUN: %clang_cc1 -triple spir64-unknown-unknown -ffp-builtin-accuracy=sycl \
// RUN: -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=CHECK-SPIR %s

float a, b, c, d, f;
double e;

extern double sincos (double __x) __attribute__ ((__nothrow__ ));

void foo(void) {
  // CHECK-LABEL: define {{.*}}void @foo()
  a = cosf(b);
  c = sin(a);
  d = tan(c);
  e = cos(d);
  d = sincos(e);

  // CHECK: call float @llvm.fpbuiltin.cos.f32(float {{.*}})        [[ATTR_HIGH:#[0-9]+]]
  // CHECK: call double @llvm.fpbuiltin.sin.f64(double {{.*}})      [[ATTR_HIGH]]
  // CHECK: call double @llvm.fpbuiltin.tan.f64(double {{.*}})      [[ATTR_HIGH]]
  // CHECK: call double @llvm.fpbuiltin.cos.f64(double {{.*}})      [[ATTR_HIGH]]

  // CHECK-FUNC-1: call float @llvm.fpbuiltin.cos.f32(float {{.*}})   [[ATTR_HIGH:#[0-9]+]]
  // CHECK-FUNC-1: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) [[ATTR_HIGH]]
  // CHECK-FUNC-1: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) [[ATTR_LOW:#[0-9]+]]
  // CHECK-FUNC-1: call double @llvm.cos.f64(double {{.*}})
  // CHECK-FUNC-1: call double @sincos(double {{.*}})                 [[ATTR_MEDIUM:#[0-9]+]]

  // CHECK-FUNC-2: call float @llvm.fpbuiltin.cos.f32(float {{.*}})   [[ATTR_MEDIUM:#[0-9]+]]
  // CHECK-FUNC-2: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) [[ATTR_MEDIUM]]
  // CHECK-FUNC-2: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) [[ATTR_HIGH:#[0-9]+]]
  // CHECK-FUNC-2: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) [[ATTR_CUDA:#[0-9]+]]
  // CHECK-FUNC-2: call double @sincos(double {{.*}})                 [[ATTR_MEDIUM]]

  // CHECK-SPIR: call float @llvm.fpbuiltin.cos.f32(float %0)       [[ATTR_MEDIUM:#[0-9]+]]
  // CHECK-SPIR: call double @llvm.fpbuiltin.sin.f64(double %conv)  [[ATTR_MEDIUM]]
  // CHECK-SPIR: call double @llvm.fpbuiltin.tan.f64(double %conv2) [[ATTR_SYCL:#[0-9]+]]
  // CHECK-SPIR: call double @llvm.fpbuiltin.cos.f64(double %conv4) [[ATTR_MEDIUM]]
  // CHECK-SPIR: call spir_func double @sincos(double %8)           [[ATTR_MEDIUM]]
}

// CHECK: attributes [[ATTR_HIGH]] = {{{.*}}"fpbuiltin-max-error="="1.0f"

// CHECK-FUNC-1: attributes [[ATTR_HIGH]] = {{{.*}}"fpbuiltin-max-error="="1.0f"
// CHECK-FUNC-1: attributes [[ATTR_LOW]] = {{{.*}}"fpbuiltin-max-error="="67108864.0f"
// CHECK-FUNC-1: attributes [[ATTR_MEDIUM]] = {{{.*}}"fpbuiltin-max-error="="4.0f"

// CHECK-FUNC-2: attributes [[ATTR_MEDIUM]] = {{{.*}}"fpbuiltin-max-error="="4.0f"
// CHECK-FUNC-2: attributes [[ATTR_HIGH]] = {{{.*}}"fpbuiltin-max-error="="1.0f"
// CHECK-FUNC-2: attributes [[ATTR_CUDA]] = {{{.*}}"fpbuiltin-max-error="="2.0f"

// CHECK-SPIR: attributes [[ATTR_MEDIUM]] = {{{.*}}"fpbuiltin-max-error="="4.0f"
// CHECK-SPIR: attributes [[ATTR_SYCL]] = {{{.*}}"fpbuiltin-max-error="="5.0f"
