// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffp-accuracy-attr=high \
// RUN: -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: "-ffp-accuracy-attr=high:[sin,cosf] low:[tan] medium:[sincos]" \
// RUN: -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=CHECK-FUNC %s

// RUN: %clang_cc1 -triple spir64-unknown-unknown \
// RUN: -D SPIR -ffp-accuracy-attr=sycl \
// RUN: -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=CHECK-SPIR %s

float a, b, c, d, e, f;

void foo(void) {
  // CHECK-LABEL: define {{.*}}void @foo()
  a = cosf(b);
  c = sin(a);
  d = tan(c);
  e = cos(d);
#ifndef SPIR
  d = sincos(e);
#endif
  // CHECK: call float @llvm.fpbuiltin.cos.f32(float {{.*}})        [[ATTR3:#[0-9]+]]
  // CHECK: call double @llvm.fpbuiltin.sin.f64(double {{.*}})      [[ATTR3:#[0-9]+]]
  // CHECK: call double @llvm.fpbuiltin.tan.f64(double {{.*}})      [[ATTR3:#[0-9]+]]
  // CHECK: call double @llvm.fpbuiltin.cos.f64(double {{.*}})      [[ATTR3:#[0-9]+]]

  // CHECK-FUNC: call float @llvm.fpbuiltin.cos.f32(float {{.*}})   [[ATTR3:#[0-9]+]]
  // CHECK-FUNC: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) [[ATTR3:#[0-9]+]]
  // CHECK-FUNC: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) [[ATTR4:#[0-9]+]]
  // CHECK-FUNC: call double @llvm.fpbuiltin.cos.f64(double {{.*}})
  // CHECK-FUNC: call i32 (double, ...) @sincos(double {{.*}})      [[ATTR5:#[0-9]+]]

  // CHECK-SPIR: call float @llvm.fpbuiltin.cos.f32(float {{.*}}    [[ATTR5:#[0-9]+]]
  // CHECK-SPIR: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) [[ATTR5:#[0-9]+]]
  // CHECK-SPIR: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) [[ATTR6:#[0-9]+]]
  // CHECK-SPIR: call double @llvm.fpbuiltin.cos.f64(double {{.*}}  [[ATTR5:#[0-9]+]]
}

// CHECK-FUNC: attributes [[ATTR3]] = {{{.*}}"fpbuiltin-max-error="="1.0f"
// CHECK-FUNC: attributes [[ATTR4]] = {{{.*}}"fpbuiltin-max-error="="67108864.0f"
// CHECK-FUNC: attributes [[ATTR5]] = {{{.*}}"fpbuiltin-max-error="="4.0f"
// CHECK-SPIR: attributes [[ATTR6]] = {{{.*}}"fpbuiltin-max-error="="5.0f"
