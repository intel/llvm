// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffp-accuracy-attr=high \
// RUN: -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: "-ffp-accuracy-attr=high:[sin,cosf] low:[tan]" \
// RUN: -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=CHECK-FUNC %s

float a, b, c, d, e;

void foo(void) {
  // CHECK-LABEL: define {{.*}}void @foo()
  a = cosf(b);
  c = sin(a);
  d = tan(c);
  e = cos(d);
  // CHECK: call float @llvm.cos.f32(float {{.*}})   [[ATTR3:#[0-9]+]]
  // CHECK: call double @llvm.sin.f64(double {{.*}}) [[ATTR3:#[0-9]+]]
  // CHECK: call double @tan(double noundef {{.*}})  [[ATTR3:#[0-9]+]]
  // CHECK: call double @llvm.cos.f64(double {{.*}}) [[ATTR3:#[0-9]+]]

  // CHECK-FUNC: call float @llvm.cos.f32(float {{.*}})   [[ATTR3:#[0-9]+]]
  // CHECK-FUNC: call double @llvm.sin.f64(double {{.*}}) [[ATTR3:#[0-9]+]]
  // CHECK-FUNC: call double @tan(double noundef {{.*}})  [[ATTR4:#[0-9]+]]
  // CHECK-FUNC: call double @llvm.cos.f64(double {{.*}})
}

// CHECK-FUNC: attributes [[ATTR3]] = {{{.*}}"fpaccuracy="="high"
// CHECK-FUNC: attributes [[ATTR4]] = {{{.*}} "fpaccuracy="="low"
