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
// RUN: -D SPIR -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=CHECK-SPIR %s

#ifdef SPIR
void sincos(float, float *, float *);
#endif

void foo(float f1, float f2) {
  // CHECK-LABEL: define {{.*}}void @foo(float noundef {{.*}}, float noundef {{.*}})
  float sin = 0.f, cos = 0.f;

  f2 = cosf(f1);
  f2 = sinf(f1);
  f2 = tan(f1);
  f2 = tanf(f1);
  sincos(f1, &sin, &cos);

  // CHECK: call float @llvm.fpbuiltin.cos.f32(float {{.*}})    [[ATTR_HIGH:#[0-9]+]]
  // CHECK: call float @llvm.fpbuiltin.sin.f32(float {{.*}})    [[ATTR_HIGH]]
  // CHECK: call double @llvm.fpbuiltin.tan.f64(double {{.*}})  [[ATTR_HIGH]]
  // CHECK: call float @llvm.fpbuiltin.tan.f32(float {{.*}})    [[ATTR_HIGH]]
  // CHECK: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) [[ATTR_HIGH]]

  // CHECK-FUNC-1: call float @llvm.fpbuiltin.cos.f32(float {{.*}})   [[ATTR_HIGH:#[0-9]+]]
  // CHECK-FUNC-1: call float @llvm.sin.f32(float {{.*}})
  // CHECK-FUNC-1: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) [[ATTR_LOW:#[0-9]+]]
  // CHECK-FUNC-1: call float @llvm.fpbuiltin.tan.f32(float {{.*}})
  // CHECK-FUNC-1: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) [[ATTR_MEDIUM:#[0-9]+]]

  // CHECK-FUNC-2: call float @llvm.fpbuiltin.cos.f32(float {{.*}})      [[ATTR_MEDIUM:#[0-9]+]]
  // CHECK-FUNC-2: call float @llvm.fpbuiltin.sin.f32(float {{.*}})      [[ATTR_MEDIUM]]
  // CHECK-FUNC-2: call double @llvm.fpbuiltin.tan.f64(double {{.*}})    [[ATTR_HIGH:#[0-9]+]]
  // CHECK-FUNC-2: call float @llvm.fpbuiltin.tan.f32(float {{.*}})      [[ATTR_MEDIUM]]
  // CHECK-FUNC-2: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) [[ATTR_MEDIUM]]

  // CHECK-SPIR: call float @llvm.fpbuiltin.cos.f32(float {{.*}})      [[ATTR_SYCL1:#[0-9]+]]
  // CHECK-SPIR: call float @llvm.fpbuiltin.sin.f32(float {{.*}})      [[ATTR_SYCL1]]
  // CHECK-SPIR: call double @llvm.fpbuiltin.tan.f64(double {{.*}})    [[ATTR_SYCL2:#[0-9]+]]
  // CHECK-SPIR: call float @llvm.fpbuiltin.tan.f32(float {{.*}})      [[ATTR_SYCL2]]
  // CHECK-SPIR: call void @llvm.fpbuiltin.sincos.f32(float {{.*}}, ptr {{.*}}, ptr {{.*}}) [[ATTR_SYCL1]]

}

// CHECK: attributes [[ATTR_HIGH]] = {{{.*}}"fpbuiltin-max-error="="1.0f"

// CHECK-FUNC-1: attributes [[ATTR_HIGH]] = {{{.*}}"fpbuiltin-max-error="="1.0f"
// CHECK-FUNC-1: attributes [[ATTR_LOW]] = {{{.*}}"fpbuiltin-max-error="="67108864.0f"
// CHECK-FUNC-1: attributes [[ATTR_MEDIUM]] = {{{.*}}"fpbuiltin-max-error="="4.0f"

// CHECK-FUNC-2: attributes #5 = { "fpbuiltin-max-error="="4.0f" }
// CHECK-FUNC-2: attributes #6 = { "fpbuiltin-max-error="="1.0f" }

// CHECK-SPIR: attributes [[ATTR_SYCL1]] = {{{.*}}"fpbuiltin-max-error="="4.0f"
// CHECK-SPIR: attributes [[ATTR_SYCL2]] = {{{.*}}"fpbuiltin-max-error="="5.0f"
