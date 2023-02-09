// The next two lines are commented until the call isFPAccuracyAvailable is
// complete.
// TODO: Add next two lines to testing
// RUN %clang_cc1 -Wno-implicit-function-declaration -emit-llvm -o - %s
// RUN %clang_cc1 -ffp-accuracy=default -Wno-implicit-function-declaration -emit-llvm -o - %s

// TODO: Add cuda and sycl lines for testing

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffp-accuracy=high -Wno-implicit-function-declaration -emit-llvm -o - %s | FileCheck %s -check-prefix=FPAHIGH
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffp-accuracy=medium -Wno-implicit-function-declaration -emit-llvm -o - %s | FileCheck %s -check-prefix=FPAMED
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffp-accuracy=low -Wno-implicit-function-declaration -emit-llvm -o - %s | FileCheck %s -check-prefix=FPALOW

float a, b;

void foo(void) {
 // CHECK-LABEL: define {{.*}}void @foo()
  a = cosf(b);
  // FPAHIGH: call float @llvm.experimental.fpaccuracy.cos.f32(float %{{.*}}, metadata !"fpaccuracy.high")
  // FPAMED: call float @llvm.experimental.fpaccuracy.cos.f32(float %{{.*}}, metadata !"fpaccuracy.medium")
  // FPALOW: call float @llvm.experimental.fpaccuracy.cos.f32(float %{{.*}}, metadata !"fpaccuracy.low")
}
