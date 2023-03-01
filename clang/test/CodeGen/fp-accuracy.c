// The next two lines are commented until the call isFPAccuracyAvailable is
// complete.
// TODO: Add next two lines to testing
// RUN %clang_cc1 -Wno-implicit-function-declaration -emit-llvm -o - %s
// RUN %clang_cc1 -ffp-accuracy=default -Wno-implicit-function-declaration -emit-llvm -o - %s

// TODO: Add cuda and sycl lines for testing

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffp-accuracy=high \
// RUN: -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefixes=CHECK,FPAHIGH

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffp-accuracy=medium \
// RUN: -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefixes=CHECK,FPAMED

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffp-accuracy=low \
// RUN: -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefixes=CHECK,FPALOW

float a, b;

void foo(void) {
 // CHECK-LABEL: define {{.*}}void @foo()
  a = cosf(b);
  // FPAHIGH: call float @llvm.experimental.fpaccuracy.cos.f32(float %{{.*}}) #2
  // FPAMED: call float @llvm.experimental.fpaccuracy.cos.f32(float %{{.*}}) #2
  // FPALOW: call float @llvm.experimental.fpaccuracy.cos.f32(float %{{.*}}) #2
}

// CHECK: declare float @llvm.experimental.fpaccuracy.cos.f32(float)
// CHECK-SAME: #1

// TODO: Needs to add the value of the error.
// CHECK: attributes #2 = { fpbultin_max_error }
