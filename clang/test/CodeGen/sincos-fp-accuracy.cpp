// RUN: %clang_cc1 -ffp-builtin-accuracy=high -emit-llvm -triple \
// RUN: x86_64-unknown-linux-gnu %s -o - | FileCheck %s

float sincos(float x, float *cosval);

// CHECK-LABEL: define {{.*}}void @_Z4testf
// CHECK: call {{.*}}float @_Z6sincosfPf(float {{.*}}, ptr {{.*}})
// CHECK-NOT: call {{.*}}@llvm.fpbuiltin.sincos
void test(float x) {
  float c;
  float s = sincos(x, &c);
}
