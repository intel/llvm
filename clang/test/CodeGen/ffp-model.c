// REQUIRES: x86-registered-target
// INTEL RUN: %clang -Xclang -opaque-pointers -S -emit-llvm -ffp-model=fast -emit-llvm %s -o - \
// INTEL RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-FAST

// INTEL RUN: %clang -Xclang -opaque-pointers -S -emit-llvm -ffp-model=precise %s -o - \
// INTEL RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-PRECISE

// INTEL RUN: %clang -Xclang -opaque-pointers -S -emit-llvm -ffp-model=strict %s -o - \
// INTEL RUN: -target x86_64 | FileCheck %s --check-prefixes=CHECK,CHECK-STRICT

// INTEL RUN: %clang -Xclang -opaque-pointers -S -emit-llvm -ffp-model=strict -ffast-math \
// INTEL RUN: -target x86_64 %s -o - | FileCheck %s \
// INTEL RUN: --check-prefixes CHECK,CHECK-STRICT-FAST

// INTEL RUN: %clang -Xclang -opaque-pointers -S -emit-llvm -ffp-model=precise -ffast-math \
// INTEL RUN: %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-FAST1

float mymuladd(float x, float y, float z) {
  // CHECK: define{{.*}} float @mymuladd
  return x * y + z;

  // CHECK-FAST: fmul fast float
  // CHECK-FAST: load float, ptr
  // CHECK-FAST: fadd fast float

  // CHECK-PRECISE: load float, ptr
  // CHECK-PRECISE: load float, ptr
  // CHECK-PRECISE: load float, ptr
  // CHECK-PRECISE: call float @llvm.fmuladd.f32(float {{.*}}, float {{.*}}, float {{.*}})

  // CHECK-STRICT: load float, ptr
  // CHECK-STRICT: load float, ptr
  // CHECK-STRICT: call float @llvm.experimental.constrained.fmul.f32(float {{.*}}, float {{.*}}, {{.*}})
  // CHECK-STRICT: load float, ptr
  // CHECK-STRICT: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, {{.*}})

  // CHECK-STRICT-FAST: load float, ptr
  // CHECK-STRICT-FAST: load float, ptr
  // CHECK-STRICT-FAST: fmul fast float {{.*}}, {{.*}}
  // CHECK-STRICT-FAST: load float, ptr
  // CHECK-STRICT-FAST: fadd fast float {{.*}}, {{.*}}

  // CHECK-FAST1: load float, ptr
  // CHECK-FAST1: load float, ptr
  // CHECK-FAST1: fmul fast float {{.*}}, {{.*}}
  // CHECK-FAST1: load float, ptr {{.*}}
  // CHECK-FAST1: fadd fast float {{.*}}, {{.*}}
}
