// Test if the llvm.arithmetic.fence intrinsic is created. The intrinsic
// is generated only when in unsafe math mode.

// Fast math
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-device -DFAST \
// RUN: -disable-llvm-passes -emit-llvm -mreassociate -opaque-pointers \
// RUN: -o - %s | FileCheck --check-prefixes CHECK,CHECKFAST %s

// RUN: %clang_cc1 -triple i386-pc-linux-gnu -fsycl-is-device -DFAST \
// RUN: -disable-llvm-passes -emit-llvm -mreassociate -opaque-pointers \
// RUN: -o - %s | FileCheck --check-prefixes CHECK,CHECKFAST %s

// No fast math: intrinsic not created.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-device -emit-llvm \
// RUN: -disable-llvm-passes -o - %s -opaque-pointers \
// RUN: | FileCheck --implicit-check-not="llvm.arithmetic.fence" %s

int __attribute__((sycl_device)) addit(float a, float b) {
  int v;
  // CHECK: define {{.*}}@addit(float {{.*}}, float {{.*}}) #0

  _Complex double cd, cd1;
  cd = __arithmetic_fence(cd1);
 
  // CHECKFAST: [[A:%.*]] = alloca float, align 4
  // CHECKFAST-NEXT: [[B:%.*]] = alloca float, align 4
  // CHECKFAST-NEXT: [[V:%.*]] = alloca i32, align 4
  // CHECKFAST-NEXT: [[CD:%.*]] = alloca { double, double }, align 8
  // CHECKFAST-NEXT: [[CD1:%.*]] = alloca { double, double }, align 8
  // CHECKFAST-NEXT: [[VEC1:%.*]] = alloca <2 x float>, align 8
  // CHECKFAST-NEXT: [[VEC2:%.*]] = alloca <2 x float>, align 8
  // CHECKFAST-NEXT: store float {{.*}}, ptr [[A]], align 4
  // CHECKFAST-NEXT: store float {{.*}}, ptr [[B]], align 4
  // CHECKFAST: [[GEP_CD1_R:%.*]] = getelementptr inbounds { double, double }, ptr [[CD1]], i32 0, i32 0
  // CHECKFAST-NEXT: [[CD1_REAL:%.*]] = load double, ptr [[GEP_CD1_R]], align 8
  // CHECKFAST-NEXT: [[GEP_CD1_I:%.*]] = getelementptr inbounds { double, double }, ptr [[CD1]], i32 0, i32 1
  // CHECKFAST-NEXT: [[CD1_IMAG:%.*]] = load double, ptr [[GEP_CD1_I]], align 8
  // CHECKFAST-NEXT: [[TMP0:%.*]] = call{{.*}} double @llvm.arithmetic.fence.f64(double [[CD1_REAL]])
  // CHECKFAST-NEXT: [[TMP1:%.*]] = call{{.*}} double @llvm.arithmetic.fence.f64(double [[CD1_IMAG]])

  typedef float __v2f32 __attribute__((__vector_size__(8)));
  __v2f32 vec1, vec2;
  vec1 = __arithmetic_fence(vec2);
  // CHECKFAST-NEXT: [[CD_REAL:%.*]] = getelementptr inbounds { double, double }, ptr [[CD]], i32 0, i32 0
  // CHECKFAST-NEXT: [[CD_IMAG:%.*]] = getelementptr inbounds { double, double }, ptr [[CD]], i32 0, i32 1

  // CHECKFAST-NEXT: store double [[TMP0]], ptr [[CD_REAL]], align 8
  // CHECKFAST-NEXT: store double [[TMP1]], ptr [[CD_IMAG]], align 8

  // CHECKFAST: [[L2:%.*]] = load <2 x float>, ptr [[VEC2]], align 8
  // CHECKFAST-NEXT: [[TMP3:%.*]] = call{{.*}} <2 x float> @llvm.arithmetic.fence.v2f32(<2 x float> [[L2]]
  // CHECKFAST-NEXT: store <2 x float> [[TMP3]], ptr [[VEC1]], align 8

  vec2 = (vec2 + vec1);
  // CHECKFAST-NEXT: [[TMP4:%.*]] = load <2 x float>, ptr [[VEC2]], align 8
  // CHECKFAST-NEXT: [[TMP5:%.*]] = load <2 x float>, ptr [[VEC1]], align 8
  // CHECKFAST: [[TMP6:%.*]] = fadd reassoc <2 x float> [[TMP4]], [[TMP5]]

  // CHECKFAST-NEXT: store <2 x float> [[TMP6]], ptr [[VEC2]], align 8

  v = __arithmetic_fence(a + b);
  // CHECKFAST: [[TMP9:%.*]] = load float, ptr [[A]], align 4
  // CHECKFAST-NEXT: [[TMP10:%.*]] = load float, ptr [[B]], align 4
  // CHECKFAST-NEXT: [[ADD1:%.*]] = fadd reassoc float [[TMP9]], [[TMP10]]
  // CHECKFAST-NEXT: [[CALL8:%.*]] = call{{.*}} float @llvm.arithmetic.fence.f32(float [[ADD1]])
  // CHECKFAST-NEXT: [[CONV:%.*]] = fptosi float [[CALL8]] to i32
  // CHECKFAST: store i32 [[CONV]], ptr [[V]], align 4

  v = (a + b);
  // CHECKFAST-NEXT: [[TMP11:%.*]] = load float, ptr [[A]], align 4
  // CHECKFAST-NEXT: [[TMP12:%.*]] = load float, ptr [[B]], align 4
  // CHECKFAST: [[TMP13:%.*]] = fadd reassoc float [[TMP11]], [[TMP12]]

  // CHECKFAST-NEXT: [[CONV2:%.*]] = fptosi float [[TMP13]] to i32
  // CHECKFAST-NEXT: store i32 [[CONV2]], ptr [[V]], align 4

  v = a + (b * b);
  // CHECKFAST: [[TMP15:%.*]] = load float, ptr [[A]], align 4
  // CHECKFAST-NEXT: [[TMP16:%.*]] = load float, ptr [[B]], align 4
  // CHECKFAST-NEXT: [[TMP17:%.*]] = load float, ptr [[B]], align 4
  // CHECKFAST: [[TMP18:%.*]] = fmul reassoc float [[TMP16]], [[TMP17]]

  b = (a);
  // CHECKFAST: [[TMP22:%.*]] = load float, ptr [[A]], align 4
  // CHECKFAST-NEXT: store float [[TMP22]], ptr [[B]], align 4

  (a) = b;
  // CHECKFAST-NEXT: [[TMP23:%.*]] = load float, ptr [[B]], align 4
  // CHECKFAST-NEXT: store float [[TMP23]], ptr [[A]], align 4

  return 0;
  // CHECK-NEXT ret i32 0
}

int  __attribute__((sycl_device)) addit1(int a, int b) {
  int v;
  // CHECK: define {{.*}}@addit1(i32 {{.*}}, i32 {{.*}}
  v = (a + b);
  // CHECK-NOT: call{{.*}} float @llvm.arithmetic.fence.int(float {{.*}})
  return 0;
}

#ifdef FAST
#pragma float_control(precise, on)
int __attribute__((sycl_device)) subit(float a, float b, float *fp) {
  // CHECKFAST: define {{.*}}@subit(float {{.*}}, float {{.*}}
  *fp = __arithmetic_fence(a - b);
  *fp = (a + b);
  // CHECK-NOT: call{{.*}} float @llvm.arithmetic.fence.f32(float {{.*}})
  return 0;
}
#endif
