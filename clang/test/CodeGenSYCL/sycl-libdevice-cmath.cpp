// SYCL compilation uses libdevice in order to implement platform specific
// versions of funcs like cosf, logf, etc. In order for the libdevice funcs
// to be used, we need to make sure that llvm intrinsics such as llvm.cos.f32
// are not emitted since many backends do not have lowerings for such
// intrinsics. This allows the driver to link in the libdevice definitions for
// cosf etc. later in the driver flow.

// RUN: %clang_cc1 %s -fsycl-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple nvptx64-nvidia-cuda -ffast-math -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple amdgcn-amd-amdhsa -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple amdgcn-amd-amdhsa -ffast-math -emit-llvm -o - | FileCheck %s

#include "Inputs/sycl.hpp"

extern "C" {
float scalbnf(float x, int n);
float logf(float x);
float expf(float x);
float frexpf(float x, int *exp);
float ldexpf(float x, int exp);
float log10f(float x);
float modff(float x, float *intpart);
float exp2f(float x);
float expm1f(float x);
int ilogbf(float x);
float log1pf(float x);
float log2f(float x);
float logbf(float x);
float sqrtf(float x);
float cbrtf(float x);
float hypotf(float x, float y);
float erff(float x);
float erfcf(float x);
float tgammaf(float x);
float lgammaf(float x);
float fmodf(float x, float y);
float remainderf(float x, float y);
float remquof(float x, float y, int *q);
float nextafterf(float x, float y);
float fdimf(float x, float y);
float fmaf(float x, float y, float z);
float sinf(float x);
float cosf(float x);
float tanf(float x);
float powf(float x, float y);
float acosf(float x);
float asinf(float x);
float atanf(float x);
float atan2f(float x, float y);
float coshf(float x);
float sinhf(float x);
float tanhf(float x);
float acoshf(float x);
float asinhf(float x);
float atanhf(float x);
};

// CHECK-NOT: llvm.abs.
// CHECK-NOT: llvm.scalbnf.
// CHECK-NOT: llvm.log.
// CHECK-NOT: llvm.exp.
// CHECK-NOT: llvm.frexp.
// CHECK-NOT: llvm.ldexp.
// CHECK-NOT: llvm.log10.
// CHECK-NOT: llvm.mod.
// CHECK-NOT: llvm.exp2.
// CHECK-NOT: llvm.expm1.
// CHECK-NOT: llvm.ilogb.
// CHECK-NOT: llvm.log1p.
// CHECK-NOT: llvm.log2.
// CHECK-NOT: llvm.logb.
// CHECK-NOT: llvm.sqrt.
// CHECK-NOT: llvm.cbrt.
// CHECK-NOT: llvm.hypot.
// CHECK-NOT: llvm.erf.
// CHECK-NOT: llvm.erfc.
// CHECK-NOT: llvm.tgamma.
// CHECK-NOT: llvm.lgamma.
// CHECK-NOT: llvm.fmod.
// CHECK-NOT: llvm.remainder.
// CHECK-NOT: llvm.remquo.
// CHECK-NOT: llvm.nextafter.
// CHECK-NOT: llvm.fdim.
// CHECK-NOT: llvm.fma.
// CHECK-NOT: llvm.sin.
// CHECK-NOT: llvm.cos.
// CHECK-NOT: llvm.tan.
// CHECK-NOT: llvm.pow.
// CHECK-NOT: llvm.acos.
// CHECK-NOT: llvm.asin.
// CHECK-NOT: llvm.atan.
// CHECK-NOT: llvm.atan2.
// CHECK-NOT: llvm.cosh.
// CHECK-NOT: llvm.sinh.
// CHECK-NOT: llvm.tanh.
// CHECK-NOT: llvm.acosh.
// CHECK-NOT: llvm.asinh.
// CHECK-NOT: llvm.atanh.
void sycl_kernel(float *a, int *b) {
  sycl::queue{}.submit([&](sycl::handler &cgh) {
    cgh.single_task<class kernel>([=]() {
      a[0] = scalbnf(a[0], b[0]);
      a[0] = logf(a[0]);
      a[0] = expf(a[0]);
      a[0] = frexpf(a[0], b);
      a[0] = ldexpf(a[0], b[0]);
      a[0] = log10f(a[0]);
      a[0] = modff(a[0], a);
      a[0] = exp2f(a[0]);
      a[0] = expm1f(a[0]);
      a[0] = ilogbf(a[0]);
      a[0] = log1pf(a[0]);
      a[0] = log2f(a[0]);
      a[0] = logbf(a[0]);
      a[0] = sqrtf(a[0]);
      a[0] = cbrtf(a[0]);
      a[0] = hypotf(a[0], a[0]);
      a[0] = erff(a[0]);
      a[0] = erfcf(a[0]);
      a[0] = tgammaf(a[0]);
      a[0] = lgammaf(a[0]);
      a[0] = fmodf(a[0], a[0]);
      a[0] = remainderf(a[0], a[0]);
      a[0] = remquof(a[0], a[0], b);
      a[0] = nextafterf(a[0], a[0]);
      a[0] = fdimf(a[0], a[0]);
      a[0] = fmaf(a[0], a[0], a[0]);
      a[0] = sinf(a[0]);
      a[0] = cosf(a[0]);
      a[0] = tanf(a[0]);
      a[0] = powf(a[0], a[0]);
      a[0] = acosf(a[0]);
      a[0] = asinf(a[0]);
      a[0] = atanf(a[0]);
      a[0] = atan2f(a[0], a[0]);
      a[0] = coshf(a[0]);
      a[0] = sinhf(a[0]);
      a[0] = tanhf(a[0]);
      a[0] = acoshf(a[0]);
      a[0] = asinhf(a[0]);
      a[0] = atanhf(a[0]);
    });
  });
}
