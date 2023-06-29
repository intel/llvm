// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffp-builtin-accuracy=high \
// RUN: -Wno-return-type -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefixes=CHECK %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: "-ffp-builtin-accuracy=high:[acosf,cos,pow] low:[tan] medium:[sincos,log10]" \
// RUN: -Wno-return-type -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=CHECK-F1 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: "-ffp-builtin-accuracy=medium high:[tan] cuda:[cos]" \
// RUN: -Wno-return-type -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=CHECK-F2 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown		     \
// RUN: "-ffp-builtin-accuracy=high low:[tan] medium:[sincos,log10]" \
// RUN: -Wno-return-type -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=CHECK-F3 %s

// RUN: %clang_cc1 -triple spir64-unknown-unknown -ffp-builtin-accuracy=sycl \
// RUN: -D SPIR -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=CHECK-SPIR %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: "-ffp-builtin-accuracy=default:[acosf,cos,pow]" \
// RUN: -Wno-return-type -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefixes=CHECK-DEFAULT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -Wno-return-type -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefixes=CHECK-DEFAULT %s

#ifdef SPIR
// This is a declaration when compiling with -fsycl to avoid
// the compilation error "function with no prototype cannot use
// the spir_function calling convention".
void sincos(float, float *, float *);
double exp10(double);
double fadd(double, double);
float fdiv(float, float);
float fmul(float, float);
float frem(float, float);
float fsub(float, float);
double rsqrt(double);
#endif


// CHECK-LABEL: define dso_local void @f1
// CHECK: call double @llvm.fpbuiltin.acos.f64(double {{.*}}) #[[ATTR_HIGH:[0-9]+]]
// CHECK: call double @llvm.fpbuiltin.acosh.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.asin.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.asinh.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.atan.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.atan2.f64(double {{.*}}, double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.atanh.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.cosh.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.erf.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.erfc.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.exp.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.exp10.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.exp2.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.expm1.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.fadd.f64(double {{.*}}, double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.fdiv.f64(double {{.*}}, double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.fmul.f64(double {{.*}}, double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.frem.f64(double {{.*}}, double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.fsub.f64(double {{.*}}, double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.hypot.f64(double {{.*}}, double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.ldexp.f64(double {{.*}}, i32 {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.log.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.log1p.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.log2.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.pow.f64(double {{.*}}, double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.rsqrt.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK:    call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.sinh.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.sqrt.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.tanh.f64(double {{.*}}) #[[ATTR_HIGH]]

// CHECK-F1-LABEL: define dso_local void @f1
// CHECK-F1: call double @llvm.fpbuiltin.acos.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.acosh.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.asin.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.asinh.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.atan.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.atan2.f64(double {{.*}}, double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.atanh.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR_F1_HIGH:[0-9]+]]
// CHECK-F1: call double @llvm.fpbuiltin.cosh.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.erf.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.erfc.f64(double {{.*}})
// CHECK-F1: call double @llvm.exp.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.exp10.f64(double {{.*}})
// CHECK-F1: call double @llvm.exp2.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.expm1.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.fadd.f64(double {{.*}}, double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.fdiv.f64(double {{.*}}, double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.fmul.f64(double {{.*}}, double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.frem.f64(double {{.*}}, double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.fsub.f64(double {{.*}}, double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.hypot.f64(double {{.*}}, double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.ldexp.f64(double {{.*}}, i32 {{.*}})
// CHECK-F1: call double @llvm.log.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_F1_MEDIUM:[0-9]+]]
// CHECK-F1: call double @llvm.fpbuiltin.log1p.f64(double {{.*}})
// CHECK-F1: call double @llvm.log2.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.pow.f64(double {{.*}}, double {{.*}}) #[[ATTR_F1_HIGH]]
// CHECK-F1: call double @llvm.fpbuiltin.rsqrt.f64(double {{.*}})
// CHECK-F1: call double @llvm.sin.f64(double {{.*}})
// CHECK-F1: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_F1_MEDIUM]]
// CHECK-F1: call double @llvm.fpbuiltin.sinh.f64(double {{.*}})
// CHECK-F1: call double @llvm.sqrt.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F1_LOW:[0-9]+]]
// CHECK-F1: call double @llvm.fpbuiltin.tanh.f64(double {{.*}})
//
// CHECK-F2-LABEL: define dso_local void @f1
// CHECK-F2: call double @llvm.fpbuiltin.acos.f64(double {{.*}}) #[[ATTR_F2_MEDIUM:[0-9]+]]
// CHECK-F2: call double @llvm.fpbuiltin.acosh.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.asin.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.asinh.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.atan.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.atan2.f64(double {{.*}}, double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.atanh.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR_F2_CUDA:[0-9]+]]
// CHECK-F2: call double @llvm.fpbuiltin.cosh.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.erf.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.erfc.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.exp.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.exp10.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.exp2.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.expm1.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.fadd.f64(double {{.*}}, double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.fdiv.f64(double {{.*}}, double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.fmul.f64(double {{.*}}, double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.frem.f64(double {{.*}}, double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.fsub.f64(double {{.*}}, double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.hypot.f64(double {{.*}}, double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.ldexp.f64(double {{.*}}, i32 {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.log.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.log1p.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.log2.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.pow.f64(double {{.*}}, double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.rsqrt.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2:    call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.sinh.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.sqrt.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F2_HIGH:[0-9]+]]
// CHECK-F2: call double @llvm.fpbuiltin.tanh.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
//
// CHECK-F3-LABEL: define dso_local void @f1
// CHECK-F3: call double @llvm.fpbuiltin.acos.f64(double %conv) #[[ATTR_F3_HIGH:[0-9]+]]
// CHECK-F3: call double @llvm.fpbuiltin.acosh.f64(double %conv2) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.asin.f64(double %conv4) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.asinh.f64(double %conv6) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.atan.f64(double %conv8) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.atan2.f64(double %conv10, double %conv11) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.atanh.f64(double %conv13) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.cos.f64(double %conv15) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.cosh.f64(double %conv17) #[[ATTR_F3_HIGH]]
// CHECk-F3: call double @llvm.fpbuiltin.erf.f64(double %conv19) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.erfc.f64(double %conv21) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.exp.f64(double %conv23) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.exp10.f64(double %conv25) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.exp2.f64(double %conv27) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.expm1.f64(double %conv29) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.fadd.f64(double %conv31, double %conv32) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.fdiv.f64(double %conv34, double %conv35) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.fmul.f64(double %conv37, double %conv38) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.frem.f64(double %conv40, double %conv41) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.fsub.f64(double %conv43, double %conv44) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.hypot.f64(double %conv46, double %conv47) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.ldexp.f64(double %conv49, i32 %conv50) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.log.f64(double %conv52) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.log10.f64(double %conv54) #[[ATTR_F3_MEDIUM:[0-9]+]]
// CHECK-F3: call double @llvm.fpbuiltin.log1p.f64(double %conv56) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.log2.f64(double %conv58) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.pow.f64(double %conv60, double %conv61) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.rsqrt.f64(double %conv63) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.sin.f64(double %conv65) #[[ATTR_F3_HIGH]]
// CHECK-F3: call void @llvm.fpbuiltin.sincos.f64(double %conv67, ptr %p1, ptr %p2) #[[ATTR_F3_MEDIUM]]
// CHECK-F3: call double @llvm.fpbuiltin.sinh.f64(double %conv68) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.sqrt.f64(double %conv70) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.tan.f64(double %conv72) #[[ATTR_F3_LOW:[0-9]+]]
// CHECK-F3: call double @llvm.fpbuiltin.tanh.f64(double %conv74) #[[ATTR_F3_HIGH]]

// CHECK-F3: attributes #[[ATTR_F3_HIGH]] = {{.*}}"fpbuiltin-max-error="="1.0f"
// CHECK-F3: attributes #[[ATTR_F3_MEDIUM]] = {{.*}}"fpbuiltin-max-error="="4.0f"
// CHECK-F3: attributes #[[ATTR_F3_LOW]] = {{.*}}"fpbuiltin-max-error="="67108864.0f"
//
// CHECK-SPIR-LABEL: define dso_local spir_func void @f1
// CHECK-SPIR: call double @llvm.fpbuiltin.acos.f64(double {{.*}}) #[[ATTR_SYCL1:[0-9]+]]
// CHECK-SPIR: call double @llvm.fpbuiltin.acosh.f64(double {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call double @llvm.fpbuiltin.asin.f64(double {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call double @llvm.fpbuiltin.asinh.f64(double {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call double @llvm.fpbuiltin.atan.f64(double {{.*}}) #[[ATTR_SYCL2:[0-9]+]]
// CHECK-SPIR: call double @llvm.fpbuiltin.atan2.f64(double {{.*}}, double {{.*}}) #[[ATTR_SYCL3:[0-9]+]]
// CHECK-SPIR: call double @llvm.fpbuiltin.atanh.f64(double {{.*}}) #[[ATTR_SYCL2]]
// CHECK-SPIR: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call double @llvm.fpbuiltin.cosh.f64(double {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call double @llvm.fpbuiltin.erf.f64(double {{.*}}) #[[ATTR_SYCL4:[0-9]+]]
// CHECK-SPIR: call double @llvm.fpbuiltin.erfc.f64(double {{.*}}) #[[ATTR_SYCL4]]
// CHECK-SPIR: call double @llvm.fpbuiltin.exp.f64(double {{.*}}) #[[ATTR_SYCL5:[0-9]+]]
// CHECK-SPIR: call double @llvm.fpbuiltin.exp10.f64(double {{.*}}) #[[ATTR_SYCL5]]
// CHECK-SPIR: call double @llvm.fpbuiltin.exp2.f64(double {{.*}}) #[[ATTR_SYCL5]]
// CHECK-SPIR: call double @llvm.fpbuiltin.expm1.f64(double {{.*}}) #[[ATTR_SYCL5]]
// CHECK-SPIR: call double @llvm.fpbuiltin.fadd.f64(double {{.*}}, double {{.*}}) #[[ATTR_SYCL6:[0-9]+]]
// CHECK-SPIR: call float @llvm.fpbuiltin.fdiv.f32(float {{.*}}, float {{.*}}) #[[ATTR_SYCL7:[0-9]+]]
// CHECK-SPIR: call float @llvm.fpbuiltin.fmul.f32(float {{.*}}, float {{.*}}) #[[ATTR_SYCL6]]
// CHECK-SPIR: call float @llvm.fpbuiltin.frem.f32(float {{.*}}, float {{.*}}) #[[ATTR_SYCL6]]
// CHECK-SPIR: call float @llvm.fpbuiltin.fsub.f32(float {{.*}}, float {{.*}}) #[[ATTR_SYCL6]]
// CHECK-SPIR: call double @llvm.fpbuiltin.hypot.f64(double {{.*}}, double {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call double @llvm.fpbuiltin.ldexp.f64(double {{.*}}, i32 {{.*}}) #[[ATTR_SYCL6]]
// CHECK-SPIR: call double @llvm.fpbuiltin.log.f64(double {{.*}}) #[[ATTR_SYCL5]]
// CHECK-SPIR: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_SYCL5]]
// CHECK-SPIR: call double @llvm.fpbuiltin.log1p.f64(double {{.*}}) #[[ATTR_SYCL8:[0-9]+]]
// CHECK-SPIR: call double @llvm.fpbuiltin.log2.f64(double {{.*}}) #[[ATTR_SYCL5]]
// CHECK-SPIR: call double @llvm.fpbuiltin.pow.f64(double {{.*}}, double {{.*}}) #[[ATTR_SYCL4]]
// CHECK-SPIR: call double @llvm.fpbuiltin.rsqrt.f64(double {{.*}}) #[[ATTR_SYCL8]]
// CHECK-SPIR: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR:    call void @llvm.fpbuiltin.sincos.f32(float {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call double @llvm.fpbuiltin.sinh.f64(double {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call double @llvm.fpbuiltin.sqrt.f64(double {{.*}}) #[[ATTR_SYCL6]]
// CHECK-SPIR: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_SYCL2]]
// CHECK-SPIR: call double @llvm.fpbuiltin.tanh.f64(double {{.*}}) #[[ATTR_SYCL2]]
//
void f1(float a, float b) {
  float p1 = 0.f, p2 = 0.f;

  b = acos(b);
  b = acosh(b);
  b = asin(b);
  b = asinh(b);
  b = atan(b);
  b = atan2(b,b);
  b = atanh(b);
  b = cos(b);
  b = cosh(b);
  b = erf(b);
  b = erfc(b);
  b = exp(b);
  b = exp10(b);
  b = exp2(b);
  b = expm1(b);
  b = fadd(b,b);
  b = fdiv(b,b);
  b = fmul(b,b);
  b = frem(b,b);
  b = fsub(b,b);
  b = hypot(b,b);
  b = ldexp(b,b);
  b = log(b);
  b = log10(b);
  b = log1p(b);
  b = log2(b);
  b = pow(b,b);
  b = rsqrt(b);
  b = sin(b);
  sincos(b,&p1,&p2);
  b = sinh(b);
  b = sqrt(b);
  b =tan(b);
  b = tanh(b);
}
// CHECK-LABEL: define dso_local void @f2
// CHECK: call float @llvm.fpbuiltin.cos.f32(float {{.*}}) #[[ATTR_HIGH]]
// CHECK: call float @llvm.fpbuiltin.sin.f32(float {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_HIGH]]
// CHECK: call float @tanf(float noundef {{.*}})
//
// CHECK-F1-LABEL: define dso_local void @f2
// CHECK-F1: call float @llvm.cos.f32(float {{.*}})
// CHECK-F1: call float @llvm.sin.f32(float {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F1_LOW]]
// CHECK-F1: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_F1_MEDIUM]]
// CHECK-F1: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_F1_MEDIUM]]
// CHECK-F1: call float @tanf(float noundef {{.*}})
//
// CHECK-F2-LABEL: define dso_local void @f2
// CHECK-F2: call float @llvm.fpbuiltin.cos.f32(float {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call float @llvm.fpbuiltin.sin.f32(float {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F2_HIGH]]
// CHECK-F2: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call float @tanf(float noundef {{.*}})
//
// CHECK-SPIR-LABEL: define dso_local spir_func void @f2
// CHECK-SPIR: call float @llvm.fpbuiltin.cos.f32(float {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call float @llvm.fpbuiltin.sin.f32(float {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_SYCL2]]
// CHECK-SPIR: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_SYCL5]]
// CHECK-SPIR: call void @llvm.fpbuiltin.sincos.f32(float {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call spir_func float @tanf(float noundef {{.*}})

// CHECK-LABEL: define dso_local void @f3
// CHECK: call float @fake_exp10(float {{.*}})
// CHECK-F1: call float @fake_exp10(float {{.*}})
// CHECK-F2: call float @fake_exp10(float {{.*}})
// CHECK-SPIR-LABEL: define dso_local spir_func void @f3
// CHECK-SPIR: call spir_func float @fake_exp10(float {{.*}})

// CHECK: attributes #[[ATTR_HIGH]] = {{.*}}"fpbuiltin-max-error="="1.0f"

// CHECK-F1: attributes #[[ATTR_F1_HIGH]] = {{.*}}"fpbuiltin-max-error="="1.0f"
// CHECK-F1: attributes #[[ATTR_F1_MEDIUM]] = {{.*}}"fpbuiltin-max-error="="4.0f"
// CHECK-F1: attributes #[[ATTR_F1_LOW]] = {{.*}}"fpbuiltin-max-error="="67108864.0f"

// CHECK-F2: attributes #[[ATTR_F2_MEDIUM]] = {{.*}}"fpbuiltin-max-error="="4.0f"
// CHECK-F2: attributes #[[ATTR_F2_CUDA]] = {{.*}}"fpbuiltin-max-error="="2.0f"
// CHECK-F2: attributes #[[ATTR_F2_HIGH]] = {{.*}}"fpbuiltin-max-error="="1.0f"

// CHECK-SPIR: attributes #[[ATTR_SYCL1]] = {{.*}}"fpbuiltin-max-error="="4.0f"
// CHECK-SPIR: attributes #[[ATTR_SYCL2]] = {{.*}}"fpbuiltin-max-error="="5.0f"
// CHECK-SPIR: attributes #[[ATTR_SYCL3]] = {{.*}}"fpbuiltin-max-error="="6.0f"
// CHECK-SPIR: attributes #[[ATTR_SYCL4]] = {{.*}}"fpbuiltin-max-error="="16.0f"
// CHECK-SPIR: attributes #[[ATTR_SYCL5]] = {{.*}}"fpbuiltin-max-error="="3.0f"
// CHECK-SPIR: attributes #[[ATTR_SYCL6]] = {{.*}}"fpbuiltin-max-error="="0.0f"
// CHECK-SPIR: attributes #[[ATTR_SYCL7]] = {{.*}}"fpbuiltin-max-error="="2.5f"
// CHECK-SPIR: attributes #[[ATTR_SYCL8]] = {{.*}}"fpbuiltin-max-error="="2.0f"

// CHECK-DEFAULT-LABEL: define dso_local void @f1
// CHECK-DEFAULT: call double @acos(double noundef {{.*}})
// CHECK-DEFAULT: call double @acosh(double noundef {{.*}})
// CHECK-DEFAULT: call double @asin(double noundef {{.*}})
// CHECK-DEFAULT: call double @asinh(double noundef {{.*}})
// CHECK-DEFAULT: call double @atan(double noundef {{.*}})
// CHECK-DEFAULT: call double @atan2(double noundef {{.*}}, double noundef {{.*}})
// CHECK-DEFAULT: call double @atanh(double noundef {{.*}})
// CHECK-DEFAULT: call double @llvm.cos.f64(double {{.*}})
// CHECK-DEFAULT: call double @cosh(double noundef {{.*}})
// CHECK-DEFAULT: call double @erf(double noundef {{.*}})
// CHECK-DEFAULT: call double @erfc(double noundef {{.*}})
// CHECK-DEFAULT: call double @llvm.exp.f64(double {{.*}})
// CHECK-DEFAULT: call i32 (double, ...) @exp10(double noundef {{.*}})
// CHECK-DEFAULT: call double @llvm.exp2.f64(double {{.*}})
// CHECK-DEFAULT: call double @expm1(double noundef {{.*}})
// CHECK-DEFAULT: call i32 (double, double, ...) @fadd(double noundef {{.*}}, double noundef {{.*}})
// CHECK-DEFAULT: call i32 (double, double, ...) @fdiv(double noundef {{.*}}, double noundef {{.*}})
// CHECK-DEFAULT: call i32 (double, double, ...) @fmul(double noundef {{.*}}, double noundef {{.*}})
// CHECK-DEFAULT: call i32 (double, double, ...) @frem(double noundef {{.*}}, double noundef {{.*}})
// CHECK-DEFAULT: call i32 (double, double, ...) @fsub(double noundef {{.*}}, double noundef {{.*}})
// CHECK-DEFAULT: call double @hypot(double noundef {{.*}}, double noundef {{.*}})
// CHECK-DEFAULT: call double @ldexp(double noundef {{.*}}, i32 noundef {{.*}})
// CHECK-DEFAULT: call double @llvm.log.f64(double {{.*}})
// CHECK-DEFAULT: call double @llvm.log10.f64(double {{.*}})
// CHECK-DEFAULT: call double @log1p(double noundef {{.*}})
// CHECK-DEFAULT: call double @llvm.log2.f64(double {{.*}})
// CHECK-DEFAULT: call double @llvm.pow.f64(double {{.*}}, double {{.*}})
// CHECK-DEFAULT: call i32 (double, ...) @rsqrt(double noundef {{.*}})
// CHECK-DEFAULT: call double @llvm.sin.f64(double {{.*}})
// CHECK-DEFAULT: call i32 (double, ptr, ptr, ...) @sincos(double noundef {{.*}}, ptr noundef {{.*}}, ptr noundef {{.*}})
// CHECK-DEFAULT: call double @sinh(double noundef {{.*}})
// CHECK-DEFAULT: call double @llvm.sqrt.f64(double {{.*}})
// CHECK-DEFAULT: call double @tan(double noundef {{.*}})
// CHECK-DEFAULT: call double @tanh(double noundef {{.*}})
//
// CHECK-DEFAULT-LABEL: define dso_local void @f2
// CHECK-DEFAULT: call float @llvm.cos.f32(float {{.*}})
// CHECK-DEFAULT: call float @llvm.sin.f32(float {{.*}})
// CHECK-DEFAULT: call double @tan(double noundef {{.*}})
// CHECK-DEFAULT: call double @llvm.log10.f64(double {{.*}})
// CHECK-DEFAULT: call i32 (double, ptr, ptr, ...) @sincos(double noundef {{.*}}, ptr noundef {{.*}}, ptr noundef {{.*}})
// CHECK-DEFAULT: call float @tanf(float noundef {{.*}})

// CHECK-DEFAULT-LABEL: define dso_local void @f3
// CHECK-DEFAULT: call float @fake_exp10(float {{.*}})

void f2(float a, float b) {
  float sin = 0.f, cos = 0.f;

  b = cosf(b);
  b = sinf(b);
  b = tan(b);
  b = log10(b);
  sincos(b, &sin, &cos);
  b = tanf(b);
}

float fake_exp10(float a) __attribute__((no_builtin)){}
void f3(float a, float b) {
  a = fake_exp10(b);
}
