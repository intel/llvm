// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ffp-builtin-accuracy=high \
// RUN: -Wno-return-type -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefixes=CHECK %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: "-ffp-builtin-accuracy=high:[acosf,cos,pow] low:[tan] medium:[sincos,log10]" \
// RUN:  -Wno-return-type -Wno-implicit-function-declaration \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-F1 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: "-ffp-builtin-accuracy=medium high:[tan] cuda:[cos]" \
// RUN: -Wno-return-type -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=CHECK-F2 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: "-ffp-builtin-accuracy=high low:[tan] medium:[sincos,log10]" \
// RUN: -Wno-return-type -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=CHECK-F3 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: "-ffp-builtin-accuracy=high:sin medium" -Wno-return-type \
// RUN: -Wno-implicit-function-declaration -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefixes=CHECK-F4 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: "-ffp-builtin-accuracy=medium:[sin,cos] high:[sin,tan]" \
// RUN: -Wno-return-type -Wno-implicit-function-declaration \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK-F5 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: "-ffp-builtin-accuracy=medium high:[sin,atan]" \
// RUN: -Wno-return-type -Wno-implicit-function-declaration \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK-F6 %s

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
// CHECK: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.sinh.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.sqrt.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_HIGH]]
// CHECK: call double @llvm.fpbuiltin.tanh.f64(double {{.*}}) #[[ATTR_HIGH]]

// CHECK-F1-LABEL: define dso_local void @f1
// CHECK-F1: call double @acos(double {{.*}})
// CHECK-F1: call double @acosh(double {{.*}})
// CHECK-F1: call double @asin(double {{.*}})
// CHECK-F1: call double @asinh(double {{.*}})
// CHECK-F1: call double @atan(double {{.*}})
// CHECK-F1: call double @atan2(double {{.*}}, double {{.*}})
// CHECK-F1: call double @atanh(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR_F1_HIGH:[0-9]+]]
// CHECK-F1: call double @cosh(double {{.*}})
// CHECK-F1: call double @erf(double {{.*}})
// CHECK-F1: call double @erfc(double {{.*}})
// CHECK-F1: call i32 (double, ...) @exp10(double {{.*}})
// CHECK-F1: call double @llvm.exp2.f64(double {{.*}})
// CHECK-F1: call double @expm1(double {{.*}})
// CHECK-F1: call i32 (double, double, ...) @fadd(double {{.*}}, double {{.*}})
// CHECK-F1: call i32 (double, double, ...) @fdiv(double {{.*}}, double {{.*}})
// CHECK-F1: call i32 (double, double, ...) @fmul(double {{.*}}, double {{.*}})
// CHECK-F1: call i32 (double, double, ...) @frem(double {{.*}}, double {{.*}})
// CHECK-F1: call i32 (double, double, ...) @fsub(double {{.*}}, double {{.*}})
// CHECK-F1: call double @hypot(double {{.*}}, double {{.*}})
// CHECK-F1: call double @ldexp(double {{.*}}, i32 {{.*}})
// CHECK-F1: call double @llvm.log.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_F1_MEDIUM:[0-9]+]]
// CHECK-F1: call double @log1p(double {{.*}})
// CHECK-F1: call double @llvm.log2.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.pow.f64(double {{.*}}, double {{.*}}) #[[ATTR_F1_HIGH]]
// CHECK-F1: call i32 (double, ...) @rsqrt(double {{.*}})
// CHECK-F1: call double @llvm.sin.f64(double {{.*}})
// CHECK-F1: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_F1_MEDIUM]]
// CHECK-F1: call double @sinh(double {{.*}})
// CHECK-F1: call double @llvm.sqrt.f64(double {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F1_LOW:[0-9]+]]
// CHECK-F1: call double @tanh(double {{.*}})
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
// CHECK-F3: call double @llvm.fpbuiltin.acos.f64(double {{.*}}) #[[ATTR_F3_HIGH:[0-9]+]]
// CHECK-F3: call double @llvm.fpbuiltin.acosh.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.asin.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.asinh.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.atan.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.atan2.f64(double {{.*}}, double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.atanh.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.cosh.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECk-F3: call double @llvm.fpbuiltin.erf.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.erfc.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.exp.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.exp10.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.exp2.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.expm1.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.fadd.f64(double {{.*}}, double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.fdiv.f64(double {{.*}}, double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.fmul.f64(double {{.*}}, double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.frem.f64(double {{.*}}, double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.fsub.f64(double {{.*}}, double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.hypot.f64(double {{.*}}, double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.ldexp.f64(double {{.*}}, i32 {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.log.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_F3_MEDIUM:[0-9]+]]
// CHECK-F3: call double @llvm.fpbuiltin.log1p.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.log2.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.pow.f64(double {{.*}}, double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.rsqrt.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_F3_MEDIUM]]
// CHECK-F3: call double @llvm.fpbuiltin.sinh.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.sqrt.f64(double {{.*}}) #[[ATTR_F3_HIGH]]
// CHECK-F3: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F3_LOW:[0-9]+]]
// CHECK-F3: call double @llvm.fpbuiltin.tanh.f64(double {{.*}}) #[[ATTR_F3_HIGH]]

// CHECK-F3: attributes #[[ATTR_F3_HIGH]] = {{.*}}"fpbuiltin-max-error"="1.0"
// CHECK-F3: attributes #[[ATTR_F3_MEDIUM]] = {{.*}}"fpbuiltin-max-error"="4.0"
// CHECK-F3: attributes #[[ATTR_F3_LOW]] = {{.*}}"fpbuiltin-max-error"="67108864.0"
//
// CHECK-LABEL-F4: define dso_local void @f1
// CHECK-F4: call double @llvm.fpbuiltin.acos.f64(double {{.*}}) #[[ATTR_F4_MEDIUM:[0-9]+]]
// CHECK-F4: call double @llvm.fpbuiltin.acosh.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.asin.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.asinh.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.atan.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.atan2.f64(double {{.*}}, double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.atanh.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.cosh.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.erf.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.erfc.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.exp.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.exp10.f64(double {{.*}})
// CHECK-F4: call double @llvm.fpbuiltin.exp2.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.expm1.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.fadd.f64(double {{.*}}, double {{.*}})
// CHECK-F4: call double @llvm.fpbuiltin.fdiv.f64(double {{.*}}, double {{.*}})
// CHECK-F4: call double @llvm.fpbuiltin.fmul.f64(double {{.*}}, double {{.*}})
// CHECK-F4: call double @llvm.fpbuiltin.frem.f64(double {{.*}}, double {{.*}})
// CHECK-F4: call double @llvm.fpbuiltin.fsub.f64(double {{.*}}, double {{.*}})
// CHECK-F4: call double @llvm.fpbuiltin.hypot.f64(double {{.*}}, double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.ldexp.f64(double {{.*}}, i32 {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.log.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.log1p.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.log2.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.pow.f64(double {{.*}}, double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.rsqrt.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK-F4: call double @llvm.fpbuiltin.sinh.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.sqrt.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.tanh.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
//
// CHECK-F5-LABEL: define dso_local void @f1
// CHECK-F5: call double @acos(double {{.*}})
// CHECK-F5: call double @acosh(double {{.*}})
// CHECK-F5: call double @asin(double {{.*}})
// CHECK-F5: call double @asinh(double {{.*}})
// CHECK-F5: call double @atan(double {{.*}})
// CHECK-F5: call double @atan2(double {{.*}}, double {{.*}})
// CHECK-F5: call double @atanh(double {{.*}})
// CHECK-F5: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR_F5_MEDIUM:[0-9]+]]
// CHECK-F5: call double @cosh(double {{.*}})
// CHECK-F5: call double @erf(double {{.*}})
// CHECK-F5: call double @erfc(double {{.*}})
// CHECK-F5: call double @llvm.exp.f64(double {{.*}})
// CHECK-F5: call i32 (double, ...) @exp10(double {{.*}})
// CHECK-F5: call double @llvm.exp2.f64(double {{.*}})
// CHECK-F5: call double @expm1(double {{.*}})
// CHECK-F5: call i32 (double, double, ...) @fadd(double {{.*}}, double {{.*}})
// CHECK-F5: call i32 (double, double, ...) @fdiv(double {{.*}}, double {{.*}})
// CHECK-F5: call i32 (double, double, ...) @fmul(double {{.*}}, double {{.*}})
// CHECK-F5: call i32 (double, double, ...) @frem(double {{.*}}, double {{.*}})
// CHECK-F5: call i32 (double, double, ...) @fsub(double {{.*}}, double {{.*}})
// CHECK-F5: call double @hypot(double {{.*}}, double {{.*}})
// CHECK-F5: call double @ldexp(double {{.*}}, i32 {{.*}})
// CHECK-F5: call double @llvm.log.f64(double {{.*}})
// CHECK-F5: call double @llvm.log10.f64(double {{.*}})
// CHECK-F5: call double @log1p(double {{.*}})
// CHECK-F5: call double @llvm.log2.f64(double {{.*}})
// CHECK-F5: call double @llvm.pow.f64(double {{.*}}, double {{.*}})
// CHECK-F5: call i32 (double, ...) @rsqrt(double {{.*}})
// CHECK-F5: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) #[[ATTR_F5_HIGH:[0-9]+]]
// CHECK-F5: call i32 (double, ptr, ptr, ...) @sincos(double {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK-F5: call double @sinh(double {{.*}})
// CHECK-F5: call double @llvm.sqrt.f64(double {{.*}})
// CHECK-F5: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F5_HIGH]]
// CHECK-F5: call double @tanh(double {{.*}})
//
//
// CHECK-F6-LABEL: define dso_local void @f1
// CHECK-F6: call double @llvm.fpbuiltin.acos.f64(double {{.*}}) #[[ATTR_F6_MEDIUM:[0-9]+]]
// CHECK-F6: call double @llvm.fpbuiltin.acosh.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.asin.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.asinh.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.atan.f64(double {{.*}}) #[[ATTR_F6_HIGH:[0-9]+]]
// CHECK-F6: call double @llvm.fpbuiltin.atan2.f64(double {{.*}}, double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.atanh.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.cos.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.cosh.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.erf.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.erfc.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.exp.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.exp10.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.exp2.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.expm1.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.fadd.f64(double {{.*}}, double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.fdiv.f64(double {{.*}}, double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.fmul.f64(double {{.*}}, double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.frem.f64(double {{.*}}, double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.fsub.f64(double {{.*}}, double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.hypot.f64(double {{.*}}, double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.ldexp.f64(double {{.*}}, i32 {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.log.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.log1p.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.log2.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.pow.f64(double {{.*}}, double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.rsqrt.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.sin.f64(double {{.*}}) #[[ATTR_F6_HIGH]]
// CHECK-F6: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.sinh.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.sqrt.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.tanh.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
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
// CHECK-SPIR: call void @llvm.fpbuiltin.sincos.f32(float {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_SYCL1]]
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
// CHECK: call float @tanf(float {{.*}})
//
// CHECK-F1-LABEL: define dso_local void @f2
// CHECK-F1: call float @llvm.cos.f32(float {{.*}})
// CHECK-F1: call float @llvm.sin.f32(float {{.*}})
// CHECK-F1: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F1_LOW]]
// CHECK-F1: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_F1_MEDIUM]]
// CHECK-F1: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_F1_MEDIUM]]
// CHECK-F1: call float @tanf(float {{.*}})
//
// CHECK-F2-LABEL: define dso_local void @f2
// CHECK-F2: call float @llvm.fpbuiltin.cos.f32(float {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call float @llvm.fpbuiltin.sin.f32(float {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F2_HIGH]]
// CHECK-F2: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_F2_MEDIUM]]
// CHECK-F2: call float @tanf(float {{.*}})
//
// CHECK-LABEL-F4: define dso_local void @f2
// CHECK-F4: call float @llvm.fpbuiltin.cos.f32(float {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call float @llvm.fpbuiltin.sin.f32(float {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_F4_MEDIUM]]
// CHECK-F4: call float @tanf(float {{.*}})
//
// CHECK-F5-LABEL: define dso_local void @f2
// CHECK-F5: call float @llvm.cos.f32(float {{.*}})
// CHECK-F5: call float @llvm.sin.f32(float {{.*}})
// CHECK-F5: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F5_HIGH]]
// CHECK-F5: call double @llvm.log10.f64(double {{.*}})
// CHECK-F5: call i32 (double, ptr, ptr, ...) @sincos(double {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK-F5: call float @tanf(float {{.*}})
//
// CHECK-F5: attributes #[[ATTR_F5_MEDIUM]] = {{.*}}"fpbuiltin-max-error"="4.0"
// CHECK-F5: attributes #[[ATTR_F5_HIGH]] = {{.*}}"fpbuiltin-max-error"="1.0"
//
// CHECK-F6-LABEL: define dso_local void @f2
// CHECK-F6: call float @llvm.fpbuiltin.cos.f32(float {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call float @llvm.fpbuiltin.sin.f32(float {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call void @llvm.fpbuiltin.sincos.f64(double {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_F6_MEDIUM]]
// CHECK-F6: call float @tanf(float {{.*}}) #[[ATTR8:[0-9]+]]
//
// CHECK-F6: attributes #[[ATTR_F6_MEDIUM]] = {{.*}}"fpbuiltin-max-error"="4.0"
// CHECK-F6: attributes #[[ATTR_F6_HIGH]] = {{.*}}"fpbuiltin-max-error"="1.0"
//
// CHECK-SPIR-LABEL: define dso_local spir_func void @f2
// CHECK-SPIR: call float @llvm.fpbuiltin.cos.f32(float {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call float @llvm.fpbuiltin.sin.f32(float {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call double @llvm.fpbuiltin.tan.f64(double {{.*}}) #[[ATTR_SYCL2]]
// CHECK-SPIR: call double @llvm.fpbuiltin.log10.f64(double {{.*}}) #[[ATTR_SYCL5]]
// CHECK-SPIR: call void @llvm.fpbuiltin.sincos.f32(float {{.*}}, ptr {{.*}}, ptr {{.*}}) #[[ATTR_SYCL1]]
// CHECK-SPIR: call spir_func float @tanf(float {{.*}})

// CHECK-LABEL: define dso_local void @f3
// CHECK: call float @fake_exp10(float {{.*}})
// CHECK-F1: call float @fake_exp10(float {{.*}})
// CHECK-F2: call float @fake_exp10(float {{.*}})

// CHECK-SPIR-LABEL: define dso_local spir_func void @f3
// CHECK-SPIR: call spir_func float @fake_exp10(float {{.*}})

// CHECK: attributes #[[ATTR_HIGH]] = {{.*}}"fpbuiltin-max-error"="1.0"

// CHECK-F1: attributes #[[ATTR_F1_HIGH]] = {{.*}}"fpbuiltin-max-error"="1.0"
// CHECK-F1: attributes #[[ATTR_F1_MEDIUM]] = {{.*}}"fpbuiltin-max-error"="4.0"
// CHECK-F1: attributes #[[ATTR_F1_LOW]] = {{.*}}"fpbuiltin-max-error"="67108864.0"

// CHECK-F2: attributes #[[ATTR_F2_MEDIUM]] = {{.*}}"fpbuiltin-max-error"="4.0"
// CHECK-F2: attributes #[[ATTR_F2_CUDA]] = {{.*}}"fpbuiltin-max-error"="2.0"
// CHECK-F2: attributes #[[ATTR_F2_HIGH]] = {{.*}}"fpbuiltin-max-error"="1.0"

// CHECK-SPIR: attributes #[[ATTR_SYCL1]] = {{.*}}"fpbuiltin-max-error"="4.0"
// CHECK-SPIR: attributes #[[ATTR_SYCL2]] = {{.*}}"fpbuiltin-max-error"="5.0"
// CHECK-SPIR: attributes #[[ATTR_SYCL3]] = {{.*}}"fpbuiltin-max-error"="6.0"
// CHECK-SPIR: attributes #[[ATTR_SYCL4]] = {{.*}}"fpbuiltin-max-error"="16.0"
// CHECK-SPIR: attributes #[[ATTR_SYCL5]] = {{.*}}"fpbuiltin-max-error"="3.0"
// CHECK-SPIR: attributes #[[ATTR_SYCL6]] = {{.*}}"fpbuiltin-max-error"="0.5"
// CHECK-SPIR: attributes #[[ATTR_SYCL7]] = {{.*}}"fpbuiltin-max-error"="2.5"
// CHECK-SPIR: attributes #[[ATTR_SYCL8]] = {{.*}}"fpbuiltin-max-error"="2.0"

// CHECK-DEFAULT-LABEL: define dso_local void @f1
// CHECK-DEFAULT: call double @acos(double {{.*}})
// CHECK-DEFAULT: call double @acosh(double {{.*}})
// CHECK-DEFAULT: call double @asin(double {{.*}})
// CHECK-DEFAULT: call double @asinh(double {{.*}})
// CHECK-DEFAULT: call double @atan(double {{.*}})
// CHECK-DEFAULT: call double @atan2(double {{.*}}, double {{.*}})
// CHECK-DEFAULT: call double @atanh(double {{.*}})
// CHECK-DEFAULT: call double @llvm.cos.f64(double {{.*}})
// CHECK-DEFAULT: call double @cosh(double {{.*}})
// CHECK-DEFAULT: call double @erf(double {{.*}})
// CHECK-DEFAULT: call double @erfc(double {{.*}})
// CHECK-DEFAULT: call double @llvm.exp.f64(double {{.*}})
// CHECK-DEFAULT: call i32 (double, ...) @exp10(double {{.*}})
// CHECK-DEFAULT: call double @llvm.exp2.f64(double {{.*}})
// CHECK-DEFAULT: call double @expm1(double {{.*}})
// CHECK-DEFAULT: call i32 (double, double, ...) @fadd(double {{.*}}, double {{.*}})
// CHECK-DEFAULT: call i32 (double, double, ...) @fdiv(double {{.*}}, double {{.*}})
// CHECK-DEFAULT: call i32 (double, double, ...) @fmul(double {{.*}}, double {{.*}})
// CHECK-DEFAULT: call i32 (double, double, ...) @frem(double {{.*}}, double {{.*}})
// CHECK-DEFAULT: call i32 (double, double, ...) @fsub(double {{.*}}, double {{.*}})
// CHECK-DEFAULT: call double @hypot(double {{.*}}, double {{.*}})
// CHECK-DEFAULT: call double @ldexp(double {{.*}}, i32 {{.*}})
// CHECK-DEFAULT: call double @llvm.log.f64(double {{.*}})
// CHECK-DEFAULT: call double @llvm.log10.f64(double {{.*}})
// CHECK-DEFAULT: call double @log1p(double {{.*}})
// CHECK-DEFAULT: call double @llvm.log2.f64(double {{.*}})
// CHECK-DEFAULT: call double @llvm.pow.f64(double {{.*}}, double {{.*}})
// CHECK-DEFAULT: call i32 (double, ...) @rsqrt(double {{.*}})
// CHECK-DEFAULT: call double @llvm.sin.f64(double {{.*}})
// CHECK-DEFAULT: call i32 (double, ptr, ptr, ...) @sincos(double {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK-DEFAULT: call double @sinh(double {{.*}})
// CHECK-DEFAULT: call double @llvm.sqrt.f64(double {{.*}})
// CHECK-DEFAULT: call double @tan(double {{.*}})
// CHECK-DEFAULT: call double @tanh(double {{.*}})
//
// CHECK-DEFAULT-LABEL: define dso_local void @f2
// CHECK-DEFAULT: call float @llvm.cos.f32(float {{.*}})
// CHECK-DEFAULT: call float @llvm.sin.f32(float {{.*}})
// CHECK-DEFAULT: call double @tan(double {{.*}})
// CHECK-DEFAULT: call double @llvm.log10.f64(double {{.*}})
// CHECK-DEFAULT: call i32 (double, ptr, ptr, ...) @sincos(double {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK-DEFAULT: call float @tanf(float {{.*}})

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
