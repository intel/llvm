; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake-avx512 -passes=slp-vectorizer -slp-threshold=-100 -S | FileCheck %s

define void @same_accuracy(ptr %in, ptr %out) {
; CHECK-LABEL: @same_accuracy(
; CHECK: call <2 x double> @llvm.fpbuiltin.erf.v2f64(<2 x double> {{.*}}) #[[ATTR:[0-9]+]]
; CHECK: ret void
  %x0 = load double, ptr %in, align 8
  %p1 = getelementptr inbounds double, ptr %in, i64 1
  %x1 = load double, ptr %p1, align 8
  %r0 = call double @llvm.fpbuiltin.erf.f64(double %x0) #0
  %r1 = call double @llvm.fpbuiltin.erf.f64(double %x1) #0
  store double %r0, ptr %out, align 8
  %q1 = getelementptr inbounds double, ptr %out, i64 1
  store double %r1, ptr %q1, align 8
  ret void
}

define void @different_accuracy(ptr %in, ptr %out) {
; CHECK-LABEL: @different_accuracy(
; CHECK-NOT: @llvm.fpbuiltin.erf.v2f64
; CHECK: call double @llvm.fpbuiltin.erf.f64(double {{.*}}) #[[ATTR]]
; CHECK: call double @llvm.fpbuiltin.erf.f64(double {{.*}}) #[[ATTR1:[0-9]+]]
; CHECK: ret void
  %x0 = load double, ptr %in, align 8
  %p1 = getelementptr inbounds double, ptr %in, i64 1
  %x1 = load double, ptr %p1, align 8
  %r0 = call double @llvm.fpbuiltin.erf.f64(double %x0) #0
  %r1 = call double @llvm.fpbuiltin.erf.f64(double %x1) #1
  store double %r0, ptr %out, align 8
  %q1 = getelementptr inbounds double, ptr %out, i64 1
  store double %r1, ptr %q1, align 8
  ret void
}

; CHECK: attributes #[[ATTR]] = { "fpbuiltin-max-error"="4.0" }
; CHECK: attributes #[[ATTR1]] = { "fpbuiltin-max-error"="1.0" }

attributes #0 = { "fpbuiltin-max-error"="4.0" }
attributes #1 = { "fpbuiltin-max-error"="1.0" }
