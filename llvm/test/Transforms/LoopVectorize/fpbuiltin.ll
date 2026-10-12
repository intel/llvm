; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define void @sqrt(ptr noalias %in, ptr noalias %out, i64 %n) {
; CHECK-LABEL: @sqrt(
; CHECK: call <4 x float> @llvm.fpbuiltin.sqrt.v4f32(<4 x float> {{.*}}) #[[ATTR:[0-9]+]]
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %p.in = getelementptr inbounds float, ptr %in, i64 %iv
  %x = load float, ptr %p.in, align 4
  %r = call float @llvm.fpbuiltin.sqrt.f32(float %x) #0
  %p.out = getelementptr inbounds float, ptr %out, i64 %iv
  store float %r, ptr %p.out, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %iv.next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @pow(ptr noalias %a, ptr noalias %b, ptr noalias %out, i64 %n) {
; CHECK-LABEL: @pow(
; CHECK: call <4 x float> @llvm.fpbuiltin.pow.v4f32(<4 x float> {{.*}}, <4 x float> {{.*}}) #[[ATTR]]
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %p.a = getelementptr inbounds float, ptr %a, i64 %iv
  %x = load float, ptr %p.a, align 4
  %p.b = getelementptr inbounds float, ptr %b, i64 %iv
  %y = load float, ptr %p.b, align 4
  %r = call float @llvm.fpbuiltin.pow.f32(float %x, float %y) #0
  %p.out = getelementptr inbounds float, ptr %out, i64 %iv
  store float %r, ptr %p.out, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %iv.next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @ldexp(ptr noalias %a, ptr noalias %b, ptr noalias %out, i64 %n) {
; CHECK-LABEL: @ldexp(
; CHECK: call <4 x float> @llvm.fpbuiltin.ldexp.v4f32(<4 x float> {{.*}}, <4 x i32> {{.*}}) #[[ATTR]]
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %p.a = getelementptr inbounds float, ptr %a, i64 %iv
  %x = load float, ptr %p.a, align 4
  %p.b = getelementptr inbounds i32, ptr %b, i64 %iv
  %e = load i32, ptr %p.b, align 4
  %r = call float @llvm.fpbuiltin.ldexp.f32(float %x, i32 %e) #0
  %p.out = getelementptr inbounds float, ptr %out, i64 %iv
  store float %r, ptr %p.out, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %iv.next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @sincos(ptr noalias %in, ptr noalias %s, ptr noalias %c, i64 %n) {
; CHECK-LABEL: @sincos(
; CHECK-NOT: call {{.*}} @llvm.fpbuiltin.sincos.v
; CHECK: call void @llvm.fpbuiltin.sincos.f32.p0.p0(
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %p.in = getelementptr inbounds float, ptr %in, i64 %iv
  %x = load float, ptr %p.in, align 4
  %p.s = getelementptr inbounds float, ptr %s, i64 %iv
  %p.c = getelementptr inbounds float, ptr %c, i64 %iv
  call void @llvm.fpbuiltin.sincos.f32.p0.p0(float %x, ptr %p.s, ptr %p.c) #0
  %iv.next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %iv.next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; CHECK: attributes #[[ATTR]] = { "fpbuiltin-max-error"="1.0" }

attributes #0 = { "fpbuiltin-max-error"="1.0" }
