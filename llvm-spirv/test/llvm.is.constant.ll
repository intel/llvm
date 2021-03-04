; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t_recover.bc
; RUN: llvm-dis %t_recover.bc -o - | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK: spir_func i1 @TestIsConstantInt32_False
; CHECK: ret i1 false

; Function Attrs: nounwind readnone
define spir_func i1 @TestIsConstantInt32_False(i32 %x) local_unnamed_addr #0 {
entry:
  %0 = tail call i1 @llvm.is.constant.i32(i32 %x)
  ret i1 %0
}

; CHECK: spir_func i1 @TestIsConstantInt32_True
; CHECK: ret i1 true

; Function Attrs: nounwind readnone
define spir_func i1 @TestIsConstantInt32_True() local_unnamed_addr #0 {
entry:
  %0 = tail call i1 @llvm.is.constant.i32(i32 1)
  ret i1 %0
}

; CHECK: spir_func i1 @TestIsConstantInt64_False
; CHECK: ret i1 false

; Function Attrs: nounwind readnone
define spir_func i1 @TestIsConstantInt64_False(i64 %x) local_unnamed_addr #0 {
entry:
  %0 = tail call i1 @llvm.is.constant.i64(i64 %x)
  ret i1 %0
}


; CHECK: spir_func i1 @TestIsConstantInt64_True
; CHECK: ret i1 true

; Function Attrs: nounwind readnone
define spir_func i1 @TestIsConstantInt64_True() local_unnamed_addr #0 {
entry:
  %0 = tail call i1 @llvm.is.constant.i64(i64 1)
  ret i1 %0
}

; CHECK: spir_func i1 @TestIsConstantF32_False
; CHECK: ret i1 false

; Function Attrs: nounwind readnone
define spir_func i1 @TestIsConstantF32_False(float %x) local_unnamed_addr #0 {
entry:
  %0 = tail call i1 @llvm.is.constant.f32(float %x)
  ret i1 %0
}


; CHECK: spir_func i1 @TestIsConstantF32_True
; CHECK: ret i1 true

; Function Attrs: nounwind readnone
define spir_func i1 @TestIsConstantF32_True() local_unnamed_addr #0 {
entry:
  %0 = tail call i1 @llvm.is.constant.f32(float 0.5)
  ret i1 %0
}

; CHECK: spir_func i1 @TestIsConstantF64_False
; CHECK: ret i1 false

; Function Attrs: nounwind readnone
define spir_func i1 @TestIsConstantF64_False(double %x) local_unnamed_addr #0 {
entry:
  %0 = tail call i1 @llvm.is.constant.f64(double %x)
  ret i1 %0
}

; CHECK: spir_func i1 @TestIsConstantF64_True
; CHECK: ret i1 true

; Function Attrs: nounwind readnone
define spir_func i1 @TestIsConstantF64_True() local_unnamed_addr #0 {
entry:
  %0 = tail call i1 @llvm.is.constant.f64(double 0.5)
  ret i1 %0
}

; CHECK: spir_func i1 @TestIsConstantVec_False
; CHECK: ret i1 false

; Function Attrs: nounwind readnone
define spir_func i1 @TestIsConstantVec_False(<4 x float> %x) local_unnamed_addr #0 {
entry:
  %0 = tail call i1 @llvm.is.constant.v4f32(<4 x float> %x)
  ret i1 %0
}

; CHECK: spir_func i1 @TestIsConstantVec_True
; CHECK: ret i1 true

; Function Attrs: nounwind readnone
define spir_func i1 @TestIsConstantVec_True() local_unnamed_addr #0 {
entry:
  %0 = tail call i1 @llvm.is.constant.v4f32(<4 x float> <float 0.5, float 0.5, float 0.5, float 0.5>)
  ret i1 %0
}

; Function Attrs: nounwind readnone
declare i1 @llvm.is.constant.i32(i32) #1

; Function Attrs: nounwind readnone
declare i1 @llvm.is.constant.i64(i64) #1

; Function Attrs: nounwind readnone
declare i1 @llvm.is.constant.f32(float) #1

; Function Attrs: nounwind readnone
declare i1 @llvm.is.constant.f64(double) #1

; Function Attrs: nounwind readnone
declare i1 @llvm.is.constant.v4f32(<4 x float>) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
