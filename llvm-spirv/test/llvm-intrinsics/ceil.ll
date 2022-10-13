; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK: ExtInstImport [[extinst_id:[0-9]+]] "OpenCL.std"

; CHECK: 3 TypeFloat [[var1:[0-9]+]] 32
; CHECK: 3 TypeFloat [[var2:[0-9]+]] 64
; CHECK: 4 TypeVector [[var3:[0-9]+]] [[var1]] 4

; CHECK: Function
; CHECK: 6 ExtInst [[var1]] {{[0-9]+}} [[extinst_id]] ceil
; CHECK: FunctionEnd

; Function Attrs: nounwind readnone
define spir_func float @TestCeil32(float %x) local_unnamed_addr #0 {
entry:
  %0 = tail call float @llvm.ceil.f32(float %x)
  ret float %0
}

; CHECK: Function
; CHECK: 6 ExtInst [[var2]] {{[0-9]+}} [[extinst_id]] ceil
; CHECK: FunctionEnd

; Function Attrs: nounwind readnone
define spir_func double @TestCeil64(double %x) local_unnamed_addr #0 {
entry:
  %0 = tail call double @llvm.ceil.f64(double %x)
  ret double %0
}

; CHECK: Function
; CHECK: 6 ExtInst [[var3]] {{[0-9]+}} [[extinst_id]] ceil
; CHECK: FunctionEnd

; Function Attrs: nounwind readnone
define spir_func <4 x float> @TestCeilVec(<4 x float> %x) local_unnamed_addr #0 {
entry:
  %0 = tail call <4 x float> @llvm.ceil.v4f32(<4 x float> %x)
  ret <4 x float> %0
}

; Function Attrs: nounwind readnone
declare float @llvm.ceil.f32(float) #1

; Function Attrs: nounwind readnone
declare double @llvm.ceil.f64(double) #1

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.ceil.v4f32(<4 x float>) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
