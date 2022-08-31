; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; CHECK: Name [[l:[0-9]+]] "l"
; CHECK: Name [[g:[0-9]+]] "g"
; CHECK: Name [[lv:[0-9]+]] "lv"
; CHECK: Name [[gv:[0-9]+]] "gv"
; CHECK: Name [[smax:[0-9]+]] "smax"
; CHECK: Name [[smaxv:[0-9]+]] "smaxv"
; CHECK: Name [[smin:[0-9]+]] "smin"
; CHECK: Name [[sminv:[0-9]+]] "sminv"
; CHECK: Name [[umax:[0-9]+]] "umax"
; CHECK: Name [[umaxv:[0-9]+]] "umaxv"
; CHECK: Name [[umin:[0-9]+]] "umin"
; CHECK: Name [[uminv:[0-9]+]] "uminv"

; CHECK: SGreaterThan {{[0-9]+}} [[resg:[0-9]+]] [[l]] [[g]]
; CHECK: Select {{[0-9]+}} [[smax]] [[resg]] [[l]] [[g]]
; CHECK: SGreaterThan {{[0-9]+}} [[resgv:[0-9]+]] [[lv]] [[gv]]
; CHECK: Select {{[0-9]+}} [[smaxv]] [[resgv]] [[lv]] [[gv]]
; CHECK: SLessThan {{[0-9]+}} [[resl:[0-9]+]] [[l]] [[g]]
; CHECK: Select {{[0-9]+}} [[smin]] [[resl]] [[l]] [[g]]
; CHECK: SLessThan {{[0-9]+}} [[reslv:[0-9]+]] [[lv]] [[gv]]
; CHECK: Select {{[0-9]+}} [[sminv]] [[reslv]] [[lv]] [[gv]]
; CHECK: UGreaterThan {{[0-9]+}} [[reug:[0-9]+]] [[l]] [[g]]
; CHECK: Select {{[0-9]+}} [[umax]] [[reug]] [[l]] [[g]]
; CHECK: UGreaterThan {{[0-9]+}} [[reugv:[0-9]+]] [[lv]] [[gv]]
; CHECK: Select {{[0-9]+}} [[umaxv]] [[reugv]] [[lv]] [[gv]]
; CHECK: ULessThan {{[0-9]+}} [[reul:[0-9]+]] [[l]] [[g]]
; CHECK: Select {{[0-9]+}} [[umin]] [[reul]] [[l]] [[g]]
; CHECK: ULessThan {{[0-9]+}} [[reusv:[0-9]+]] [[lv]] [[gv]]
; CHECK: Select {{[0-9]+}} [[uminv]] [[reusv]] [[lv]] [[gv]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test(i32 %l, i32 %g, <4 x i32> %lv, <4 x i32> %gv) #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  %smax = call i32 @llvm.smax.i32(i32 %l, i32 %g) #1
  %smaxv = call <4 x i32> @llvm.smax.v4i32(<4 x i32> %lv, <4 x i32> %gv) #1
  %smin = call i32 @llvm.smin.i32(i32 %l,i32 %g)  #1
  %sminv = call <4 x i32> @llvm.smin.v4i32(<4 x i32> %lv, <4 x i32> %gv) #1
  %umax = call i32 @llvm.umax.i32(i32 %l, i32 %g) #1
  %umaxv = call <4 x i32> @llvm.umax.v4i32(<4 x i32> %lv, <4 x i32> %gv) #1
  %umin = call i32 @llvm.umin.i32(i32 %l, i32 %g) #1
  %uminv = call <4 x i32> @llvm.umin.v4i32(<4 x i32> %lv, <4 x i32> %gv) #1
  ret void
}

declare i32 @llvm.smax.i32(i32 %a, i32 %b)
declare <4 x i32> @llvm.smax.v4i32(<4 x i32> %a, <4 x i32> %b)

declare i32 @llvm.smin.i32(i32 %a, i32 %b)
declare <4 x i32> @llvm.smin.v4i32(<4 x i32> %a, <4 x i32> %b)

declare i32 @llvm.umax.i32(i32 %a, i32 %b)
declare <4 x i32> @llvm.umax.v4i32(<4 x i32> %a, <4 x i32> %b)

declare i32 @llvm.umin.i32(i32 %a, i32 %b)
declare <4 x i32> @llvm.umin.v4i32(<4 x i32> %a, <4 x i32> %b)

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!1}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!0}
!opencl.used.optional.core.features = !{!0}
!opencl.compiler.options = !{!0}

!0 = !{}
!1 = !{i32 1, i32 2}
!2 = !{i32 2, i32 0}
