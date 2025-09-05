; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text --spirv-ext=+SPV_INTEL_maximum_registers %t.bc
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_maximum_registers -o %t.spv
; RUN: llvm-spirv -r %t.spv -spirv-target-env=SPV-IR -o - | llvm-dis -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: EntryPoint [[#]] [[#FUNC0:]] "main_l3"
; CHECK-SPIRV: EntryPoint [[#]] [[#FUNC1:]] "main_l6"
; CHECK-SPIRV: EntryPoint [[#]] [[#FUNC2:]] "main_l9"
; CHECK-SPIRV: EntryPoint [[#]] [[#FUNC3:]] "main_l13"
; CHECK-SPIRV: EntryPoint [[#]] [[#FUNC4:]] "main_l19"

; CHECK-SPIRV: ExecutionMode [[#FUNC0]] 6461 2
; CHECK-SPIRV: ExecutionMode [[#FUNC1]] 6461 1
; CHECK-SPIRV: ExecutionMode [[#FUNC2]] 6463 0
; CHECK-SPIRV: ExecutionModeId [[#FUNC3]] 6462 [[#Const3:]]
; CHECK-SPIRV: TypeInt [[#TypeInt:]] 32 0
; CHECK-SPIRV: Constant [[#TypeInt]] [[#Const3]] 3

; CHECK-SPIRV-NOT: ExecutionMode [[#FUNC4]]

; CHECK-LLVM: !spirv.ExecutionMode = !{![[#FLAG0:]], ![[#FLAG1:]], ![[#FLAG2:]], ![[#FLAG3:]]}
; CHECK-LLVM: ![[#FLAG0]] = !{ptr @main_l3, i32 6461, i32 2}
; CHECK-LLVM: ![[#FLAG1]] = !{ptr @main_l6, i32 6461, i32 1}
; CHECK-LLVM: ![[#FLAG2]] = !{ptr @main_l9, i32 6463, !"AutoINTEL"}
; CHECK-LLVM: ![[#FLAG3]] = !{ptr @main_l13, i32 6462, ![[#VAL:]]}
; CHECK-LLVM: ![[#VAL]] = !{i32 3}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; Function Attrs: noinline nounwind optnone
define weak dso_local spir_kernel void @main_l3() #0 !RegisterAllocMode !10 {
newFuncRoot:
  ret void
}

; Function Attrs: noinline nounwind optnone
define weak dso_local spir_kernel void @main_l6() #0 !RegisterAllocMode !11 {
newFuncRoot:
  ret void
}

; Function Attrs: noinline nounwind optnone
define weak dso_local spir_kernel void @main_l9() #0 !RegisterAllocMode !12 {
newFuncRoot:
  ret void
}

; Function Attrs: noinline nounwind optnone
define weak dso_local spir_kernel void @main_l13() #0 !RegisterAllocMode !13 {
newFuncRoot:
  ret void
}

; Function Attrs: noinline nounwind optnone
define weak dso_local spir_kernel void @main_l19() #0 {
newFuncRoot:
  ret void
}

attributes #0 = { noinline nounwind optnone }


!opencl.compiler.options = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!2, !3, !3, !3, !3, !3, !2, !3, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!llvm.module.flags = !{!4, !5, !6, !7, !8}
!spirv.MemoryModel = !{!9, !9, !9, !9, !9, !9}
!spirv.ExecutionMode = !{}

!0 = !{}
!2 = !{i32 4, i32 200000}
!3 = !{i32 3, i32 200000}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"openmp", i32 50}
!6 = !{i32 7, !"openmp-device", i32 50}
!7 = !{i32 8, !"PIC Level", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{i32 2, i32 2}
!10 = !{i32 2}
!11 = !{i32 1}
!12 = !{!"AutoINTEL"}
!13 = !{!14}
!14 = !{i32 3}
