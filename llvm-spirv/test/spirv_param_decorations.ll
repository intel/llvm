; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-IR
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: convergent nounwind
define spir_kernel void @k(float %a, float %b, float %c) #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_type_qual !7 !kernel_arg_base_type !6 !spirv.ParameterDecorations !14 {
entry:
  ret void
}

; CHECK-SPIRV: Decorate [[PId1:[0-9]+]] Restrict
; CHECK-SPIRV: Decorate [[PId1]] FPRoundingMode 2
; CHECK-SPIRV: Decorate [[PId2:[0-9]+]] Volatile
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[PId1]]
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[PId2]]

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{!"clang version 14.0.0"}
!4 = !{i32 0, i32 0, i32 0}
!5 = !{!"none", !"none", !"none"}
!6 = !{!"float", !"float", !"float"}
!7 = !{!"", !"", !""}
!8 = !{i32 19}
!9 = !{i32 39, i32 2}
!10 = !{i32 21}
!11 = !{!8, !9}
!12 = !{}
!13 = !{!10}
!14 = !{!11, !12, !13}

; CHECK-SPV-IR: define spir_kernel void @k(float %a, float %b, float %c) {{.*}} !spirv.ParameterDecorations ![[ParamDecoListId:[0-9]+]] {
; CHECK-SPV-IR-DAG: ![[ParamDecoListId]] = !{![[Param1DecoId:[0-9]+]], ![[Param2DecoId:[0-9]+]], ![[Param3DecoId:[0-9]+]]}
; CHECK-SPV-IR-DAG: ![[Param1DecoId]] = !{![[Deco1Id:[0-9]+]], ![[Deco2Id:[0-9]+]]}
; CHECK-SPV-IR-DAG: ![[Param2DecoId]] = !{}
; CHECK-SPV-IR-DAG: ![[Param3DecoId]] = !{![[Deco3Id:[0-9]+]]}
; CHECK-SPV-IR-DAG: ![[Deco1Id]] = !{i32 19}
; CHECK-SPV-IR-DAG: ![[Deco2Id]] = !{i32 39, i32 2}
; CHECK-SPV-IR-DAG: ![[Deco3Id]] = !{i32 21}

; CHECK-LLVM-NOT: define spir_kernel void @k(float %a, float %b, float %c) {{.*}} !spirv.ParameterDecorations ![[ParamDecoListId:[0-9]+]] {
; CHECK-LLVM: define spir_kernel void @k(float %a, float %b, float %c) {{.*}} {
