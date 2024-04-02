; Confirm that we handle fpmath metadata correctly

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fp_max_error -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability FPMaxErrorINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_fp_max_error"
; CHECK-SPIRV: ExtInstImport [[#OCLEXTID:]] "OpenCL.std"

; CHECK-SPIRV: Name [[#CalleeName:]] "callee"
; CHECK-SPIRV: Name [[#F3:]] "f3"
; CHECK-SPIRV: Decorate [[#F3]] FPMaxErrorDecorationINTEL 1075838976
; CHECK-SPIRV: Decorate [[#Callee:]] FPMaxErrorDecorationINTEL 1065353216

; CHECK-SPIRV: TypeFloat [[#FloatTy:]] 32

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define float @callee(float %f1, float %f2) {
entry:
ret float %f1
}

define void @test_fp_max_error_decoration(float %f1, float %f2) {
entry:
; CHECK-LLVM: fdiv float %f1, %f2, !fpbuiltin-max-error ![[#ME0:]]
%f3 = fdiv float %f1, %f2, !fpmath !0

; CHECK-LLVM: call {{.*}} float @callee(float %f1, float %f2) #[[#ATTR0:]]
; CHECK-SPIRV: FunctionCall [[#FloatTy]] [[#Callee]] [[#CalleeName]]
call float @callee(float %f1, float %f2), !fpmath !1
ret void
}

; CHECK-LLVM: attributes #[[#ATTR0]] = {{{.*}}"fpbuiltin-max-error"="1.000000"{{.*}}}

; CHECK-LLVM: ![[#ME0]] = !{!"2.500000"}
!0 = !{float 2.500000e+00}
!1 = !{float 1.000000e+00}
