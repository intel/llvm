; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_global_variable_fpga_decorations,+SPV_INTEL_global_variable_host_access -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-SPV-IR

; Expected to fail - the decorations require enabled extensions to be translated.
; RUN: not llvm-spirv %t.bc -o %t.spv
; RUN: not llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_global_variable_fpga_decorations -o %t.spv
; RUN: not llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_global_variable_host_access -o %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@int_var = addrspace(1) global i32 42, !spirv.Decorations !1
@float_var = addrspace(1) global float 1.0, !spirv.Decorations !6
@bool_var = addrspace(1) global i1 0, !spirv.Decorations !9

; CHECK-SPIRV-DAG: Capability GlobalVariableFPGADecorationsINTEL
; CHECK-SPIRV-DAG: Capability GlobalVariableHostAccessINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_global_variable_fpga_decorations"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_global_variable_host_access"
; CHECK-SPIRV: Decorate [[#INT_VAR_ID:]] HostAccessINTEL ReadINTEL "IntVarName"
; CHECK-SPIRV: Decorate [[#INT_VAR_ID]] ImplementInRegisterMapINTEL 1
; CHECK-SPIRV: Decorate [[#INT_VAR_ID]] InitModeINTEL InitOnDeviceReprogramINTEL

; CHECK-SPIRV: Decorate [[#FLOAT_VAR_ID:]] ImplementInRegisterMapINTEL 1
; CHECK-SPIRV: Decorate [[#FLOAT_VAR_ID]] InitModeINTEL InitOnDeviceResetINTEL

; CHECK-SPIRV: Decorate [[#BOOL_VAR_ID:]] HostAccessINTEL ReadWriteINTEL "BoolVarName"
; CHECK-SPIRV: Decorate [[#BOOL_VAR_ID]] ImplementInRegisterMapINTEL 0
; CHECK-SPIRV: Decorate [[#BOOL_VAR_ID]] InitModeINTEL InitOnDeviceReprogramINTEL

; 5 is a global storage
; CHECK-SPIRV: Variable [[#]] [[#INT_VAR_ID]] 5
; CHECK-SPIRV: Variable [[#]] [[#FLOAT_VAR_ID]] 5
; CHECK-SPIRV: Variable [[#]] [[#BOOL_VAR_ID]] 5

!1 = !{!2, !3, !4}
!2 = !{i32 6188, i32 1, !"IntVarName"} ; HostAccessINTEL 1 "IntVarName"
!3 = !{i32 6191, i1 true} ; ImplementInRegisterMapINTEL = true
!4 = !{i32 6190, i32 0} ; InitModeINTEL = 0
!5 = !{i32 6190, i32 1} ; InitModeINTEL = 1
!6 = !{!3, !5}
!7 = !{i32 6188, i32 3, !"BoolVarName"} ; HostAccessINTEL 3 "BoolVarName"
!8 = !{i32 6191, i1 false} ; ImplementInRegisterMapINTEL = false
!9 = !{!7, !8, !4}

; CHECK-SPV-IR: @int_var = addrspace(1) global i32 42, !spirv.Decorations ![[#INT_VAR_DEC:]]
; CHECK-SPV-IR: @float_var = addrspace(1) global float 1.000000e+00, !spirv.Decorations ![[#FLOAT_VAR_DEC:]]
; CHECK-SPV-IR: @bool_var = addrspace(1) global i1 false, !spirv.Decorations ![[#BOOL_VAR_DEC:]]

; CHECK-SPV-IR: ![[#INT_VAR_DEC]] = !{![[#]], ![[#MD_HOST_ACCESS_INTVAR:]], ![[#MD_INIT_0:]], ![[#MD_CSR_1:]]}
; CHECK-SPV-IR: ![[#MD_HOST_ACCESS_INTVAR]] = !{i32 6188, i32 1, !"IntVarName"}
; CHECK-SPV-IR: ![[#MD_INIT_0]] = !{i32 6190, i32 0}
; CHECK-SPV-IR: ![[#MD_CSR_1]] = !{i32 6191, i32 1}
; CHECK-SPV-IR: ![[#FLOAT_VAR_DEC]] = !{![[#]], ![[#MD_INIT_1:]], ![[#MD_CSR_1]]}
; CHECK-SPV-IR: ![[#MD_INIT_1]] = !{i32 6190, i32 1}
; CHECK-SPV-IR: ![[#BOOL_VAR_DEC]] = !{![[#]], ![[#MD_HOST_ACCESS_BOOLVAR:]], ![[#MD_INIT_0]], ![[#MD_CSR_0:]]}
; CHECK-SPV-IR: ![[#MD_HOST_ACCESS_BOOLVAR]] = !{i32 6188, i32 3, !"BoolVarName"}
; CHECK-SPV-IR: ![[#MD_CSR_0]] = !{i32 6191, i32 0}
