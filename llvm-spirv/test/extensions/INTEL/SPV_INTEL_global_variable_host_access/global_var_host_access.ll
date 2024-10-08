; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_global_variable_host_access -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-SPV-IR

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; Expected to fail - the decorations require enabled extension to be translated.
; RUN: not llvm-spirv %t.bc -o %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@int_var = addrspace(1) global i32 42, !spirv.Decorations !1
@bool_var = addrspace(1) global i1 0, !spirv.Decorations !4

; CHECK-SPIRV: Capability GlobalVariableHostAccessINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_global_variable_host_access"
; CHECK-SPIRV: Decorate [[#INT_VAR_ID:]] HostAccessINTEL ReadINTEL "IntVarName"
; CHECK-SPIRV: Decorate [[#BOOL_VAR_ID:]] HostAccessINTEL ReadWriteINTEL "BoolVarName"

; 5 is a global storage
; CHECK-SPIRV: Variable [[#]] [[#INT_VAR_ID]] 5
; CHECK-SPIRV: Variable [[#]] [[#BOOL_VAR_ID]] 5

!1 = !{!2}
!2 = !{i32 6188, i32 1, !"IntVarName"} ; HostAccessINTEL 1 "IntVarName"
!3 = !{i32 6188, i32 3, !"BoolVarName"} ; HostAccessINTEL 3 "BoolVarName"
!4 = !{!3}

; CHECK-SPV-IR: @int_var = addrspace(1) global i32 42, !spirv.Decorations ![[#INT_VAR_DEC:]]
; CHECK-SPV-IR: @bool_var = addrspace(1) global i1 false, !spirv.Decorations ![[#BOOL_VAR_DEC:]]

; CHECK-SPV-IR: ![[#INT_VAR_DEC]] = !{![[#]], ![[#MD_HOST_ACCESS_INTVAR:]]}
; CHECK-SPV-IR: ![[#MD_HOST_ACCESS_INTVAR]] = !{i32 6188, i32 1, !"IntVarName"}
; CHECK-SPV-IR: ![[#BOOL_VAR_DEC]] = !{![[#]], ![[#MD_HOST_ACCESS_BOOLVAR:]]}
; CHECK-SPV-IR: ![[#MD_HOST_ACCESS_BOOLVAR]] = !{i32 6188, i32 3, !"BoolVarName"}

; CHECK-LLVM-NOT: @int_var = {{.*}}, !spirv.Decorations ![[#]]
; CHECK-LLVM-NOT: @bool_var = {{.*}}, !spirv.Decorations ![[#]]

; CHECK-LLVM: @int_var = addrspace(1) global i32 42
; CHECK-LLVM: @bool_var = addrspace(1) global i1 false
