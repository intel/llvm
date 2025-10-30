; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK: Capability Linkage
; CHECK-DAG: Name [[#AbsFun:]] "abs"
; CHECK-DAG: Name [[#ExternalFun:]] "__devicelib_abs"
; CHECK-DAG: Decorate [[#AbsFun]] LinkageAttributes "abs" Export
; CHECK-DAG: Decorate [[#ExternalFun]] LinkageAttributes "__devicelib_abs" Import

define weak dso_local spir_func i32 @abs(i32 noundef %x) {
entry:
  %call = tail call spir_func i32 @__devicelib_abs(i32 noundef %x)
  ret i32 %call
}

declare extern_weak dso_local spir_func i32 @__devicelib_abs(i32 noundef)
