; REQUIRES: pass-plugin
; UNSUPPORTED: target={{.*windows.*}}

; RUN: opt %load_spirv_lib -passes=llvm-to-spirv -disable-output -debug-pass-manager %s 2>&1 | FileCheck %s

; CHECK: Running pass: SPIRV::LLVMToSPIRVPass
; CHECK: Running analysis: SPIRV::OCLTypeToSPIRVPass

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"
