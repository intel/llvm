; RUN: llvm-as < %s > %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM

; CHECK-SPIRV: Source 4 100000
; CHECK-LLVM-NOT: opencl.cxx.version
; CHECK-LLVM-NOT: opencl.ocl.version
; CHECK-LLVM: !spirv.Source = !{[[SPVSource:![0-9]+]]}
; CHECK-LLVM: [[SPVSource]] = !{i32 4, i32 100000}

; This lit checks SourceLanguageOpenCL_CPP is preserved on spirv->LLVM IR translation.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

!spirv.Source = !{!0}

!0 = !{i32 4, i32 100000}
