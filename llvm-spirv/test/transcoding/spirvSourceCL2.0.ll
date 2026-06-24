; RUN: llvm-as < %s > %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM
; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefixes=CHECK-LLVM < %t.llc.rev.ll %}

; CHECK-SPIRV: Source 3 200000
; CHECK-LLVM-NOT: opencl.cxx.version
; CHECK-LLVM: !spirv.Source = !{[[SPVSource:![0-9]+]]}
; CHECK-LLVM: !opencl.ocl.version = !{[[OCLVer:![0-9]+]]}
; CHECK-LLVM: [[SPVSource]] = !{i32 3, i32 200000}
; CHECK-LLVM: [[OCLVer]] = !{i32 2, i32 0}


target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}

!0 = !{i32 2, i32 0}
