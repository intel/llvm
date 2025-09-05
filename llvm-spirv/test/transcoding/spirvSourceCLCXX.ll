; RUN: llvm-as < %s > %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM

; CHECK-SPIRV: Source 6 100000
; CHECK-LLVM: !spirv.Source = !{[[SPVSource:![0-9]+]]}
; CHECK-LLVM: !opencl.cxx.version = !{[[OCLCXXVer:![0-9]+]]}
; CHECK-LLVM: !opencl.ocl.version = !{[[OCLVer:![0-9]+]]}
; CHECK-LLVM: [[SPVSource]] = !{i32 6, i32 100000}
; CHECK-LLVM: [[OCLCXXVer]] = !{i32 1, i32 0}
; CHECK-LLVM: [[OCLVer]] = !{i32 2, i32 0}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

!opencl.ocl.version = !{!1}
!opencl.cxx.version = !{!2}
!opencl.spir.version = !{!1}

!1 = !{i32 2, i32 0}
!2 = !{i32 1, i32 0}
