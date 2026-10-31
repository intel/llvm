; A use the legalizer declines to rewrite must keep the oversized value alive so
; the residual check reports it, rather than being overwritten with poison.

; RUN: not llvm-spirv -s %s -o %t.out.bc 2>&1 | FileCheck %s
; CHECK: LLVM ERROR: Unsupported vector type with 32 elements

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_kernel void @unhandled_use(ptr addrspace(1) %out, <8 x i32> %v) {
  %bc = bitcast <8 x i32> %v to <32 x i8>
  %lane = extractelement <32 x i8> %bc, i64 3
  store i8 %lane, ptr addrspace(1) %out, align 1
  store <32 x i8> %bc, ptr addrspace(1) %out, align 4
  ret void
}

!opencl.ocl.version = !{!0}
!0 = !{i32 2, i32 0}
