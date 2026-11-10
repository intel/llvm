; RUN: not llvm-spirv %s -o %t.spv 2>&1 | FileCheck %s

; Splitting this bitcast would create bitcasts to <16 x i1>, which have a
; legal vector length but are not allowed by SPIR-V. Keep rejecting this case.
; CHECK: LLVM ERROR: Unsupported vector type with 32 elements

target triple = "spirv64"

define spir_func <4 x i1> @bool_slice(<4 x i8> %v) {
  %bc = bitcast <4 x i8> %v to <32 x i1>
  %slice = shufflevector <32 x i1> %bc, <32 x i1> poison, <4 x i32> <i32 14, i32 15, i32 16, i32 17>
  ret <4 x i1> %slice
}
