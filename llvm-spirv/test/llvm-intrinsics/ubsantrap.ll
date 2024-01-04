; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target triple = "spir64-unknown-unknown"

define spir_func void @test_ubsantrap() {
entry:
  call void @llvm.ubsantrap(i8 16)
  ret void
}

declare void @llvm.ubsantrap(i8)
