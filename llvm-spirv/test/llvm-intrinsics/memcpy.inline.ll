; RUN: llvm-spirv %s -o %t.spv
; RUN: llvm-spirv --to-text %t.spv -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target triple = "spir64-unknown-unknown"

@g = addrspace(1) global [2 x i32] zeroinitializer

; CHECK-SPIRV-DAG: TypeInt [[#I64:]] 64 0
; CHECK-SPIRV-DAG: Constant [[#I64]] [[#C8:]] 8 0
; CHECK-SPIRV-DAG: Variable [[#]] [[#G:]] 5

; CHECK-SPIRV: Variable [[#]] [[#A:]] 7
; CHECK-SPIRV: Bitcast [[#]] [[#CAST:]] [[#A]]
; CHECK-SPIRV: CopyMemorySized [[#CAST]] [[#G]] [[#C8]] 2 4

; CHECK-LLVM: call void @llvm.memcpy.p0.p1.i64(ptr align 4 %{{.*}}, ptr addrspace(1) align 4 @g, i64 8, i1 false)

define spir_kernel void @copy() {
entry:
  %a = alloca { i32, float }, align 4
  call void @llvm.memcpy.inline.p0.p1.i64(ptr align 4 %a, ptr addrspace(1) align 4 @g, i64 8, i1 false)
  ret void
}
