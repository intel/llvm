; Check names and decoration of global variables.

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK: Name [[#id18:]] "G1"
; CHECK: Name [[#id22:]] "g1"
; CHECK: Name [[#id23:]] "g2"
; CHECK: Name [[#id27:]] "g4"
; CHECK: Name [[#id30:]] "c1"
; CHECK: Name [[#id31:]] "n_t"
; CHECK: Name [[#id32:]] "w"
; CHECK: Name [[#id34:]] "a.b"
; CHECK: Name [[#id35:]] "e"
; CHECK: Name [[#id36:]] "y.z"
; CHECK: Name [[#id38:]] "x"

; CHECK-NOT: Decorate [[#id18]] LinkageAttributes
; CHECK-DAG: Decorate [[#id18]] Constant
; CHECK-DAG: Decorate [[#id22]] Alignment 4
; CHECK-DAG: Decorate [[#id22]] LinkageAttributes "g1" Export
; CHECK-DAG: Decorate [[#id23]] Alignment 4
; CHECK-DAG: Decorate [[#id27]] Alignment 4
; CHECK-DAG: Decorate [[#id27]] LinkageAttributes "g4" Export
; CHECK-DAG: Decorate [[#id30]] Constant
; CHECK-DAG: Decorate [[#id30]] Alignment 4
; CHECK-DAG: Decorate [[#id30]] LinkageAttributes "c1" Export
; CHECK-DAG: Decorate [[#id31]] Constant
; CHECK-DAG: Decorate [[#id31]] LinkageAttributes "n_t" Import
; CHECK-DAG: Decorate [[#id32]] Constant
; CHECK-DAG: Decorate [[#id32]] Alignment 4
; CHECK-DAG: Decorate [[#id32]] LinkageAttributes "w" Export
; CHECK-DAG: Decorate [[#id34]] Constant
; CHECK-DAG: Decorate [[#id34]] Alignment 4
; CHECK-DAG: Decorate [[#id35]] LinkageAttributes "e" Import
; CHECK-DAG: Decorate [[#id36]] Alignment 4
; CHECK-DAG: Decorate [[#id38]] Constant
; CHECK-DAG: Decorate [[#id38]] Alignment 4

; CHECK-LLVM: @G1 = internal addrspace(1) constant %"class.sycl::_V1::nd_item" undef, align 1
; CHECK-LLVM: @g1 = addrspace(1) global i32 1, align 4
; CHECK-LLVM: @g2 = internal addrspace(1) global i32 2, align 4
; CHECK-LLVM: @g4 = common addrspace(1) global i32 0, align 4
; CHECK-LLVM: @c1 = addrspace(2) constant [2 x i32] [i32 0, i32 1], align 4
; CHECK-LLVM: @n_t = external addrspace(2) constant [256 x i32]
; CHECK-LLVM: @w = addrspace(1) constant i32 0, align 4
; CHECK-LLVM: @a.b = internal addrspace(2) constant [2 x i32] [i32 2, i32 3], align 4
; CHECK-LLVM: @e = external addrspace(1) global i32
; CHECK-LLVM: @y.z = internal addrspace(1) global i32 0, align 4
; CHECK-LLVM: @x = internal addrspace(2) constant float 1.000000e+00, align 4

%"class.sycl::_V1::nd_item" = type { i8 }

@G1 = private unnamed_addr addrspace(1) constant %"class.sycl::_V1::nd_item" poison, align 1
@g1 = addrspace(1) global i32 1, align 4
@g2 = internal addrspace(1) global i32 2, align 4
@g4 = common addrspace(1) global i32 0, align 4
@c1 = addrspace(2) constant [2 x i32] [i32 0, i32 1], align 4
@n_t = external addrspace(2) constant [256 x i32]
@w = addrspace(1) constant i32 0, align 4
@a.b = internal addrspace(2) constant [2 x i32] [i32 2, i32 3], align 4
@e = external addrspace(1) global i32
@y.z = internal addrspace(1) global i32 0, align 4
@x = internal addrspace(2) constant float 1.000000e+00, align 4

define internal spir_func void @foo(ptr addrspace(4) align 1 %arg) {
  ret void
}
