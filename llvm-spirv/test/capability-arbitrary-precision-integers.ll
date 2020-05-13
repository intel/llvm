; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_arbitrary_precision_integers -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv -spirv-text -r %t.spt -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability ArbitraryPrecisionIntegersINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_arbitrary_precision_integers"

; CHECK-SPIRV-DAG: TypeInt {{[0-9]+}} 13 0
; CHECK-SPIRV-DAG: TypeInt {{[0-9]+}} 58 0
; CHECK-SPIRV-DAG: TypeInt {{[0-9]+}} 30 0

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-LLVM: @a = addrspace(1) global i13 0, align 2
; CHECK-LLVM: @b = addrspace(1) global i58 0, align 8
@a = addrspace(1) global i13 0, align 2
@b = addrspace(1) global i58 0, align 8

; Function Attrs: noinline nounwind optnone
; CHECK-LLVM: void @_Z4funci(i30 %a)
define spir_func void @_Z4funci(i30 %a) {
entry:
; CHECK-LLVM: %a.addr = alloca i30
  %a.addr = alloca i30, align 4
; CHECK-LLVM: store i30 %a, i30* %a.addr
  store i30 %a, i30* %a.addr, align 4
; CHECK-LLVM: store i30 1, i30* %a.addr
  store i30 1, i30* %a.addr, align 4
  ret void
}
