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
; CHECK-SPIRV-DAG: TypeInt [[#I96:]] 96 0
; CHECK-SPIRV-DAG: TypeInt [[#I128:]] 128 0
; CHECK-SPIRV-DAG: TypeInt [[#I256:]] 256 0
; CHECK-SPIRV-DAG: TypeInt [[#I2048:]] 2048 0
; CHECK-SPIRV-DAG: Constant [[#I96]] [[#]] 4 0 1
; CHECK-SPIRV-DAG: Constant [[#I128]] [[#]] 1 0 0 0
; CHECK-SPIRV-DAG: Constant [[#I256]] [[#]] 1 0 0 0 0 0 0 0
; CHECK-SPIRV-DAG: Constant [[#I2048]] [[#]] 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-LLVM: @a = addrspace(1) global i13 0, align 2
@a = addrspace(1) global i13 0, align 2
; CHECK-LLVM: @b = addrspace(1) global i58 0, align 8
@b = addrspace(1) global i58 0, align 8
; CHECK-LLVM: @c = addrspace(1) global i48 0, align 8
@c = addrspace(1) global i48 0, align 8
@d = addrspace(1) global i96 0, align 8
@e = addrspace(1) global i128 0, align 8
@f = addrspace(1) global i256 0, align 8
@g = addrspace(1) global i2048 0, align 8

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
; CHECK-LLVM: store i48 -4294901761, i48 addrspace(1)* @c
  store i48 -4294901761, i48 addrspace(1)* @c, align 8
  store i96 18446744073709551620, i96 addrspace(1)* @d, align 8
; CHECK-LLVM: store i96 18446744073709551620, i96 addrspace(1)* @d
  store i128 1, i128 addrspace(1)* @e, align 8
; CHECK-LLVM: store i128 1, i128 addrspace(1)* @e
  store i256 1, i256 addrspace(1)* @f, align 8
; CHECK-LLVM: store i256 1, i256 addrspace(1)* @f
  store i2048 1, i2048 addrspace(1)* @g, align 8
; CHECK-LLVM: store i2048 1, i2048 addrspace(1)* @g
  ret void
}
