; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_arithmetic_fence -o %t.spv
; RUN: llvm-spirv %t.spv --to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc -o %t.negative.spv
; RUN: llvm-spirv %t.negative.spv --to-text -o %t.negative.spt
; RUN: FileCheck < %t.negative.spt %s --check-prefix=CHECK-SPIRV-NEG

; RUN: llvm-spirv -r %t.negative.spv -o %t.negative.rev.bc
; RUN: llvm-dis < %t.negative.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-NEG

; CHECK-SPIRV: Capability FPArithmeticFenceINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_arithmetic_fence"
; CHECK-SPIRV: Name [[#Res:]] "t"
; CHECK-SPIRV: TypeFloat [[#ResTy:]] 64
; CHECK-SPIRV: FAdd [[#ResTy]] [[#Target:]]
; CHECK-SPIRV: ArithmeticFenceINTEL [[#ResTy]] [[#Res]] [[#Target]]

; CHECK-LLVM: [[#Op:]] = fadd fast double %a, %a
; CHECK-LLVM: %t =  call double @llvm.arithmetic.fence.f64(double %[[#Op]])
; CHECK-LLVM: declare double @llvm.arithmetic.fence.f64(double)

; CHECK-SPIRV-NEG-NOT: Capability FPArithmeticFenceINTEL
; CHECK-SPIRV-NEG-NOT: Extension "SPV_INTEL_arithmetic_fence"
; CHECK-SPIRV-NEG-NOT: ArithmeticFenceINTEL

; CHECK-LLVM-NEG-NOT: declare double @llvm.arithmetic.fence.f64(double)

target triple = "spir64-unknown-unknown"

define double @f1(double %a) {
  %1 = fadd fast double %a, %a
  %t = call double @llvm.arithmetic.fence.f64(double %1)
  %2 = fadd fast double %a, %a
  %3 = fadd fast double %t, %2
  ret double %3
}

declare double @llvm.arithmetic.fence.f64(double)