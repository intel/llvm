; Check translation of the buffer surface target extension type
;
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-ext=+SPV_INTEL_vector_compute %t.bc -spirv-text -o %t.spv.txt
; RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV: Capability VectorComputeINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_vector_compute"
; CHECK-SPIRV: EntryPoint {{[0-9]+}} [[#FuncName:]] "foo"
; CHECK-SPIRV: Name [[#ParamName:]] "a"
; CHECK-SPIRV: TypeVoid  [[#VoidT:]]
; CHECK-SPIRV: TypeBufferSurfaceINTEL [[#BufferID:]]
; CHECK-SPIRV: Function [[#VoidT]] [[#FuncID:]]
; CHECK-SPIRV-NEXT: FunctionParameter [[#BufferID]] [[#ParamName]]

define spir_kernel void @foo(target("spirv.BufferSurfaceINTEL", 0) %a) #0 {
  entry:
  ret void
 }

attributes #0 = { noinline norecurse nounwind readnone "VCFunction"}