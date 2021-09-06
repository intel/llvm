; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidModule: Invalid SPIR-V module: vloada_half should be of a half vector type

; ModuleID = 'loada.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: noinline nounwind
define spir_kernel void @test(half %val, half addrspace(4)* %res) #0 {
entry:
  %call1 = call spir_func float @_Z12vloada_half1mPU3AS4KDh(half %val, half addrspace(4)* %res) #1
  ret void
}

; Function Attrs: nounwind
declare spir_func float @_Z12vloada_half1mPU3AS4KDh(half, half addrspace(4)*) #1

attributes #0 = { noinline nounwind }
attributes #1 = { nounwind }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!3}
!opencl.used.extensions = !{!4}
!opencl.used.optional.core.features = !{!4}
!spirv.Generator = !{!5}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 300000}
!2 = !{i32 2, i32 0}
!3 = !{i32 3, i32 0}
!4 = !{}
!5 = !{i16 6, i16 14}
