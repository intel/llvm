target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s

; Most of atomics lost information about the sign of the integer operand
; but since this concerns only built-ins  with two-complement's arithmetics
; it shouldn't cause any problems.


; Function Attrs: nounwind
define spir_kernel void @test_atomic_global(i32 addrspace(1)* %dst) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !5 !kernel_arg_type_qual !4 {
  ; atomic_inc
  %inc_ig = tail call spir_func i32 @_Z10atomic_incPU3AS1Vi(i32 addrspace(1)* %dst) #0
  ; CHECK: _Z10atomic_incPU3AS1Vi(i32 addrspace(1)* %dst) #0
  %dec_jg = tail call spir_func i32 @_Z10atomic_decPU3AS1Vj(i32 addrspace(1)* %dst) #0
  ; CHECK: _Z10atomic_decPU3AS1Vi(i32 addrspace(1)* %dst) #0

  ; atomic_max
  %max_ig = tail call spir_func i32 @_Z10atomic_maxPU3AS1Vii(i32 addrspace(1)* %dst, i32 0) #0
  ; CHECK: _Z10atomic_maxPU3AS1Vii(i32 addrspace(1)* %dst, i32 0) #0
  %max_jg = tail call spir_func i32 @_Z10atomic_maxPU3AS1Vjj(i32 addrspace(1)* %dst, i32 0) #0
  ; CHECK: _Z10atomic_maxPU3AS1Vjj(i32 addrspace(1)* %dst, i32 0) #0

  ; atomic_min
  %min_ig = tail call spir_func i32 @_Z10atomic_minPU3AS1Vii(i32 addrspace(1)* %dst, i32 0) #0
  ; CHECK: _Z10atomic_minPU3AS1Vii(i32 addrspace(1)* %dst, i32 0) #0
  %min_jg = tail call spir_func i32 @_Z10atomic_minPU3AS1Vjj(i32 addrspace(1)* %dst, i32 0) #0
  ; CHECK: _Z10atomic_minPU3AS1Vjj(i32 addrspace(1)* %dst, i32 0) #0

  ; atomic_add
  %add_ig = tail call spir_func i32 @_Z10atomic_addPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_addPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0
  %add_jg = tail call spir_func i32 @_Z10atomic_addPU3AS1Vjj(i32 addrspace(1)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_addPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0

  ; atomic_sub
  %sub_ig = tail call spir_func i32 @_Z10atomic_subPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_subPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0
  %sub_jg = tail call spir_func i32 @_Z10atomic_subPU3AS1Vjj(i32 addrspace(1)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_subPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0

  ; atomic_or
  %or_ig = tail call spir_func i32 @_Z9atomic_orPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0
  ; CHECK: _Z9atomic_orPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0
  %or_jg = tail call spir_func i32 @_Z9atomic_orPU3AS1Vjj(i32 addrspace(1)* %dst, i32 1) #0
  ; CHECK: _Z9atomic_orPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0

  ; atomic_xor
  %xor_ig = tail call spir_func i32 @_Z10atomic_xorPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_xorPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0
  %xor_jg = tail call spir_func i32 @_Z10atomic_xorPU3AS1Vjj(i32 addrspace(1)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_xorPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0

  ; atomic_and
  %and_ig = tail call spir_func i32 @_Z10atomic_andPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_andPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0
  %and_jg = tail call spir_func i32 @_Z10atomic_andPU3AS1Vjj(i32 addrspace(1)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_andPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0

  ; atomic_cmpxchg
  %cmpxchg_ig = call spir_func i32 @_Z14atomic_cmpxchgPU3AS1Viii(i32 addrspace(1)* %dst, i32 0, i32 1) #0
  ; CHECK: _Z14atomic_cmpxchgPU3AS1Viii(i32 addrspace(1)* %dst, i32 0, i32 1) #0
  %cmpxchg_jg = call spir_func i32 @_Z14atomic_cmpxchgPU3AS1Vjjj(i32 addrspace(1)* %dst, i32 0, i32 1) #0
  ; CHECK: _Z14atomic_cmpxchgPU3AS1Viii(i32 addrspace(1)* %dst, i32 0, i32 1) #0
  
  ; atomic_xchg
  %xchg_ig = call spir_func i32 @_Z11atomic_xchgPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0
  ; CHECK: _Z11atomic_xchgPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0
  %xchg_jg = call spir_func i32 @_Z11atomic_xchgPU3AS1Vjj(i32 addrspace(1)* %dst, i32 1) #0
  ; CHECK: _Z11atomic_xchgPU3AS1Vii(i32 addrspace(1)* %dst, i32 1) #0
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @test_atomic_local(i32 addrspace(3)* %dst) #0 !kernel_arg_addr_space !11 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !5 !kernel_arg_type_qual !4 {
  ; atomic_inc
  %inc_il = tail call spir_func i32 @_Z10atomic_incPU3AS3Vi(i32 addrspace(3)* %dst) #0
  ; CHECK: _Z10atomic_incPU3AS3Vi(i32 addrspace(3)* %dst) #0

  ; atomic dec
  %dec_jl = tail call spir_func i32 @_Z10atomic_decPU3AS3Vj(i32 addrspace(3)* %dst) #0
  ; CHECK: _Z10atomic_decPU3AS3Vi(i32 addrspace(3)* %dst) #0

  ; atomic_max
  %max_il = tail call spir_func i32 @_Z10atomic_maxPU3AS3Vii(i32 addrspace(3)* %dst, i32 0) #0
  ; CHECK: _Z10atomic_maxPU3AS3Vii(i32 addrspace(3)* %dst, i32 0) #0
  %max_jl = tail call spir_func i32 @_Z10atomic_maxPU3AS3jVj(i32 addrspace(3)* %dst, i32 0) #0
  ; CHECK: _Z10atomic_maxPU3AS3Vjj(i32 addrspace(3)* %dst, i32 0) #0

  ; atomic_min
  %min_il = tail call spir_func i32 @_Z10atomic_minPU3AS3Vii(i32 addrspace(3)* %dst, i32 0) #0
  ; CHECK: _Z10atomic_minPU3AS3Vii(i32 addrspace(3)* %dst, i32 0) #0
  %min_jl = tail call spir_func i32 @_Z10atomic_minPU3AS3jVj(i32 addrspace(3)* %dst, i32 0) #0
  ; CHECK: _Z10atomic_minPU3AS3Vjj(i32 addrspace(3)* %dst, i32 0) #0

  ; atomic_add
  %add_il = tail call spir_func i32 @_Z10atomic_addPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_addPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0
  %add_jl = tail call spir_func i32 @_Z10atomic_addPU3AS3jVj(i32 addrspace(3)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_addPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0

  ; atomic_sub
  %sub_il = tail call spir_func i32 @_Z10atomic_subPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_subPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0
  %sub_jl = tail call spir_func i32 @_Z10atomic_subPU3AS3jVj(i32 addrspace(3)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_subPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0

  ; atomic_or
  %or_il = tail call spir_func i32 @_Z9atomic_orPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0
  ; CHECK: _Z9atomic_orPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0
  %or_jl = tail call spir_func i32 @_Z9atomic_orPU3AS3jVj(i32 addrspace(3)* %dst, i32 1) #0
  ; CHECK: _Z9atomic_orPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0

  ; atomic_xor
  %xor_il = tail call spir_func i32 @_Z10atomic_xorPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_xorPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0
  %xor_jl = tail call spir_func i32 @_Z10atomic_xorPU3AS3jVj(i32 addrspace(3)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_xorPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0

  ; atomic_and
  %and_il = tail call spir_func i32 @_Z10atomic_andPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_andPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0
  %and_jl = tail call spir_func i32 @_Z10atomic_andPU3AS3jVj(i32 addrspace(3)* %dst, i32 1) #0
  ; CHECK: _Z10atomic_andPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0

  ; atomic_cmpxchg
  %cmpxchg_il = call spir_func i32 @_Z14atomic_cmpxchgPU3AS3Viii(i32 addrspace(3)* %dst, i32 0, i32 1) #0
  ; CHECK: _Z14atomic_cmpxchgPU3AS3Viii(i32 addrspace(3)* %dst, i32 0, i32 1) #0
  %cmpxchg_jl = call spir_func i32 @_Z14atomic_cmpxchgPU3AS3jVjj(i32 addrspace(3)* %dst, i32 0, i32 1) #0
  ; CHECK: _Z14atomic_cmpxchgPU3AS3Viii(i32 addrspace(3)* %dst, i32 0, i32 1) #0

  ; atomic_xchg
  %xchg_il = call spir_func i32 @_Z11atomic_xchgPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0
  ; CHECK: _Z11atomic_xchgPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0
  %xchg_jl = call spir_func i32 @_Z11atomic_xchgPU3AS3jVj(i32 addrspace(3)* %dst, i32 1) #0
  ; CHECK: _Z11atomic_xchgPU3AS3Vii(i32 addrspace(3)* %dst, i32 1) #0

  ret void
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z10atomic_incPU3AS1Vi(i32 addrspace(1)*)
declare spir_func i32 @_Z10atomic_decPU3AS1Vj(i32 addrspace(1)*)
declare spir_func i32 @_Z10atomic_maxPU3AS1Vii(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z10atomic_maxPU3AS1Vjj(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z10atomic_minPU3AS1Vii(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z10atomic_minPU3AS1Vjj(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z10atomic_addPU3AS1Vii(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z10atomic_addPU3AS1Vjj(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z10atomic_subPU3AS1Vii(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z10atomic_subPU3AS1Vjj(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z9atomic_orPU3AS1Vii(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z9atomic_orPU3AS1Vjj(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z10atomic_xorPU3AS1Vii(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z10atomic_xorPU3AS1Vjj(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z10atomic_andPU3AS1Vii(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z10atomic_andPU3AS1Vjj(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z14atomic_cmpxchgPU3AS1Viii(i32 addrspace(1)*, i32, i32)
declare spir_func i32 @_Z14atomic_cmpxchgPU3AS1Vjjj(i32 addrspace(1)*, i32, i32)
declare spir_func i32 @_Z11atomic_xchgPU3AS1Vii(i32 addrspace(1)*, i32)
declare spir_func i32 @_Z11atomic_xchgPU3AS1Vjj(i32 addrspace(1)*, i32)

declare spir_func i32 @_Z10atomic_incPU3AS3Vi(i32 addrspace(3)*)
declare spir_func i32 @_Z10atomic_decPU3AS3Vj(i32 addrspace(3)*)
declare spir_func i32 @_Z10atomic_maxPU3AS3Vii(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z10atomic_maxPU3AS3jVj(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z10atomic_minPU3AS3Vii(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z10atomic_minPU3AS3jVj(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z10atomic_addPU3AS3Vii(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z10atomic_addPU3AS3jVj(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z10atomic_subPU3AS3Vii(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z10atomic_subPU3AS3jVj(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z9atomic_orPU3AS3Vii(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z9atomic_orPU3AS3jVj(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z10atomic_xorPU3AS3Vii(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z10atomic_xorPU3AS3jVj(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z10atomic_andPU3AS3Vii(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z10atomic_andPU3AS3jVj(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z14atomic_cmpxchgPU3AS3Viii(i32 addrspace(3)*, i32, i32)
declare spir_func i32 @_Z14atomic_cmpxchgPU3AS3jVjj(i32 addrspace(3)*, i32, i32)
declare spir_func i32 @_Z11atomic_xchgPU3AS3Vii(i32 addrspace(3)*, i32)
declare spir_func i32 @_Z11atomic_xchgPU3AS3jVj(i32 addrspace(3)*, i32)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!7}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!9}

!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"int*"}
!4 = !{!"volatile"}
!5 = !{!"int*"}
!7 = !{i32 1, i32 2}
!8 = !{}
!9 = !{!"-cl-kernel-arg-info"}
!11 = !{i32 1}
