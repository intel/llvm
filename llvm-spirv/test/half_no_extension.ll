;__kernel void test( __global float4 *p, __global half *f )
;{
;   __private float4 data;
;   data = p[0];
;   vstorea_half4_rtp( data, 0, f );
;}

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; CHECK-SPIRV: {{[0-9]+}} Capability Float16Buffer
; CHECK-SPIRV-NOT: {{[0-9]+}} Capability Float16

; ModuleID = 'main'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test(ptr addrspace(1) %p, ptr addrspace(1) %f) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %p.addr = alloca ptr addrspace(1), align 8
  %f.addr = alloca ptr addrspace(1), align 8
  %data = alloca <4 x float>, align 16
  store ptr addrspace(1) %p, ptr %p.addr, align 8
  store ptr addrspace(1) %f, ptr %f.addr, align 8
  %0 = load ptr addrspace(1), ptr %p.addr, align 8
  %1 = load <4 x float>, ptr addrspace(1) %0, align 16
  store <4 x float> %1, ptr %data, align 16
  %2 = load <4 x float>, ptr %data, align 16
  %3 = load ptr addrspace(1), ptr %f.addr, align 8
  call spir_func void @_Z17vstorea_half4_rtpDv4_fmPU3AS1Dh(<4 x float> %2, i64 0, ptr addrspace(1) %3)
  ret void
}

declare spir_func void @_Z17vstorea_half4_rtpDv4_fmPU3AS1Dh(<4 x float>, i64, ptr addrspace(1)) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!7}
!opencl.compiler.options = !{!7}
!llvm.ident = !{!8}

!1 = !{i32 1, i32 1}
!2 = !{!"none", !"none"}
!3 = !{!"float4*", !"half*"}
!4 = !{!"float4*", !"half*"}
!5 = !{!"", !""}
!6 = !{i32 1, i32 2}
!7 = !{}
!8 = !{!"clang version 3.6.1"}
