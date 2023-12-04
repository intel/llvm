; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; TODO: Validation is disabled till the moment the tools in CI are updated (passes locally)
; R/UN: spirv-val %t.spv

; RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: TypeInt [[#TypeInt:]] 32 0
; CHECK-SPIRV: TypeCooperativeMatrixKHR [[#MatrixTypeInt:]] [[#TypeInt]]
; CHECK-SPIRV: TypeFloat [[#TypeFloat:]] 32
; CHECK-SPIRV: TypeCooperativeMatrixKHR [[#MatrixTypeFloat:]] [[#TypeFloat]]

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt]] [[#MatrixIn:]] [[#]] {{$}}
; CHECK-SPIRV: SNegate [[#MatrixTypeInt]] [[#]] [[#MatrixIn]]
; CHECK-LLVM: %1 = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructi(i32 0)
; CHECK-LLVM: %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z15__spirv_SNegatePU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %1)
define spir_kernel void @testSNegate(i32 %a) #0 !kernel_arg_addr_space !10 !kernel_arg_access_qual !11 !kernel_arg_type !12 !kernel_arg_type_qual !9 !kernel_arg_base_type !12 {
  %1 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 0)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z15__spirv_SNegate(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %1)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeFloat]] [[#MatrixIn:]] [[#]] {{$}}
; CHECK-SPIRV: FNegate [[#MatrixTypeFloat]] [[#]] [[#MatrixIn]]
; CHECK-LLVM: %0 = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructf(float 0.000000e+00)
; CHECK-LLVM: %call = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z15__spirv_FNegatePU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %0)
define spir_kernel void @testFNeg(float %a) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !9 {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float 0.000000e+00)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z15__spirv_FNegate(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %0)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt]] [[#MatrixA:]] [[#]] {{$}}
; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt]] [[#MatrixB:]] [[#]] {{$}}
; CHECK-SPIRV: IAdd [[#MatrixTypeInt]] [[#]] [[#MatrixA]] [[#MatrixB]]
; CHECK-LLVM: %1 = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructi(i32 0)
; CHECK-LLVM: %2 = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructi(i32 0)
; CHECK-LLVM: %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_IAddPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2S1_(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %1, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %2)
define spir_kernel void @testIAdd(i32 %a, i32 %b) #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_type_qual !7 !kernel_arg_base_type !6 {
  %1 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 0)
  %2 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 0)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_IAdd(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %1, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %2)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt]] [[#MatrixA:]] [[#]] {{$}}
; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt]] [[#MatrixB:]] [[#]] {{$}}
; CHECK-SPIRV: ISub [[#MatrixTypeInt]] [[#]] [[#MatrixA]] [[#MatrixB]]
; CHECK-LLVM: %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_ISubPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2S1_(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %1, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %2)
define spir_kernel void @testISub(i32 %a, i32 %b) #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_type_qual !7 !kernel_arg_base_type !6 {
  %1 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 0)
  %2 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 0)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_ISub(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %1, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %2)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt]] [[#MatrixA:]] [[#]] {{$}}
; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt]] [[#MatrixB:]] [[#]] {{$}}
; CHECK-SPIRV: IMul [[#MatrixTypeInt]] [[#]] [[#MatrixA]] [[#MatrixB]]
; CHECK-LLVM: %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_IMulPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2S1_(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %1, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %2)
define spir_kernel void @testIMul(i32 %a, i32 %b) #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_type_qual !7 !kernel_arg_base_type !6 {
  %1 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 0)
  %2 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 0)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_IMul(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %1, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %2)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt]] [[#MatrixA:]] [[#]] {{$}}
; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt]] [[#MatrixB:]] [[#]] {{$}}
; CHECK-SPIRV: SDiv [[#MatrixTypeInt]] [[#]] [[#MatrixA]] [[#MatrixB]]
; CHECK-LLVM: %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_SDivPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2S1_(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %1, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %2)
define void @testSDiv(i32 %a, i32 %b) {
  %1 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 0)
  %2 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 0)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_SDiv(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %1, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %2)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt]] [[#MatrixA:]] [[#]] {{$}}
; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt]] [[#MatrixB:]] [[#]] {{$}}
; CHECK-SPIRV: UDiv [[#MatrixTypeInt]] [[#]] [[#MatrixA]] [[#MatrixB]]
; CHECK-LLVM: %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_UDivPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2S1_(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %1, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %2)
define void @testUDiv(i32 %a, i32 %b) {
  %1 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 0)
  %2 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 0)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_UDiv(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %1, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %2)
  ret void
}


; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeFloat]] [[#MatrixA:]] [[#]] {{$}}
; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeFloat]] [[#MatrixB:]] [[#]] {{$}}
; CHECK-SPIRV: FAdd [[#MatrixTypeFloat]] [[#]] [[#MatrixA]] [[#MatrixB]]
; CHECK-LLVM: %0 = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructf(float 0.000000e+00)
; CHECK-LLVM: %1 = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructf(float 0.000000e+00)
; CHECK-LLVM: %call = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z12__spirv_FAddPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2S1_(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %0, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %1)
define spir_kernel void @testFAdd(float %a, float %b) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float 0.000000e+00)
  %1 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float 0.000000e+00)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z12__spirv_FAdd(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %0, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %1)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeFloat]] [[#MatrixA:]] [[#]] {{$}}
; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeFloat]] [[#MatrixB:]] [[#]] {{$}}
; CHECK-SPIRV: FSub [[#MatrixTypeFloat]] [[#]] [[#MatrixA]] [[#MatrixB]]
; CHECK-LLVM: %call = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z12__spirv_FSubPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2S1_(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %0, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %1)
define spir_kernel void @testFSub(float %a, float %b) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float 0.000000e+00)
  %1 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float 0.000000e+00)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z12__spirv_FSub(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %0, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %1)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeFloat]] [[#MatrixA:]] [[#]] {{$}}
; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeFloat]] [[#MatrixB:]] [[#]] {{$}}
; CHECK-SPIRV: FMul [[#MatrixTypeFloat]] [[#]] [[#MatrixA]] [[#MatrixB]]
; CHECK-LLVM: %call = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z12__spirv_FMulPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2S1_(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %0, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %1)
define spir_kernel void @testFMul(float %a, float %b) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float 0.000000e+00)
  %1 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float 0.000000e+00)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z12__spirv_FMul(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %0, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %1)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeFloat]] [[#MatrixA:]] [[#]] {{$}}
; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeFloat]] [[#MatrixB:]] [[#]] {{$}}
; CHECK-SPIRV: FDiv [[#MatrixTypeFloat]] [[#]] [[#MatrixA]] [[#MatrixB]]
; CHECK-LLVM: %call = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z12__spirv_FDivPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2S1_(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %0, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %1)
define spir_kernel void @testFDiv(float %a, float %b) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float 0.000000e+00)
  %1 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float 0.000000e+00)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z12__spirv_FDiv(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %0, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %1)
  ret void
}

declare spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float noundef)
declare spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z15__spirv_FNegate(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef)
declare spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z15__spirv_SNegate(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_IAdd(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef)
declare spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_ISub(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef)
declare spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_IMul(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef)
declare spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_SDiv(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef)
declare spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z12__spirv_UDiv(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z12__spirv_FAdd(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef)
declare spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z12__spirv_FSub(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef)
declare spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z12__spirv_FMul(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef)
declare spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z12__spirv_FDiv(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef)


attributes #0 = { nounwind }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!0}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!spirv.Generator = !{!3}

!0 = !{i32 1, i32 2}
!1 = !{i32 3, i32 102000}
!2 = !{}
!3 = !{i16 7, i16 0}
!4 = !{i32 0, i32 0}
!5 = !{!"none", !"none"}
!6 = !{!"int", !"int"}
!7 = !{!"", !""}
!8 = !{!2, !2}
!9 = !{!""}
!10 = !{i32 0}
!11 = !{!"none"}
!12 = !{!"int"}
