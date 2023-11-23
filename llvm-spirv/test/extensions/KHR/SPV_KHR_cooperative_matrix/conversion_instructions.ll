; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix -o %t.spv
; TODO: Validation is disabled till the moment the tools in CI are updated (passes locally)
; R/UN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc 
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: TypeInt [[#TypeInt32:]] 32 0
; CHECK-SPIRV: TypeInt [[#TypeInt16:]] 16 0
; CHECK-SPIRV: TypeInt [[#TypeInt8:]] 8 0
; CHECK-SPIRV: TypeFloat [[#TypeFloat:]] 32
; CHECK-SPIRV: TypeCooperativeMatrixKHR [[#MatrixTypeFloat:]] [[#TypeFloat]]
; CHECK-SPIRV: TypeCooperativeMatrixKHR [[#MatrixTypeInt32:]] [[#TypeInt32]]
; CHECK-SPIRV: TypeCooperativeMatrixKHR [[#MatrixTypeInt16:]] [[#TypeInt16]]
; CHECK-SPIRV: TypeFloat [[#TypeFloat16:]] 16
; CHECK-SPIRV: TypeCooperativeMatrixKHR [[#MatrixTypeFloat16:]] [[#TypeFloat16]]
; CHECK-SPIRV: TypeCooperativeMatrixKHR [[#MatrixTypeInt8:]] [[#TypeInt8]]

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeFloat]] [[#MatrixIn:]] [[#]] {{$}}
; CHECK-SPIRV: ConvertFToU [[#MatrixTypeInt32]] [[#]] [[#MatrixIn]]

; CHECK-LLVM: %[[#Matrix:]] = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructf(float 0.000000e+00)
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z77__spirv_ConvertFToU_RPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2_rtpPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %[[#Matrix]])

define void @convert_f_to_u() {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float 0.000000e+00)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z77__spirv_ConvertFToU_RPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2_rtpPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %0)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeFloat]] [[#MatrixIn:]] [[#]] {{$}}
; CHECK-SPIRV: ConvertFToS [[#MatrixTypeInt32]] [[#]] [[#MatrixIn]]

; CHECK-LLVM: %[[#Matrix:]] = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructf(float 0.000000e+00)
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z77__spirv_ConvertFToS_RPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2_rtpPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %[[#Matrix]])

define void @convert_f_to_s() {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float 0.000000e+00)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z77__spirv_ConvertFToS_RPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2_rtpPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %0)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt16]] [[#MatrixIn:]] [[#]] {{$}}
; CHECK-SPIRV: ConvertSToF [[#MatrixTypeFloat16]] [[#]] [[#MatrixIn]]

; CHECK-LLVM: %[[#Matrix:]] = call spir_func target("spirv.CooperativeMatrixKHR", i16, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructs(i16 0)
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) @_Z77__spirv_ConvertSToF_RPU3AS144__spirv_CooperativeMatrixKHR__half_3_12_12_2_rtpPU3AS145__spirv_CooperativeMatrixKHR__short_3_12_12_2(target("spirv.CooperativeMatrixKHR", i16, 3, 12, 12, 2) %[[#Matrix]])

define void @convert_s_to_f() {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt16(i16 0)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) @_Z77__spirv_ConvertSToF_RPU3AS144__spirv_CooperativeMatrixKHR__half_3_12_12_2_rtpPU3AS145__spirv_CooperativeMatrixKHR__short_3_12_12_2(target("spirv.CooperativeMatrixKHR", i16, 3, 12, 12, 2) %0)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt16]] [[#MatrixIn:]] [[#]] {{$}}
; CHECK-SPIRV: ConvertUToF [[#MatrixTypeFloat16]] [[#]] [[#MatrixIn]]

; CHECK-LLVM: %[[#Matrix:]] = call spir_func target("spirv.CooperativeMatrixKHR", i16, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructs(i16 0)
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) @_Z77__spirv_ConvertUToF_RPU3AS144__spirv_CooperativeMatrixKHR__half_3_12_12_2_rtpPU3AS145__spirv_CooperativeMatrixKHR__short_3_12_12_2(target("spirv.CooperativeMatrixKHR", i16, 3, 12, 12, 2) %[[#Matrix]])

define void @convert_u_to_f() {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt16(i16 0)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) @_Z77__spirv_ConvertUToF_RPU3AS144__spirv_CooperativeMatrixKHR__half_3_12_12_2_rtpPU3AS145__spirv_CooperativeMatrixKHR__short_3_12_12_2(target("spirv.CooperativeMatrixKHR", i16, 3, 12, 12, 2) %0)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt32]] [[#MatrixIn:]] [[#]] {{$}}
; CHECK-SPIRV: UConvert [[#MatrixTypeInt8]] [[#]] [[#MatrixIn]]

; CHECK-LLVM: %[[#Matrix:]] = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructi(i32 0)
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", i8, 3, 12, 12, 2) @_Z74__spirv_UConvert_RPU3AS144__spirv_CooperativeMatrixKHR__char_3_12_12_2_satPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %[[#Matrix]])

define void @u_convert() {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 0)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", i8, 3, 12, 12, 2) @_Z74__spirv_UConvert_RPU3AS144__spirv_CooperativeMatrixKHR__char_3_12_12_2_satPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %0)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeInt8]] [[#MatrixIn:]] [[#]] {{$}}
; CHECK-SPIRV: SConvert [[#MatrixTypeInt32]] [[#]] [[#MatrixIn]]

; CHECK-LLVM: %[[#Matrix:]] = call spir_func target("spirv.CooperativeMatrixKHR", i8, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructc(i8 0)
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z74__spirv_SConvert_RPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2_satPU3AS144__spirv_CooperativeMatrixKHR__char_3_12_12_2(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 12, 2) %[[#Matrix]])

define void @s_convert() {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt8(i8 0)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z74__spirv_SConvert_RPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2_satPU3AS144__spirv_CooperativeMatrixKHR__char_3_12_12_2(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 12, 2) %0)
  ret void
}

; CHECK-SPIRV: CompositeConstruct [[#MatrixTypeFloat16]] [[#MatrixIn:]] [[#]] {{$}}
; CHECK-SPIRV: FConvert [[#MatrixTypeFloat]] [[#]] [[#MatrixIn]]

; CHECK-LLVM: %[[#Matrix:]] = call spir_func target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructDh(half 0xH0000)
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z75__spirv_FConvert_RPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2_satPU3AS144__spirv_CooperativeMatrixKHR__half_3_12_12_2(target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) %[[#Matrix]])

define void @f_convert() {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructHalf(half 0xH0000)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z75__spirv_FConvert_RPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2_satPU3AS144__spirv_CooperativeMatrixKHR__half_3_12_12_2(target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) %0)
  ret void
}

declare spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructHalf(half noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt32(i32 noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt16(i16 noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructInt8(i8 noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z77__spirv_ConvertFToU_RPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2_rtpPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z77__spirv_ConvertFToS_RPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2_rtpPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) @_Z77__spirv_ConvertSToF_RPU3AS144__spirv_CooperativeMatrixKHR__half_3_12_12_2_rtpPU3AS145__spirv_CooperativeMatrixKHR__short_3_12_12_2(target("spirv.CooperativeMatrixKHR", i16, 3, 12, 12, 2) noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) @_Z77__spirv_ConvertUToF_RPU3AS144__spirv_CooperativeMatrixKHR__half_3_12_12_2_rtpPU3AS145__spirv_CooperativeMatrixKHR__short_3_12_12_2(target("spirv.CooperativeMatrixKHR", i16, 3, 12, 12, 2) noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 3, 12, 12, 2) @_Z74__spirv_UConvert_RPU3AS144__spirv_CooperativeMatrixKHR__char_3_12_12_2_satPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z74__spirv_SConvert_RPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2_satPU3AS144__spirv_CooperativeMatrixKHR__char_3_12_12_2(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 12, 2) noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z75__spirv_FConvert_RPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2_satPU3AS144__spirv_CooperativeMatrixKHR__half_3_12_12_2(target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) noundef)

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{!"clang version 16.0.0 (https://github.com/llvm/llvm-project.git 08d094a0e457360ad8b94b017d2dc277e697ca76)"}
