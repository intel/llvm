; RUN: opt -passes=sycl-builtin-remangle -mtriple=nvptx64-nvidia-cuda -S < %s | FileCheck %s

; Test template parameter transformation for various types
; Focuses on _Float16 -> half, long long -> long, and signed char -> char in templates

;===------------------------------------------------------------------------===;
; Template with _Float16 parameter
; __spirv_ImageRead<_Float16>(unsigned long, float)
;===------------------------------------------------------------------------===

declare spir_func half @_Z17__spirv_ImageReadIDF16_mfET_T0_T1_(i64, float)
; CHECK: declare spir_func half @_Z17__spirv_ImageReadIDhmfET_T0_T1_(i64, float)

; Template with vector<4, _Float16> parameter
; __spirv_ImageRead<vector<4, _Float16>>(unsigned long, float)
declare spir_func <4 x half> @_Z17__spirv_ImageReadIDv4_DF16_mfET_T0_T1_(i64, float)
; CHECK: declare spir_func <4 x half> @_Z17__spirv_ImageReadIDv4_DhmfET_T0_T1_(i64, float)

; Template with vector<2, _Float16> parameter
; __spirv_ImageRead<vector<2, _Float16>>(unsigned long, float)
declare spir_func <2 x half> @_Z17__spirv_ImageReadIDv2_DF16_mfET_T0_T1_(i64, float)
; CHECK: declare spir_func <2 x half> @_Z17__spirv_ImageReadIDv2_DhmfET_T0_T1_(i64, float)

; Template with int parameter (should stay as-is)
; __spirv_ImageRead<int>(unsigned long, float)
declare spir_func i32 @_Z17__spirv_ImageReadIimfET_T0_T1_(i64, float)
; CHECK: declare spir_func i32 @_Z17__spirv_ImageReadIimfET_T0_T1_(i64, float)

; Template with float parameter
; __spirv_ImageRead<float>(unsigned long, float)
declare spir_func float @_Z17__spirv_ImageReadIfmfET_T0_T1_(i64, float)
; CHECK: declare spir_func float @_Z17__spirv_ImageReadIfmfET_T0_T1_(i64, float)

; Template with vector<4, float> parameter
; __spirv_ImageRead<vector<4, float>>(unsigned long, float)
declare spir_func <4 x float> @_Z17__spirv_ImageReadIDv4_fmfET_T0_T1_(i64, float)
; CHECK: declare spir_func <4 x float> @_Z17__spirv_ImageReadIDv4_fmfET_T0_T1_(i64, float)

;===------------------------------------------------------------------------===;
; Template with signed char parameter
; signed char -> char transformation
;===------------------------------------------------------------------------===

; Template with signed char parameter
; __spirv_ImageRead<signed char>(unsigned long, float)
; After fixing libspirv.bc: transform 'a' -> 'c' for ALL functions
declare spir_func i8 @_Z17__spirv_ImageReadIamfET_T0_T1_(i64, float)
; CHECK: declare spir_func i8 @_Z17__spirv_ImageReadIcmfET_T0_T1_(i64, float)

; Template with vector<4, signed char> parameter
; __spirv_ImageRead<vector<4, signed char>>(unsigned long, float)
; After fixing libspirv.bc: transform 'a' -> 'c'
declare spir_func <4 x i8> @_Z17__spirv_ImageReadIDv4_amfET_T0_T1_(i64, float)
; CHECK: declare spir_func <4 x i8> @_Z17__spirv_ImageReadIDv4_cmfET_T0_T1_(i64, float)

; Template with unsigned char parameter (should stay as unsigned char)
; __spirv_ImageRead<unsigned char>(unsigned long, float)
declare spir_func i8 @_Z17__spirv_ImageReadIhmfET_T0_T1_(i64, float)
; CHECK: declare spir_func i8 @_Z17__spirv_ImageReadIhmfET_T0_T1_(i64, float)

; Test scalar signed char transformation (from actual e2e failure)
; __spirv_ImageArrayWrite<ulong, int, signed char>
declare void @_Z23__spirv_ImageArrayWriteImiaEvT_T0_iT1_(i64, i32, i32, i8)
; CHECK: declare void @_Z23__spirv_ImageArrayWriteImicEvT_T0_iT1_(i64, i32, i32, i8)

;===------------------------------------------------------------------------===;
; Complex template with multiple parameters and function name containing 'E'
; Tests that we don't corrupt symbols with 'E' in the function name
; __spirv_ImageSampleExplicitLod<__spirv_SampledImage__image1d_ro, vec4<uint>, float>
;===------------------------------------------------------------------------===

; This tests the fix for template parsing where function names like "ExplicitLod"
; contain the character 'E' which could be confused with template delimiters.
; The pass should NOT transform this (no type transformations needed) and NOT corrupt it.
declare spir_func <4 x i32> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_jfET0_T_T1_if(i64, i32, float, i32, float)
; CHECK: declare spir_func <4 x i32> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_jfET0_T_T1_if(i64, i32, float, i32, float)

; Same function but with _Float16 in template args -> should transform DF16_ to Dh
; __spirv_ImageSampleExplicitLod<__spirv_SampledImage__image1d_ro, vec4<_Float16>, float>
declare spir_func <4 x half> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_DF16_fET0_T_T1_if(i64, i32, float, i32, float)
; CHECK: declare spir_func <4 x half> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_DhfET0_T_T1_if(i64, i32, float, i32, float)

; Same function but with signed char in template args -> should transform 'a' to 'c'
; __spirv_ImageSampleExplicitLod<__spirv_SampledImage__image1d_ro, vec4<signed char>, float>
declare spir_func <4 x i8> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_afET0_T_T1_if(i64, i32, float, i32, float)
; CHECK: declare spir_func <4 x i8> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_cfET0_T_T1_if(i64, i32, float, i32, float)

;===------------------------------------------------------------------------===;
; Template substitution expansion tests
; Tests that substitution references (S_, S0_, S1_) are correctly expanded
;===------------------------------------------------------------------------===;

; Test substitution preservation in array functions
; libspirv.bc now correctly generates S0_ for matching substitutable types
; __spirv_ImageArrayWrite<ulong, vec2<int>, vec2<int>>
; Template args: m (primitive, not substitutable), Dv2_i (substitutable), S0_ (reference to Dv2_i)
; The pass should NOT expand S0_ - libspirv.bc has S0_ in the symbol
declare void @_Z23__spirv_ImageArrayWriteImDv2_iS0_EvT_T0_iT1_(i64, <2 x i32>, i32, <2 x i32>)
; CHECK: declare void @_Z23__spirv_ImageArrayWriteImDv2_iS0_EvT_T0_iT1_(i64, <2 x i32>, i32, <2 x i32>)

; Test substitution preservation with SampledImageArrayFetch
; __spirv_SampledImageArrayFetch<vec2<int>, ulong, vec2<int>>
; Template args: Dv2_i (substitutable), m (primitive), S0_ (reference to Dv2_i)
; The pass should NOT expand S0_ - libspirv.bc has S0_ in the symbol
declare <2 x i32> @_Z30__spirv_SampledImageArrayFetchIDv2_imS0_ET_T0_T1_i(i64, <2 x i32>, i32)
; CHECK: declare <2 x i32> @_Z30__spirv_SampledImageArrayFetchIDv2_imS0_ET_T0_T1_i(i64, <2 x i32>, i32)

; Test substitution preservation for non-array functions
; __spirv_ImageFetch<vec2<int>, ulong, vec2<int>>
; Non-array functions also use substitutions in libspirv.bc
declare <2 x i32> @_Z18__spirv_ImageFetchIDv2_imS0_ET_T0_T1_(i64, <2 x i32>)
; CHECK: declare <2 x i32> @_Z18__spirv_ImageFetchIDv2_imS0_ET_T0_T1_(i64, <2 x i32>)

; Test substitution preservation with vec4
; __spirv_ImageFetch<vec4<float>, ulong, vec4<float>>
declare <4 x float> @_Z18__spirv_ImageFetchIDv4_fmS0_ET_T0_T1_(i64, <4 x float>)
; CHECK: declare <4 x float> @_Z18__spirv_ImageFetchIDv4_fmS0_ET_T0_T1_(i64, <4 x float>)

define spir_func void @test() {
  %1 = call spir_func half @_Z17__spirv_ImageReadIDF16_mfET_T0_T1_(i64 0, float 0.0)
  %2 = call spir_func <4 x half> @_Z17__spirv_ImageReadIDv4_DF16_mfET_T0_T1_(i64 0, float 0.0)
  %3 = call spir_func <2 x half> @_Z17__spirv_ImageReadIDv2_DF16_mfET_T0_T1_(i64 0, float 0.0)
  %4 = call spir_func i32 @_Z17__spirv_ImageReadIimfET_T0_T1_(i64 0, float 0.0)
  %5 = call spir_func float @_Z17__spirv_ImageReadIfmfET_T0_T1_(i64 0, float 0.0)
  %6 = call spir_func <4 x float> @_Z17__spirv_ImageReadIDv4_fmfET_T0_T1_(i64 0, float 0.0)
  %7 = call spir_func i8 @_Z17__spirv_ImageReadIamfET_T0_T1_(i64 0, float 0.0)
  %8 = call spir_func <4 x i8> @_Z17__spirv_ImageReadIDv4_amfET_T0_T1_(i64 0, float 0.0)
  %9 = call spir_func i8 @_Z17__spirv_ImageReadIhmfET_T0_T1_(i64 0, float 0.0)
  call void @_Z23__spirv_ImageArrayWriteImiaEvT_T0_iT1_(i64 0, i32 0, i32 0, i8 0)
  %10 = call spir_func <4 x i32> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_jfET0_T_T1_if(i64 0, i32 0, float 0.0, i32 2, float 0.0)
  %11 = call spir_func <4 x half> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_DF16_fET0_T_T1_if(i64 0, i32 0, float 0.0, i32 2, float 0.0)
  %12 = call spir_func <4 x i8> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_afET0_T_T1_if(i64 0, i32 0, float 0.0, i32 2, float 0.0)

  ; Substitution tests - array functions expand, non-array preserve
  call void @_Z23__spirv_ImageArrayWriteImDv2_iS0_EvT_T0_iT1_(i64 0, <2 x i32> zeroinitializer, i32 0, <2 x i32> zeroinitializer)
  %13 = call <2 x i32> @_Z30__spirv_SampledImageArrayFetchIDv2_imS0_ET_T0_T1_i(i64 0, <2 x i32> zeroinitializer, i32 0)
  %14 = call <2 x i32> @_Z18__spirv_ImageFetchIDv2_imS0_ET_T0_T1_(i64 0, <2 x i32> zeroinitializer)
  %15 = call <4 x float> @_Z18__spirv_ImageFetchIDv4_fmS0_ET_T0_T1_(i64 0, <4 x float> zeroinitializer)

  ret void
}
