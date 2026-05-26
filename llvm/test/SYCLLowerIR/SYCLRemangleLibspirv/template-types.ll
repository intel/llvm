; RUN: opt -passes=sycl-remangle-libspirv --remangle-long-width=64 --remangle-char-signedness=signed -mtriple=nvptx64-nvidia-cuda -S < %s | FileCheck %s

; Test template parameter transformation for various types.

;===------------------------------------------------------------------------===;
; Template with _Float16 parameter.
; __spirv_ImageRead<_Float16>(unsigned long, float)
;===------------------------------------------------------------------------===

define half @_Z17__spirv_ImageReadIDhmfET_T0_T1_(i64, float) { unreachable }
; CHECK-DAG: define half @_Z17__spirv_ImageReadIDF16_yfET_T0_T1_(
; CHECK-DAG: define half @_Z17__spirv_ImageReadIDF16_mfET_T0_T1_(

; Template with vector<4, _Float16> parameter.
; __spirv_ImageRead<vector<4, _Float16>>(unsigned long, float)
define <4 x half> @_Z17__spirv_ImageReadIDv4_DhmfET_T0_T1_(i64, float) { unreachable }
; CHECK-DAG: define <4 x half> @_Z17__spirv_ImageReadIDv4_DF16_yfET_T0_T1_(
; CHECK-DAG: define <4 x half> @_Z17__spirv_ImageReadIDv4_DF16_mfET_T0_T1_(

; Template with int parameter (should stay as-is).
; __spirv_ImageRead<int>(unsigned long, float)
define i32 @_Z17__spirv_ImageReadIimfET_T0_T1_(i64, float) { unreachable }
; CHECK-DAG: define i32 @_Z17__spirv_ImageReadIiyfET_T0_T1_(
; CHECK-DAG: define i32 @_Z17__spirv_ImageReadIimfET_T0_T1_(

; Template with float parameter.
; __spirv_ImageRead<float>(unsigned long, float)
define float @_Z17__spirv_ImageReadIfmfET_T0_T1_(i64, float) { unreachable }
; CHECK-DAG: define float @_Z17__spirv_ImageReadIfyfET_T0_T1_(
; CHECK-DAG: define float @_Z17__spirv_ImageReadIfmfET_T0_T1_(

; Template with vector<4, float> parameter.
; __spirv_ImageRead<vector<4, float>>(unsigned long, float)
define <4 x float> @_Z17__spirv_ImageReadIDv4_fmfET_T0_T1_(i64, float) { unreachable }
; CHECK-DAG: define <4 x float> @_Z17__spirv_ImageReadIDv4_fyfET_T0_T1_(
; CHECK-DAG: define <4 x float> @_Z17__spirv_ImageReadIDv4_fmfET_T0_T1_(

;===------------------------------------------------------------------------===;
; Template with char parameter.
;===------------------------------------------------------------------------===

; Template with char parameter.
; __spirv_ImageRead<char>(unsigned long, float)
define i8 @_Z17__spirv_ImageReadIcmfET_T0_T1_(i64, float) { unreachable }
; CHECK-DAG: define i8 @_Z17__spirv_ImageReadIayfET_T0_T1_(
; CHECK-DAG: define i8 @_Z17__spirv_ImageReadIcmfET_T0_T1_(
; CHECK-DAG: define i8 @_Z17__spirv_ImageReadIamfET_T0_T1_(

; Template with vector<4, char> parameter
; __spirv_ImageRead<vector<4, char>>(unsigned long, float)
define <4 x i8> @_Z17__spirv_ImageReadIDv4_cmfET_T0_T1_(i64, float) { unreachable }
; CHECK-DAG: define <4 x i8> @_Z17__spirv_ImageReadIDv4_ayfET_T0_T1_(
; CHECK-DAG: define <4 x i8> @_Z17__spirv_ImageReadIDv4_cmfET_T0_T1_(
; CHECK-DAG: define <4 x i8> @_Z17__spirv_ImageReadIDv4_amfET_T0_T1_(

; Template with unsigned char parameter (should stay as unsigned char).
; __spirv_ImageRead<unsigned char>(unsigned long, float)
define i8 @_Z17__spirv_ImageReadIhmfET_T0_T1_(i64, float) { unreachable }
; CHECK-DAG: define i8 @_Z17__spirv_ImageReadIhyfET_T0_T1_(
; CHECK-DAG: define i8 @_Z17__spirv_ImageReadIhmfET_T0_T1_(

; Test scalar char transformation.
; __spirv_ImageArrayWrite<ulong, int, char>
define void @_Z23__spirv_ImageArrayWriteImicEvT_T0_iT1_(i64, i32, i32, i8) { unreachable }
; CHECK-DAG: define void @_Z23__spirv_ImageArrayWriteIyiaEvT_T0_iT1_(

;===------------------------------------------------------------------------===;
; Complex template with multiple parameters and function name containing 'E'
; Test that we don't corrupt symbols with 'E' in the function name.
; __spirv_ImageSampleExplicitLod<__spirv_SampledImage__image1d_ro, vec4<uint>, float>
;===------------------------------------------------------------------------===

; Test the fix for template parsing where function names like "ExplicitLod"
; contain the character 'E' which could be confused with template delimiters.
; The pass should NOT transform this (no type transformations needed).
define <4 x i32> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_jfET0_T_T1_if(i64, i32, float, i32, float) { unreachable }
; CHECK-DAG: define <4 x i32> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_jfET0_T_T1_if(

; Same function but with half in template args.
; __spirv_ImageSampleExplicitLod<__spirv_SampledImage__image1d_ro, vec4<half>, float>
define <4 x half> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_DhfET0_T_T1_if(i64, i32, float, i32, float) { unreachable }
; CHECK-DAG: define <4 x half> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_DF16_fET0_T_T1_if(

;===------------------------------------------------------------------------===;
; Test that substitution references (S_, S0_, S1_) are correctly expanded.
;===------------------------------------------------------------------------===;

; Test substitution preservation in array functions.
; __spirv_ImageArrayWrite<ulong, vec2<int>, vec2<int>>
; Template args: m (primitive, not substitutable), Dv2_i (substitutable), S0_ (reference to Dv2_i)
; The pass should NOT expand S0_ - libspirv.bc has S0_ in the symbol.
define void @_Z23__spirv_ImageArrayWriteImDv2_iS0_EvT_T0_iT1_(i64, <2 x i32>, i32, <2 x i32>) { unreachable }
; CHECK-DAG: define void @_Z23__spirv_ImageArrayWriteIyDv2_iS0_EvT_T0_iT1_(
; CHECK-DAG: define void @_Z23__spirv_ImageArrayWriteImDv2_iS0_EvT_T0_iT1_(

; Test substitution preservation with SampledImageArrayFetch.
; __spirv_SampledImageArrayFetch<vec2<int>, ulong, vec2<int>>
; Template args: Dv2_i (substitutable), m (primitive), S0_ (reference to Dv2_i).
; The pass should NOT expand S0_ - libspirv.bc has S0_ in the symbol.
define <2 x i32> @_Z30__spirv_SampledImageArrayFetchIDv2_imS0_ET_T0_T1_i(i64, <2 x i32>, i32) { unreachable }
; CHECK-DAG: define <2 x i32> @_Z30__spirv_SampledImageArrayFetchIDv2_iyS0_ET_T0_T1_i(
; CHECK-DAG: define <2 x i32> @_Z30__spirv_SampledImageArrayFetchIDv2_imS0_ET_T0_T1_i(

; Test substitution preservation for non-array functions.
; __spirv_ImageFetch<vec2<int>, ulong, vec2<int>>
define <2 x i32> @_Z18__spirv_ImageFetchIDv2_imS0_ET_T0_T1_(i64, <2 x i32>) { unreachable }
; CHECK-DAG: define <2 x i32> @_Z18__spirv_ImageFetchIDv2_iyS0_ET_T0_T1_(
; CHECK-DAG: define <2 x i32> @_Z18__spirv_ImageFetchIDv2_imS0_ET_T0_T1_(
