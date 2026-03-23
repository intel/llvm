; RUN: opt -passes=sycl-builtin-remangle -sycl-remangle-spirv-target -mtriple=spir64-unknown-unknown -S < %s | FileCheck %s

; Test signed char type remangling

; Test: signed char parameter
; SYCL: _Z17__spirv_ocl_s_absa (signed char)
; OpenCL: _Z17__spirv_ocl_s_absc (char)

declare spir_func i8 @_Z17__spirv_ocl_s_absa(i8)
; CHECK: declare spir_func i8 @_Z17__spirv_ocl_s_absc(i8)

; Test: char parameter (should stay as char)
; SYCL: _Z17__spirv_ocl_s_absc (char)
; OpenCL: _Z17__spirv_ocl_s_absc (char)
; Note: This declaration gets merged with the previous _absc declaration

declare spir_func i8 @_Z17__spirv_ocl_s_absc(i8)

; Test: unsigned char parameter (should stay as uchar)
; SYCL: _Z17__spirv_ocl_u_absh (unsigned char)
; OpenCL: _Z17__spirv_ocl_u_absh (unsigned char)

declare spir_func i8 @_Z17__spirv_ocl_u_absh(i8)
; CHECK: declare spir_func i8 @_Z17__spirv_ocl_u_absh(i8)

;===------------------------------------------------------------------------===
; Vector of signed char
; Real builtin: __spirv_ocl_s_mul_hi (length 20)
;===------------------------------------------------------------------------===

; Test: vec4 signed char (substitutions preserved after transformation)
declare spir_func <4 x i8> @_Z20__spirv_ocl_s_mul_hiDv4_aS_(<4 x i8>, <4 x i8>)
; CHECK: declare spir_func <4 x i8> @_Z20__spirv_ocl_s_mul_hiDv4_cS_(<4 x i8>, <4 x i8>)

; Test: vec16 signed char (substitutions preserved after transformation)
declare spir_func <16 x i8> @_Z20__spirv_ocl_s_mul_hiDv16_aS_(<16 x i8>, <16 x i8>)
; CHECK: declare spir_func <16 x i8> @_Z20__spirv_ocl_s_mul_hiDv16_cS_(<16 x i8>, <16 x i8>)

; Test: vector of signed char (alternate builtin for variety)
; SYCL: _Z17__spirv_ocl_s_absDv16_a (vector<signed char, 16>)
; OpenCL: _Z17__spirv_ocl_s_absDv16_c (vector<char, 16>)

declare spir_func <16 x i8> @_Z17__spirv_ocl_s_absDv16_a(<16 x i8>)
; CHECK: declare spir_func <16 x i8> @_Z17__spirv_ocl_s_absDv16_c(<16 x i8>)


define spir_func void @test() {
  ; scalar
  %1 = call spir_func i8 @_Z17__spirv_ocl_s_absa(i8 -1)
  %2 = call spir_func i8 @_Z17__spirv_ocl_s_absc(i8 1)
  %3 = call spir_func i8 @_Z17__spirv_ocl_u_absh(i8 1)

  ; vector
  %4 = call spir_func <4 x i8> @_Z20__spirv_ocl_s_mul_hiDv4_aS_(<4 x i8> zeroinitializer, <4 x i8> zeroinitializer)
  %5 = call spir_func <16 x i8> @_Z20__spirv_ocl_s_mul_hiDv16_aS_(<16 x i8> zeroinitializer, <16 x i8> zeroinitializer)
  %6 = call spir_func <16 x i8> @_Z17__spirv_ocl_s_absDv16_a(<16 x i8> zeroinitializer)
  ret void
}
