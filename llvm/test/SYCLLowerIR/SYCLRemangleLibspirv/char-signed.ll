; RUN: opt -passes=sycl-remangle-libspirv --remangle-long-width=64 --remangle-char-signedness=signed -mtriple=nvptx64-nvidia-cuda -S < %s | FileCheck %s

; Test checks remangling when char is signed.

; signed char parameter.
define i8 @_Z17__spirv_ocl_s_absa(i8) { unreachable }
; CHECK-DAG: define i8 @_Z17__spirv_ocl_s_absa(

; char parameter (should stay as char).
define i8 @_Z17__spirv_ocl_s_absc(i8) { unreachable }
; CHECK-DAG: define i8 @_Z17__spirv_ocl_s_absc(

; unsigned char parameter (should stay as uchar).
define i8 @_Z17__spirv_ocl_u_absh(i8) { unreachable }
; CHECK-DAG: define i8 @_Z17__spirv_ocl_u_absh(

; vec4 signed char (substitutions preserved after transformation).
define <4 x i8> @_Z20__spirv_ocl_s_mul_hiDv4_aS_(<4 x i8>, <4 x i8>) { unreachable }
; CHECK-DAG: define <4 x i8> @_Z20__spirv_ocl_s_mul_hiDv4_aS_(

; vec16 signed char (substitutions preserved after transformation).
define <16 x i8> @_Z20__spirv_ocl_s_mul_hiDv16_aS_(<16 x i8>, <16 x i8>) { unreachable }
; CHECK-DAG: define <16 x i8> @_Z20__spirv_ocl_s_mul_hiDv16_aS_(

; vector of signed char (alternate builtin for variety).
define <16 x i8> @_Z17__spirv_ocl_s_absDv16_a(<16 x i8>) { unreachable }
; CHECK-DAG: define <16 x i8> @_Z17__spirv_ocl_s_absDv16_a(
