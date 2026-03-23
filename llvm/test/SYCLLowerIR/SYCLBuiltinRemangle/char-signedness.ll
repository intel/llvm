; Test for -sycl-remangle-char-is-signed option
;
; This tests the CharIsSigned behavior (PRIMITIVE_CHAR vs PRIMITIVE_UCHAR)
; when remangling functions with 'char' type parameters.
;
; In SYCL:
;   - 'c' = char (signedness is platform-dependent)
;
; In OpenCL C:
;   - 'c' = char (always signed)
;   - 'h' = unsigned char
;
; The -sycl-remangle-char-is-signed flag controls interpretation:
;   - When true (default): SYCL's char is signed -> OpenCL's char ('c')
;   - When false: SYCL's char is unsigned -> OpenCL's unsigned char ('h')
;
; Test with default (CharIsSigned=true): char stays as char
; RUN: opt -passes=sycl-builtin-remangle -sycl-remangle-spirv-target -sycl-remangle-char-is-signed=true -mtriple=spir64-unknown-unknown -S < %s | FileCheck %s --check-prefix=CHECK-SIGNED
;
; Test with CharIsSigned=false: char becomes unsigned char
; RUN: opt -passes=sycl-builtin-remangle -sycl-remangle-spirv-target -sycl-remangle-char-is-signed=false -mtriple=spir64-unknown-unknown -S < %s | FileCheck %s --check-prefix=CHECK-UNSIGNED

;===------------------------------------------------------------------------===
; Test scalar char parameter
;===------------------------------------------------------------------------===

; Input: __spirv_ocl_clz(char)
; SYCL mangling: _Z15__spirv_ocl_clzc (char)
;
; Expected output when CharIsSigned=true:
;   OpenCL: _Z15__spirv_ocl_clzc (char - signed)
;
; Expected output when CharIsSigned=false:
;   OpenCL: _Z15__spirv_ocl_clzh (unsigned char)

declare spir_func i8 @_Z15__spirv_ocl_clzc(i8)

; CHECK-SIGNED: declare spir_func i8 @_Z15__spirv_ocl_clzc(i8)
; CHECK-UNSIGNED: declare spir_func i8 @_Z15__spirv_ocl_clzh(i8)

;===------------------------------------------------------------------------===
; Test vector of char
;===------------------------------------------------------------------------===

; Input: __spirv_ocl_popcount(vector<char, 4>)
; SYCL mangling: _Z20__spirv_ocl_popcountDv4_c
;
; Expected when CharIsSigned=true:
;   OpenCL: _Z20__spirv_ocl_popcountDv4_c (vector<char, 4> - signed)
;
; Expected when CharIsSigned=false:
;   OpenCL: _Z20__spirv_ocl_popcountDv4_h (vector<unsigned char, 4>)

declare spir_func <4 x i8> @_Z20__spirv_ocl_popcountDv4_c(<4 x i8>)

; CHECK-SIGNED: declare spir_func <4 x i8> @_Z20__spirv_ocl_popcountDv4_c(<4 x i8>)
; CHECK-UNSIGNED: declare spir_func <4 x i8> @_Z20__spirv_ocl_popcountDv4_h(<4 x i8>)

;===------------------------------------------------------------------------===
; Test char parameter with substitutions
;===------------------------------------------------------------------------===

; Input: __spirv_ocl_rotate(char, char)
; SYCL mangling: _Z18__spirv_ocl_rotatecccc (uses substitution for second char)
;
; Expected when CharIsSigned=true:
;   OpenCL: _Z18__spirv_ocl_rotatecc (char)
;
; Expected when CharIsSigned=false:
;   OpenCL: _Z18__spirv_ocl_rotatehh (unsigned char)

declare spir_func i8 @_Z18__spirv_ocl_rotatecc(i8, i8)

; CHECK-SIGNED: declare spir_func i8 @_Z18__spirv_ocl_rotatecc(i8, i8)
; CHECK-UNSIGNED: declare spir_func i8 @_Z18__spirv_ocl_rotatehh(i8, i8)

;===------------------------------------------------------------------------===
; Test multiple char parameters
;===------------------------------------------------------------------------===

; Input: __spirv_ocl_bitselect(char, char, char)
; SYCL mangling: _Z21__spirv_ocl_bitselectccc (three char parameters)
;
; Expected when CharIsSigned=true:
;   OpenCL: _Z21__spirv_ocl_bitselectccc (char)
;
; Expected when CharIsSigned=false:
;   OpenCL: _Z21__spirv_ocl_bitselecthhh (unsigned char)

declare spir_func i8 @_Z21__spirv_ocl_bitselectccc(i8, i8, i8)

; CHECK-SIGNED: declare spir_func i8 @_Z21__spirv_ocl_bitselectccc(i8, i8, i8)
; CHECK-UNSIGNED: declare spir_func i8 @_Z21__spirv_ocl_bitselecthhh(i8, i8, i8)

;===------------------------------------------------------------------------===
; Test vector with substitution preserved
;===------------------------------------------------------------------------===

; Input: __spirv_ocl_bitselect(vector<char, 16>, vector<char, 16>, vector<char, 16>)
; SYCL mangling: _Z21__spirv_ocl_bitselectDv16_cS_S_ (uses substitutions S_ for repeated vectors)
;
; Expected when CharIsSigned=true:
;   OpenCL: _Z21__spirv_ocl_bitselectDv16_cS_S_ (char)
;
; Expected when CharIsSigned=false:
;   OpenCL: _Z21__spirv_ocl_bitselectDv16_hS_S_ (unsigned char)

declare spir_func <16 x i8> @_Z21__spirv_ocl_bitselectDv16_cS_S_(<16 x i8>, <16 x i8>, <16 x i8>)

; CHECK-SIGNED: declare spir_func <16 x i8> @_Z21__spirv_ocl_bitselectDv16_cS_S_(<16 x i8>, <16 x i8>, <16 x i8>)
; CHECK-UNSIGNED: declare spir_func <16 x i8> @_Z21__spirv_ocl_bitselectDv16_hS_S_(<16 x i8>, <16 x i8>, <16 x i8>)

;===------------------------------------------------------------------------===
; Test function calls to ensure the transformation works correctly
;===------------------------------------------------------------------------===

define spir_func void @test_char_signedness() {
  %1 = call spir_func i8 @_Z15__spirv_ocl_clzc(i8 127)
  %2 = call spir_func <4 x i8> @_Z20__spirv_ocl_popcountDv4_c(<4 x i8> zeroinitializer)
  %3 = call spir_func i8 @_Z18__spirv_ocl_rotatecc(i8 1, i8 2)
  %4 = call spir_func i8 @_Z21__spirv_ocl_bitselectccc(i8 1, i8 2, i8 3)
  %5 = call spir_func <16 x i8> @_Z21__spirv_ocl_bitselectDv16_cS_S_(<16 x i8> zeroinitializer, <16 x i8> zeroinitializer, <16 x i8> zeroinitializer)
  ret void
}
