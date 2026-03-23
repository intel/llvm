; RUN: opt -passes=sycl-builtin-remangle -mtriple=nvptx64-nvidia-cuda -S < %s | FileCheck %s --check-prefix=CHECK-SPIR64
; RUN: opt -passes=sycl-builtin-remangle -mtriple=x86_64-pc-windows-msvc -S < %s | FileCheck %s --check-prefix=CHECK-WIN64
; RUN: opt -passes=sycl-builtin-remangle -mtriple=i386-pc-linux-gnu -S < %s | FileCheck %s --check-prefix=CHECK-I386

; Test long/long long type remangling

; Test: long parameter (i64 on 64-bit Linux, i32 on Windows/32-bit)
; SYCL: _Z17__spirv_ocl_s_absl (long)
; OpenCL: _Z17__spirv_ocl_s_absl (long) on 64-bit Linux
; OpenCL: _Z17__spirv_ocl_s_absi (int) on Windows/32-bit

declare spir_func i64 @_Z17__spirv_ocl_s_absl(i64)
; CHECK-SPIR64: declare spir_func i64 @_Z17__spirv_ocl_s_absl(i64)
; CHECK-WIN64: declare spir_func i64 @_Z17__spirv_ocl_s_absi(i64)
; CHECK-I386: declare spir_func i64 @_Z17__spirv_ocl_s_absi(i64)

; Test: long long parameter (always i64)
; SYCL: _Z17__spirv_ocl_s_absx (long long)
; OpenCL: _Z17__spirv_ocl_s_absl (long)
; Note: This declaration gets merged with the previous _absl declaration

declare spir_func i64 @_Z17__spirv_ocl_s_absx(i64)

; Test: unsigned long parameter
; SYCL: _Z17__spirv_ocl_u_absm (unsigned long)
; OpenCL: _Z17__spirv_ocl_u_absm (unsigned long) on 64-bit Linux
; OpenCL: _Z17__spirv_ocl_u_absj (unsigned int) on Windows/32-bit

declare spir_func i64 @_Z17__spirv_ocl_u_absm(i64)
; CHECK-SPIR64: declare spir_func i64 @_Z17__spirv_ocl_u_absm(i64)
; CHECK-WIN64: declare spir_func i64 @_Z17__spirv_ocl_u_absj(i64)
; CHECK-I386: declare spir_func i64 @_Z17__spirv_ocl_u_absj(i64)

; Test: unsigned long long parameter
; SYCL: _Z17__spirv_ocl_u_absy (unsigned long long)
; OpenCL: _Z17__spirv_ocl_u_absm (unsigned long) on SPIR64
; OpenCL: _Z17__spirv_ocl_u_absj (unsigned int) on WIN64/I386
; Note: This declaration gets merged with the previous _absm/_absj declaration

declare spir_func i64 @_Z17__spirv_ocl_u_absy(i64)

;===------------------------------------------------------------------------===
; Vector of long
; Real builtin: __spirv_ocl_s_mul_hi (length 20)
;===------------------------------------------------------------------------===

; Test: vec2 long (substitutions preserved)
declare spir_func <2 x i64> @_Z20__spirv_ocl_s_mul_hiDv2_lS_(<2 x i64>, <2 x i64>)
; CHECK-SPIR64: declare spir_func <2 x i64> @_Z20__spirv_ocl_s_mul_hiDv2_lS_(<2 x i64>, <2 x i64>)
; CHECK-WIN64: declare spir_func <2 x i64> @_Z20__spirv_ocl_s_mul_hiDv2_iS_(<2 x i64>, <2 x i64>)
; CHECK-I386: declare spir_func <2 x i64> @_Z20__spirv_ocl_s_mul_hiDv2_iS_(<2 x i64>, <2 x i64>)

; Test: vec4 long (substitutions preserved)
declare spir_func <4 x i64> @_Z20__spirv_ocl_s_mul_hiDv4_lS_(<4 x i64>, <4 x i64>)
; CHECK-SPIR64: declare spir_func <4 x i64> @_Z20__spirv_ocl_s_mul_hiDv4_lS_(<4 x i64>, <4 x i64>)
; CHECK-WIN64: declare spir_func <4 x i64> @_Z20__spirv_ocl_s_mul_hiDv4_iS_(<4 x i64>, <4 x i64>)
; CHECK-I386: declare spir_func <4 x i64> @_Z20__spirv_ocl_s_mul_hiDv4_iS_(<4 x i64>, <4 x i64>)

; Test: vec8 long (substitutions preserved)
declare spir_func <8 x i64> @_Z20__spirv_ocl_s_mul_hiDv8_lS_(<8 x i64>, <8 x i64>)
; CHECK-SPIR64: declare spir_func <8 x i64> @_Z20__spirv_ocl_s_mul_hiDv8_lS_(<8 x i64>, <8 x i64>)
; CHECK-WIN64: declare spir_func <8 x i64> @_Z20__spirv_ocl_s_mul_hiDv8_iS_(<8 x i64>, <8 x i64>)
; CHECK-I386: declare spir_func <8 x i64> @_Z20__spirv_ocl_s_mul_hiDv8_iS_(<8 x i64>, <8 x i64>)

;===------------------------------------------------------------------------===
; Vector of unsigned long
;===------------------------------------------------------------------------===

; Test: vec4 unsigned long
declare spir_func <4 x i64> @_Z20__spirv_ocl_u_mul_hiDv4_mS_(<4 x i64>, <4 x i64>)
; CHECK-SPIR64: declare spir_func <4 x i64> @_Z20__spirv_ocl_u_mul_hiDv4_mS_(<4 x i64>, <4 x i64>)
; CHECK-WIN64: declare spir_func <4 x i64> @_Z20__spirv_ocl_u_mul_hiDv4_jS_(<4 x i64>, <4 x i64>)
; CHECK-I386: declare spir_func <4 x i64> @_Z20__spirv_ocl_u_mul_hiDv4_jS_(<4 x i64>, <4 x i64>)

define spir_func void @test() {
  ; scalar long/long long
  %1 = call spir_func i64 @_Z17__spirv_ocl_s_absl(i64 0)
  %2 = call spir_func i64 @_Z17__spirv_ocl_s_absx(i64 0)
  %3 = call spir_func i64 @_Z17__spirv_ocl_u_absm(i64 0)
  %4 = call spir_func i64 @_Z17__spirv_ocl_u_absy(i64 0)

  ; vector long
  %5 = call spir_func <2 x i64> @_Z20__spirv_ocl_s_mul_hiDv2_lS_(<2 x i64> zeroinitializer, <2 x i64> zeroinitializer)
  %6 = call spir_func <4 x i64> @_Z20__spirv_ocl_s_mul_hiDv4_lS_(<4 x i64> zeroinitializer, <4 x i64> zeroinitializer)
  %7 = call spir_func <8 x i64> @_Z20__spirv_ocl_s_mul_hiDv8_lS_(<8 x i64> zeroinitializer, <8 x i64> zeroinitializer)

  ; vector unsigned long
  %8 = call spir_func <4 x i64> @_Z20__spirv_ocl_u_mul_hiDv4_mS_(<4 x i64> zeroinitializer, <4 x i64> zeroinitializer)

  ret void
}
