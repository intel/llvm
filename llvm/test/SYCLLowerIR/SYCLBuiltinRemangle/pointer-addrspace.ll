; RUN: opt -passes=sycl-builtin-remangle -sycl-remangle-spirv-target -mtriple=spir64-unknown-unknown -S < %s | FileCheck %s

; Test pointer type transformations with various address spaces

;===------------------------------------------------------------------------===
; Pointers with different address spaces
; Real builtin: __spirv_ocl_fract (length 17)
; Signature: float fract(float x, float *iptr)
;===------------------------------------------------------------------------===

; Test: Pointer with explicit AS1 (global) - preserved
declare spir_func float @_Z17__spirv_ocl_fractfPU3AS1f(float, ptr addrspace(1))
; CHECK: declare spir_func float @_Z17__spirv_ocl_fractfPU3AS1f(float, ptr addrspace(1))

; Test: Pointer with explicit AS2 (constant) - preserved
declare spir_func float @_Z17__spirv_ocl_fractfPU3AS2f(float, ptr addrspace(2))
; CHECK: declare spir_func float @_Z17__spirv_ocl_fractfPU3AS2f(float, ptr addrspace(2))

; Test: Pointer with explicit AS3 (local) - preserved
declare spir_func float @_Z17__spirv_ocl_fractfPU3AS3f(float, ptr addrspace(3))
; CHECK: declare spir_func float @_Z17__spirv_ocl_fractfPU3AS3f(float, ptr addrspace(3))

; Test: Pointer with explicit AS0 (private) - becomes implicit in OpenCL C
; SYCL explicit AS0 -> OpenCL C implicit (no AS qualifier)
declare spir_func float @_Z17__spirv_ocl_fractfPU3AS0f(float, ptr)
; CHECK: declare spir_func float @_Z17__spirv_ocl_fractfPf(float, ptr)

; Test: Pointer without AS qualifier (implicit generic) - becomes explicit AS4
; SYCL implicit (generic) -> OpenCL C explicit AS4
declare spir_func float @_Z17__spirv_ocl_fractfPf(float, ptr addrspace(4))
; CHECK: declare spir_func float @_Z17__spirv_ocl_fractfPU3AS4f(float, ptr addrspace(4))

;===------------------------------------------------------------------------===
; Pointers with different pointee types
;===------------------------------------------------------------------------===

; Test: Pointer to half
declare spir_func half @_Z17__spirv_ocl_fractDhPDh(half, ptr addrspace(4))
; CHECK: declare spir_func half @_Z17__spirv_ocl_fractDhPU3AS4Dh(half, ptr addrspace(4))

; Test: Pointer to double
declare spir_func double @_Z17__spirv_ocl_fractdPd(double, ptr addrspace(4))
; CHECK: declare spir_func double @_Z17__spirv_ocl_fractdPU3AS4d(double, ptr addrspace(4))

; Test: Pointer to vector (substitution for vector type)
; Function signature: fract(<4 x float>, <4 x float>*)
declare spir_func <4 x float> @_Z17__spirv_ocl_fractDv4_fPS_(<4 x float>, ptr addrspace(4))
; CHECK: declare spir_func <4 x float> @_Z17__spirv_ocl_fractDv4_fPU3AS4S_(<4 x float>, ptr addrspace(4))

define spir_func void @test() {
  ; Address space variations
  %ptr_as1 = alloca float, align 4, addrspace(1)
  %1 = call spir_func float @_Z17__spirv_ocl_fractfPU3AS1f(float 0.0, ptr addrspace(1) %ptr_as1)

  %ptr_as2 = alloca float, align 4, addrspace(2)
  %2 = call spir_func float @_Z17__spirv_ocl_fractfPU3AS2f(float 0.0, ptr addrspace(2) %ptr_as2)

  %ptr_as3 = alloca float, align 4, addrspace(3)
  %3 = call spir_func float @_Z17__spirv_ocl_fractfPU3AS3f(float 0.0, ptr addrspace(3) %ptr_as3)

  %ptr_as0 = alloca float, align 4
  %4 = call spir_func float @_Z17__spirv_ocl_fractfPU3AS0f(float 0.0, ptr %ptr_as0)

  %ptr_implicit = alloca float, align 4, addrspace(4)
  %5 = call spir_func float @_Z17__spirv_ocl_fractfPf(float 0.0, ptr addrspace(4) %ptr_implicit)

  ; Different pointee types
  %ptr_h = alloca half, align 2, addrspace(4)
  %6 = call spir_func half @_Z17__spirv_ocl_fractDhPDh(half 0.0, ptr addrspace(4) %ptr_h)

  %ptr_d = alloca double, align 8, addrspace(4)
  %7 = call spir_func double @_Z17__spirv_ocl_fractdPd(double 0.0, ptr addrspace(4) %ptr_d)

  %ptr_vec = alloca <4 x float>, align 16, addrspace(4)
  %8 = call spir_func <4 x float> @_Z17__spirv_ocl_fractDv4_fPS_(<4 x float> zeroinitializer, ptr addrspace(4) %ptr_vec)

  ret void
}
