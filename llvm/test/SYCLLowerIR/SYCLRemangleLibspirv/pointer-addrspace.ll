; RUN: opt -passes=sycl-remangle-libspirv --remangle-spirv-target --remangle-long-width=64 --remangle-char-signedness=signed -mtriple=spir64-unknown-unknown -S < %s | FileCheck %s

; Test pointer type transformations with various address spaces.

; Pointer with explicit AS1 (global) - preserved.
define spir_func float @_Z17__spirv_ocl_fractfPU3AS1f(float, ptr addrspace(1)) { unreachable }
; CHECK-DAG: define spir_func float @_Z17__spirv_ocl_fractfPU3AS1f(

; Pointer with explicit AS2 (constant) - preserved.
define spir_func float @_Z17__spirv_ocl_fractfPU3AS2f(float, ptr addrspace(2)) { unreachable }
; CHECK-DAG: define spir_func float @_Z17__spirv_ocl_fractfPU3AS2f(

; Pointer with explicit AS3 (local) - preserved.
define spir_func float @_Z17__spirv_ocl_fractfPU3AS3f(float, ptr addrspace(3)) { unreachable }
; CHECK-DAG: define spir_func float @_Z17__spirv_ocl_fractfPU3AS3f(

; Pointer without AS qualifier (implicit private) - becomes explicit AS0.
define spir_func float @_Z17__spirv_ocl_fractfPf(float, ptr) { unreachable }
; CHECK-DAG: define spir_func float @_Z17__spirv_ocl_fractfPU3AS0f(

; Pointer with explicit AS4 (generic) - becomes implicit.
define spir_func float @_Z17__spirv_ocl_fractfPU3AS4f(float, ptr addrspace(4)) { unreachable }
; CHECK-DAG: define spir_func float @_Z17__spirv_ocl_fractfPf(

; Pointer to half.
define spir_func half @_Z17__spirv_ocl_fractDhPDh(half, ptr) { unreachable }
; CHECK-DAG: define spir_func half @_Z17__spirv_ocl_fractDF16_PU3AS0DF16_(

; Pointer to double.
define spir_func double @_Z17__spirv_ocl_fractdPd(double, ptr) { unreachable }
; CHECK-DAG: define spir_func double @_Z17__spirv_ocl_fractdPU3AS0d(

; Pointer to vector (substitution for vector type).
; fract(<4 x float>, <4 x float>*)
define spir_func <4 x float> @_Z17__spirv_ocl_fractDv4_fPS_(<4 x float>, ptr) { unreachable }
; CHECK-DAG: define spir_func <4 x float> @_Z17__spirv_ocl_fractDv4_fPU3AS0S_(

