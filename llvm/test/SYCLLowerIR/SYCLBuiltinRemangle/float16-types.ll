; RUN: opt -passes=sycl-builtin-remangle -mtriple=nvptx64-nvidia-cuda -S < %s | FileCheck %s

; Test _Float16/half type remangling with real SPIRV builtins from libspirv.ll

; Test: _Float16 parameter
; SYCL user code: _Z16__spirv_ocl_fabsDF16_ (_Float16)
; libspirv OpenCL: _Z16__spirv_ocl_fabsDh (half)

declare spir_func half @_Z16__spirv_ocl_fabsDF16_(half)
; CHECK: declare spir_func half @_Z16__spirv_ocl_fabsDh(half)

; Test: half parameter (should stay as half)
; SYCL: _Z16__spirv_ocl_fabsDh (half)
; OpenCL: _Z16__spirv_ocl_fabsDh (half)
; Note: This declaration gets merged with the previous _fabsDh declaration

declare spir_func half @_Z16__spirv_ocl_fabsDh(half)

; Test: vector of _Float16
; SYCL: _Z16__spirv_ocl_fabsDv4_DF16_ (vector<_Float16, 4>)
; OpenCL: _Z16__spirv_ocl_fabsDv4_Dh (vector<half, 4>)

declare spir_func <4 x half> @_Z16__spirv_ocl_fabsDv4_DF16_(<4 x half>)
; CHECK: declare spir_func <4 x half> @_Z16__spirv_ocl_fabsDv4_Dh(<4 x half>)

; Test: _Float16 in complex function with address spaces, substitutions, const qualifier, and ocl_event
; SYCL: _Z22__spirv_GroupAsyncCopyiPU3AS3Dv2_DF16_PU3AS1KS_mm9ocl_event
;       GroupAsyncCopy(int, vector<2, _Float16> AS3*, const vector<2, _Float16> AS1*, unsigned long, unsigned long, ocl_event)
; OpenCL: _Z22__spirv_GroupAsyncCopyiPU3AS3Dv2_DhPU3AS1KS_mm9ocl_event
;         GroupAsyncCopy(int, vector<2, half> AS3*, const vector<2, half> AS1*, unsigned long, unsigned long, ocl_event)
; Note: _Float16 -> half transformation preserves const qualifier (K) and substitutions (S_)

declare spir_func ptr @_Z22__spirv_GroupAsyncCopyiPU3AS3Dv2_DF16_PU3AS1KS_mm9ocl_event(i32, ptr addrspace(3), ptr addrspace(1), i64, i64, ptr)
; CHECK: declare spir_func ptr @_Z22__spirv_GroupAsyncCopyiPU3AS3Dv2_DhPU3AS1KS_mm9ocl_event(i32, ptr addrspace(3), ptr addrspace(1), i64, i64, ptr)

define spir_func void @test() {
  %1 = call spir_func half @_Z16__spirv_ocl_fabsDF16_(half 1.0)
  %2 = call spir_func half @_Z16__spirv_ocl_fabsDh(half 1.0)
  %3 = call spir_func <4 x half> @_Z16__spirv_ocl_fabsDv4_DF16_(<4 x half> zeroinitializer)
  %4 = call spir_func ptr @_Z22__spirv_GroupAsyncCopyiPU3AS3Dv2_DF16_PU3AS1KS_mm9ocl_event(i32 0, ptr addrspace(3) null, ptr addrspace(1) null, i64 0, i64 0, ptr null)
  ret void
}
