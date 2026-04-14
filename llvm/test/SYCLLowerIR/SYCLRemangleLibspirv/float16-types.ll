; RUN: opt -passes=sycl-remangle-libspirv --remangle-long-width=64 --remangle-char-signedness=signed -mtriple=nvptx64-nvidia-cuda -S < %s | FileCheck %s

; Test _Float16/half type remangling.

; _Float16 parameter.
define half @_Z16__spirv_ocl_fabsDh(half) { unreachable }
; CHECK-DAG: define half @_Z16__spirv_ocl_fabsDF16_(

; vector of _Float16.
define <4 x half> @_Z16__spirv_ocl_fabsDv4_Dh(<4 x half>) { unreachable }
; CHECK-DAG: define <4 x half> @_Z16__spirv_ocl_fabsDv4_DF16_(

; _Float16 in complex function with address spaces, substitutions, const qualifier, and ocl_event.
define ptr @_Z22__spirv_GroupAsyncCopyiPU3AS3Dv2_DhPU3AS1KS_mm9ocl_event(i32, ptr addrspace(3), ptr addrspace(1), i64, i64, ptr) { unreachable }
; CHECK-DAG: define ptr @_Z22__spirv_GroupAsyncCopyiPU3AS3Dv2_DF16_PU3AS1KS_yy9ocl_event(
; CHECK-DAG: define ptr @_Z22__spirv_GroupAsyncCopyiPU3AS3Dv2_DF16_PU3AS1KS_mm9ocl_event(
