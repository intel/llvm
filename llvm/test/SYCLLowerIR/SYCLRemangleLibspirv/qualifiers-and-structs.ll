; RUN: opt -passes=sycl-remangle-libspirv --remangle-long-width=64 --remangle-char-signedness=signed -mtriple=nvptx64-nvidia-cuda -S < %s | FileCheck %s

; Test CV-qualifiers (const/volatile/restrict) preservation in remangling.

; Pointer to const long (K = const qualifier is preserved).
define i64 @_Z17__spirv_ocl_vloadlPKl(i64, ptr) { unreachable }
; CHECK-DAG: define i64 @_Z17__spirv_ocl_vloadxPKx(
; CHECK-DAG: define i64 @_Z17__spirv_ocl_vloadlPKl(

; Pointer to volatile int (V = volatile qualifier is preserved).
define i32 @_Z17__spirv_ocl_vloadiPVi(i32, ptr) { unreachable }
; CHECK-DAG: define i32 @_Z17__spirv_ocl_vloadiPVi(

; Pointer to const volatile double (both K and V qualifiers preserved).
define double @_Z17__spirv_ocl_vloaddPVKd(double, ptr) { unreachable }
; CHECK-DAG: define double @_Z17__spirv_ocl_vloaddPVKd(

; Pointer to restrict float (r = restrict qualifier is preserved).
define float @_Z17__spirv_ocl_vloadfPrf(float, ptr) { unreachable }
; CHECK-DAG: define float @_Z17__spirv_ocl_vloadfPrf(

; GroupAsyncCopy with vector<8, _Float16> and const qualifier.
define ptr @_Z22__spirv_GroupAsyncCopyiPU3AS1Dv8_DhPU3AS3KS_mm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, ptr) { unreachable }
; CHECK-DAG: define ptr @_Z22__spirv_GroupAsyncCopyiPU3AS1Dv8_DF16_PU3AS3KS_yy9ocl_event(
; CHECK-DAG: define ptr @_Z22__spirv_GroupAsyncCopyiPU3AS1Dv8_DF16_PU3AS3KS_mm9ocl_event(
