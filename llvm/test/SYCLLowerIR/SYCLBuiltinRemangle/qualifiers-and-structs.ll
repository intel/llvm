; RUN: opt -passes=sycl-builtin-remangle -sycl-remangle-spirv-target -mtriple=spir64-unknown-unknown -S < %s | FileCheck %s

; Test CV-qualifiers (const/volatile/restrict) preservation in remangling
; Uses real SPIRV builtins: __spirv_ocl_vload and __spirv_GroupAsyncCopy

;===------------------------------------------------------------------------===
; CV-qualifiers (const/volatile/restrict) on pointers
; Using real SPIRV builtins: __spirv_ocl_vload (17 chars)
;===------------------------------------------------------------------------===

; Test: Pointer to const long (K = const qualifier is preserved)
; Real builtin signature: long vload(long, const long*)
declare spir_func i64 @_Z17__spirv_ocl_vloadlPKl(i64, ptr)
; CHECK: declare spir_func i64 @_Z17__spirv_ocl_vloadlPU3AS4Kl(i64, ptr)

; Test: Pointer to volatile int (V = volatile qualifier is preserved)
; Real builtin signature: int vload(int, volatile int*)
declare spir_func i32 @_Z17__spirv_ocl_vloadiPVi(i32, ptr)
; CHECK: declare spir_func i32 @_Z17__spirv_ocl_vloadiPU3AS4Vi(i32, ptr)

; Test: Pointer to const volatile double (both K and V qualifiers preserved)
; Real builtin signature: double vload(double, const volatile double*)
declare spir_func double @_Z17__spirv_ocl_vloaddPVKd(double, ptr)
; CHECK: declare spir_func double @_Z17__spirv_ocl_vloaddPU3AS4VKd(double, ptr)

; Test: Pointer to restrict float (r = restrict qualifier is preserved)
; Real builtin signature: float vload(float, restrict float*)
declare spir_func float @_Z17__spirv_ocl_vloadfPrf(float, ptr)
; CHECK: declare spir_func float @_Z17__spirv_ocl_vloadfPU3AS4rf(float, ptr)

; Test: GroupAsyncCopy with vector<8, _Float16> and const qualifier
; This is the user-requested test case demonstrating:
; - _Float16 (DF16_) -> half (Dh) transformation
; - const qualifier (K) preservation
; - vector substitution (S_) preservation with correct type transformation
; Input:  _Z22__spirv_GroupAsyncCopyiPU3AS1Dv8_DF16_PU3AS3KS_mm9ocl_event
;         GroupAsyncCopy(int, AS1 vector<8, _Float16>*, AS3 const vector<8, _Float16>*, ulong, ulong, ocl_event)
; Output: _Z22__spirv_GroupAsyncCopyiPU3AS1Dv8_DhPU3AS3KS_mm9ocl_event
;         GroupAsyncCopy(int, AS1 vector<8, half>*, AS3 const vector<8, half>*, ulong, ulong, ocl_event)
; Note: K (const) is correctly preserved after DF16_ -> Dh transformation
declare spir_func ptr @_Z22__spirv_GroupAsyncCopyiPU3AS1Dv8_DF16_PU3AS3KS_mm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, ptr)
; CHECK: declare spir_func ptr @_Z22__spirv_GroupAsyncCopyiPU3AS1Dv8_DhPU3AS3KS_mm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, ptr)

define spir_func void @test() {
  ; CV-qualifier tests using real SPIRV builtins
  %ptr_const = alloca i64, align 8
  %1 = call spir_func i64 @_Z17__spirv_ocl_vloadlPKl(i64 0, ptr %ptr_const)

  %ptr_vol = alloca i32, align 4
  %2 = call spir_func i32 @_Z17__spirv_ocl_vloadiPVi(i32 0, ptr %ptr_vol)

  %ptr_cvol = alloca double, align 8
  %3 = call spir_func double @_Z17__spirv_ocl_vloaddPVKd(double 0.0, ptr %ptr_cvol)

  %ptr_restrict = alloca float, align 4
  %4 = call spir_func float @_Z17__spirv_ocl_vloadfPrf(float 0.0, ptr %ptr_restrict)

  ; GroupAsyncCopy test with const qualifier and _Float16 -> half transformation
  %vec_src = alloca <8 x half>, align 16, addrspace(1)
  %vec_dst = alloca <8 x half>, align 16, addrspace(3)
  %event = alloca ptr, align 8
  %5 = call spir_func ptr @_Z22__spirv_GroupAsyncCopyiPU3AS1Dv8_DF16_PU3AS3KS_mm9ocl_event(
      i32 0, ptr addrspace(1) %vec_src, ptr addrspace(3) %vec_dst, i64 8, i64 1, ptr %event)

  ret void
}
