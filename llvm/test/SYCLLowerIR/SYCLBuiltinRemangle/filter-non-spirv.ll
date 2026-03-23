; RUN: opt -passes=sycl-builtin-remangle -mtriple=nvptx64-nvidia-cuda -S < %s | FileCheck %s

; Test that non-SPIRV builtins and malformed SPIRV names are not remangled

;===------------------------------------------------------------------------===
; User functions (not SPIRV builtins) - should NOT be remangled
;===------------------------------------------------------------------------===

; Test: User function with long parameter
declare void @_Z10user_func1l(i64)
; CHECK: declare void @_Z10user_func1l(i64)

; Test: User function with long long parameter
declare void @_Z10user_func1x(i64)
; CHECK: declare void @_Z10user_func1x(i64)

; Test: User function with _Float16
declare void @_Z10user_func1DF16_(half)
; CHECK: declare void @_Z10user_func1DF16_(half)

; Test: User function with signed char
declare void @_Z10user_func1a(i8)
; CHECK: declare void @_Z10user_func1a(i8)

; Test: User function with vector of long
declare void @_Z10user_func1Dv4_l(<4 x i64>)
; CHECK: declare void @_Z10user_func1Dv4_l(<4 x i64>)

;===------------------------------------------------------------------------===
; Non-mangled and LLVM intrinsics - should NOT be remangled
;===------------------------------------------------------------------------===

; Test: LLVM intrinsic
declare i64 @llvm.abs.i64(i64, i1)
; CHECK: declare i64 @llvm.abs.i64(i64, i1{{.*}})

; Test: Non-mangled function
declare void @simple_function(i64)
; CHECK: declare void @simple_function(i64)

;===------------------------------------------------------------------------===
; Functions with definitions - should NOT be remangled (only declarations)
;===------------------------------------------------------------------------===

define void @_Z10defined_fn1l(i64 %x) {
  ret void
}
; CHECK: define void @_Z10defined_fn1l(i64 %x)

;===------------------------------------------------------------------------===
; Malformed SPIRV builtins - should NOT be remangled (can't demangle)
;===------------------------------------------------------------------------===

; Test: Wrong mangled name length (says 17 but actual is different)
; This was a fake symbol used in old tests - should NOT be transformed
declare spir_func <4 x half> @_Z17__spirv_ocl_fabsDv4_DF16_(<4 x half>)
; CHECK: declare spir_func <4 x half> @_Z17__spirv_ocl_fabsDv4_DF16_(<4 x half>)

; Test: Invalid type encoding - demangler can parse but converter will fail
declare spir_func void @_Z15__spirv_invalidXYZ(i32)
; CHECK: declare spir_func void @_Z15__spirv_invalidXYZ(i32)

;===------------------------------------------------------------------------===
; Valid SPIRV builtins - SHOULD be remangled (positive control)
;===------------------------------------------------------------------------===

; Test: Real SPIRV builtin with transformable types
declare spir_func i64 @_Z20__spirv_ocl_s_mul_hill(i64, i64)
; CHECK: declare spir_func i64 @_Z20__spirv_ocl_s_mul_hill(i64, i64)

; Test: Real SPIRV builtin without transformable types
declare spir_func void @_Z22__spirv_ControlBarrieriii(i32, i32, i32)
; CHECK: declare spir_func void @_Z22__spirv_ControlBarrieriii(i32, i32, i32)

define spir_func void @test() {
  ; User functions
  call void @_Z10user_func1l(i64 0)
  call void @_Z10user_func1x(i64 0)
  call void @_Z10user_func1DF16_(half 0.0)
  call void @_Z10user_func1a(i8 0)
  call void @_Z10user_func1Dv4_l(<4 x i64> zeroinitializer)

  ; Non-mangled and intrinsics
  %1 = call i64 @llvm.abs.i64(i64 0, i1 false)
  call void @simple_function(i64 0)

  ; Functions with definitions
  call void @_Z10defined_fn1l(i64 0)

  ; Malformed SPIRV builtins
  %2 = call spir_func <4 x half> @_Z17__spirv_ocl_fabsDv4_DF16_(<4 x half> zeroinitializer)
  call spir_func void @_Z15__spirv_invalidXYZ(i32 0)

  ; Valid SPIRV builtins
  %3 = call spir_func i64 @_Z20__spirv_ocl_s_mul_hill(i64 0, i64 0)
  call spir_func void @_Z22__spirv_ControlBarrieriii(i32 0, i32 0, i32 0)

  ret void
}
