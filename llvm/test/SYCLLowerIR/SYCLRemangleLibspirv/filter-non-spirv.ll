; RUN: opt -passes=sycl-remangle-libspirv --remangle-long-width=64 --remangle-char-signedness=signed -mtriple=nvptx64-nvidia-cuda -S < %s | FileCheck %s

; Test that non-SPIR-V builtins and malformed SPIR-V names are not remangled.

; User function with long parameter.
define void @_Z10user_func1l(i64) { unreachable }
; CHECK-DAG: define void @_Z10user_func1l(

; User function with long long parameter.
define void @_Z10user_func1x(i64) { unreachable }
; CHECK-DAG: define void @_Z10user_func1x(

; User function with _Float16.
define void @_Z10user_func1DF16_(half) { unreachable }
; CHECK-DAG: define void @_Z10user_func1DF16_(

; User function with signed char.
define void @_Z10user_func1a(i8) { unreachable }
; CHECK-DAG: define void @_Z10user_func1a(

; User function with vector of long.
define void @_Z10user_func1Dv4_l(<4 x i64>) { unreachable }
; CHECK-DAG: define void @_Z10user_func1Dv4_l(

; LLVM intrinsic.
declare i64 @llvm.abs.i64(i64)
; CHECK-DAG: declare i64 @llvm.abs.i64(

; Non-mangled function.
define void @simple_function(i64) { unreachable }
; CHECK-DAG: define void @simple_function(

; Wrong mangled name length (says 17 but actual is different).
define <4 x half> @_Z17__spirv_ocl_fabsDv4_DF16_(<4 x half>) { unreachable }
; CHECK-DAG: define <4 x half> @_Z17__spirv_ocl_fabsDv4_DF16_(

; Invalid type encoding - demangler can parse but converter fails.
define void @_Z15__spirv_invalidXYZ(i32) { unreachable }
; CHECK-DAG: define void @_Z15__spirv_invalidXYZ(
