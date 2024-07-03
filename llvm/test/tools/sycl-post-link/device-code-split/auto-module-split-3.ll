; RUN: sycl-post-link -properties -split=auto -symbols -S < %s -o %t.table
;
; In precense of indirect calls we start matching functions using their
; signatures, i.e. we have an indirect call to i32(i32) function within
; @_Z3foov, which means that all functions with i32(i32) signature should be
; placed in the same module as @_Z3foov.
;
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-TU0-IR \
; RUN:     --implicit-check-not TU0_kernel --implicit-check-not _Z3foov \
; RUN:     --implicit-check-not _Z4foo3v
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-TU1-IR \
; RUN:     --implicit-check-not TU1_kernel --implicit-check-not _Z4foo2v \
; RUN:     --implicit-check-not _Z4foo1v
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-TU0-SYM
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-TU1-SYM
;
;
; RUN: sycl-module-split -split=auto -S < %s -o %t2
;
; RUN: FileCheck %s -input-file=%t2_0.ll --check-prefixes CHECK-TU0-IR \
; RUN:     --implicit-check-not TU0_kernel --implicit-check-not _Z3foov \
; RUN:     --implicit-check-not _Z4foo3v
; RUN: FileCheck %s -input-file=%t2_1.ll --check-prefixes CHECK-TU1-IR \
; RUN:     --implicit-check-not TU1_kernel --implicit-check-not _Z4foo2v \
; RUN:     --implicit-check-not _Z4foo1v
; RUN: FileCheck %s -input-file=%t2_0.sym --check-prefixes CHECK-TU0-SYM
; RUN: FileCheck %s -input-file=%t2_1.sym --check-prefixes CHECK-TU1-SYM

; CHECK-TU0-SYM: _ZTSZ4mainE11TU1_kernel0
; CHECK-TU0-SYM: _ZTSZ4mainE11TU1_kernel1
;
; CHECK-TU1-SYM: _ZTSZ4mainE10TU0_kernel
;
; CHECK-TU0-IR: @_ZL2GV = internal addrspace(1) constant
; CHECK-TU0-IR: define dso_local spir_kernel void @_ZTSZ4mainE11TU1_kernel0
; CHECK-TU0-IR: define dso_local spir_func i32 @_Z4foo1v
; CHECK-TU0-IR: define dso_local spir_kernel void @_ZTSZ4mainE11TU1_kernel1
; CHECK-TU0-IR: define dso_local spir_func void @_Z4foo2v
;
; CHECK-TU1-IR: define dso_local spir_kernel void @_ZTSZ4mainE10TU0_kernel
; CHECK-TU1-IR: define dso_local spir_func void @_Z3foov
; CHECK-TU1-IR: define dso_local spir_func i32 @_Z4foo3v

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

$_Z3barIiET_S0_ = comdat any

@_ZL2GV = internal addrspace(1) constant [1 x i32] [i32 42], align 4

define dso_local spir_kernel void @_ZTSZ4mainE10TU0_kernel() #0 {
entry:
  call spir_func void @_Z3foov()
  ret void
}

define dso_local spir_func void @_Z3foov() {
entry:
  %a = alloca i32, align 4
  %ptr = bitcast i32* %a to i32 (i32)*
  %call = call spir_func i32 %ptr(i32 1)
  %add = add nsw i32 2, %call
  store i32 %add, i32* %a, align 4
  ret void
}

; Function Attrs: nounwind
define linkonce_odr dso_local spir_func i32 @_Z3barIiET_S0_(i32 %arg) comdat {
entry:
  %arg.addr = alloca i32, align 4
  store i32 %arg, i32* %arg.addr, align 4
  %0 = load i32, i32* %arg.addr, align 4
  ret i32 %0
}

define dso_local spir_kernel void @_ZTSZ4mainE11TU1_kernel0() #1 {
entry:
  %a = alloca i32, align 4
  %arg = load i32, i32* %a, align 4
  %call = call spir_func i32 @_Z4foo1v(i32 %arg)
  ret void
}

; Function Attrs: nounwind
define dso_local spir_func i32 @_Z4foo1v(i32 %arg) {
entry:
  %a = alloca i32, align 4
  store i32 %arg, i32* %a, align 4
  ret i32 %arg
}

; Function Attrs: nounwind
define dso_local spir_func i32 @_Z4foo3v(i32 %arg) #2 {
entry:
  %a = alloca i32, align 4
  store i32 %arg, i32* %a, align 4
  ret i32 %arg
}

define dso_local spir_kernel void @_ZTSZ4mainE11TU1_kernel1() #1 {
entry:
  call spir_func void @_Z4foo2v()
  ret void
}

; Function Attrs: nounwind
define dso_local spir_func void @_Z4foo2v() {
entry:
  %a = alloca i32, align 4
  %0 = load i32, i32 addrspace(4)* getelementptr inbounds ([1 x i32], [1 x i32] addrspace(4)* addrspacecast ([1 x i32] addrspace(1)* @_ZL2GV to [1 x i32] addrspace(4)*), i64 0, i64 0), align 4
  %add = add nsw i32 4, %0
  store i32 %add, i32* %a, align 4
  ret void
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }
attributes #1 = { "sycl-module-id"="TU2.cpp" }
attributes #2 = { "referenced-indirectly" }

!opencl.spir.version = !{!0, !0}
!spirv.Source = !{!1, !1}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
