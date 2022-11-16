; This test emulates two translation units with 3 kernels:
; TU0_kernel0 - 1st translation unit, no aspects used
; TU0_kernel1 - 1st translation unit, aspect 1 is used
; TU1_kernel2 - 2nd translation unit, no aspects used

; The test is intended to check that sycl-post-link correctly separates kernels
; that use aspects from kernels which doesn't use aspects regardless of device
; code split mode

; RUN: sycl-post-link -split=auto -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-M0-IR \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-M1-IR \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefixes CHECK-M2-IR \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-M0-SYMS \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-M1-SYMS \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_2.sym --check-prefixes CHECK-M2-SYMS \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1

; RUN: sycl-post-link -split=source -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-M0-IR \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-M1-IR \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefixes CHECK-M2-IR \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-M0-SYMS \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-M1-SYMS \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_2.sym --check-prefixes CHECK-M2-SYMS \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1

; RUN: sycl-post-link -split=kernel -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-M0-IR \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-M1-IR \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefixes CHECK-M2-IR \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-M0-SYMS \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-M1-SYMS \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1
; RUN: FileCheck %s -input-file=%t_2.sym --check-prefixes CHECK-M2-SYMS \
; RUN: --implicit-check-not kernel0 --implicit-check-not kernel1

; Regardless of device code split mode, each kernel should go into a separate
; device image

; CHECK-M2-IR: define {{.*}} @TU0_kernel0
; CHECK-M2-SYMS: TU0_kernel0

; CHECK-M1-IR: define {{.*}} @TU0_kernel1
; CHECK-M1-SYMS: TU0_kernel1

; CHECK-M0-IR: define {{.*}} @TU1_kernel2
; CHECK-M0-SYMS: TU1_kernel2

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

; FIXME: device globals should also be properly distributed across device images
; if they are of optional type
@_ZL2GV = internal addrspace(1) constant [1 x i32] [i32 42], align 4

define dso_local spir_kernel void @TU0_kernel0() #0 {
entry:
  call spir_func void @foo()
  ret void
}

define dso_local spir_func void @foo() {
entry:
  %a = alloca i32, align 4
  %call = call spir_func i32 @bar(i32 1)
  %add = add nsw i32 2, %call
  store i32 %add, i32* %a, align 4
  ret void
}

; Function Attrs: nounwind
define linkonce_odr dso_local spir_func i32 @bar(i32 %arg) {
entry:
  %arg.addr = alloca i32, align 4
  store i32 %arg, i32* %arg.addr, align 4
  %0 = load i32, i32* %arg.addr, align 4
  ret i32 %0
}

define dso_local spir_kernel void @TU0_kernel1() #0 !sycl_used_aspects !2 {
entry:
  call spir_func void @foo1()
  ret void
}

; Function Attrs: nounwind
define dso_local spir_func void @foo1() {
entry:
  %a = alloca i32, align 4
  store i32 2, i32* %a, align 4
  ret void
}

define dso_local spir_kernel void @TU1_kernel2() #1 {
entry:
  call spir_func void @foo2()
  ret void
}

; Function Attrs: nounwind
define dso_local spir_func void @foo2() {
entry:
  %a = alloca i32, align 4
  %0 = load i32, i32 addrspace(4)* getelementptr inbounds ([1 x i32], [1 x i32] addrspace(4)* addrspacecast ([1 x i32] addrspace(1)* @_ZL2GV to [1 x i32] addrspace(4)*), i64 0, i64 0), align 4
  %add = add nsw i32 4, %0
  store i32 %add, i32* %a, align 4
  ret void
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }
attributes #1 = { "sycl-module-id"="TU2.cpp" }

!opencl.spir.version = !{!0, !0}
!spirv.Source = !{!1, !1}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i32 1}
