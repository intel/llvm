; RUN: sycl-post-link -split=auto -symbols -S %s -o %t.table
; In precense of indirectly callable function auto mode is equal to no split,
; which means that separate LLVM IR file for device is not generated and we only
; need to check generated symbol table
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

$_Z3barIiET_S0_ = comdat any

@_ZL2GV = internal addrspace(1) constant [1 x i32] [i32 42], align 4

; CHECK: {{.*}}TU0_kernel0{{.*}}

define dso_local spir_kernel void @_ZTSZ4mainE11TU0_kernel0() #0 {
entry:
  call spir_func void @_Z3foov()
  ret void
}

define dso_local spir_func void @_Z3foov() #2 {
entry:
  %a = alloca i32, align 4
  %call = call spir_func i32 @_Z3barIiET_S0_(i32 1)
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

; CHECK: {{.*}}TU0_kernel1{{.*}}

define dso_local spir_kernel void @_ZTSZ4mainE11TU0_kernel1() #0 {
entry:
  call spir_func void @_Z4foo1v()
  ret void
}

; Function Attrs: nounwind
define dso_local spir_func void @_Z4foo1v() {
entry:
  %a = alloca i32, align 4
  store i32 2, i32* %a, align 4
  ret void
}
; CHECK: {{.*}}TU1_kernel{{.*}}

define dso_local spir_kernel void @_ZTSZ4mainE10TU1_kernel() #1 {
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
