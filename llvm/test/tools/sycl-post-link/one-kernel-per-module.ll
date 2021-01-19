; RUN: sycl-post-link -split=kernel -symbols -S %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.ll --check-prefixes CHECK-MODULE0,CHECK
; RUN: FileCheck %s -input-file=%t.files_1.ll --check-prefixes CHECK-MODULE1,CHECK
; RUN: FileCheck %s -input-file=%t.files_2.ll --check-prefixes CHECK-MODULE2,CHECK
; RUN: FileCheck %s -input-file=%t.files_0.sym --check-prefixes CHECK-MODULE0-TXT
; RUN: FileCheck %s -input-file=%t.files_1.sym --check-prefixes CHECK-MODULE1-TXT
; RUN: FileCheck %s -input-file=%t.files_2.sym --check-prefixes CHECK-MODULE2-TXT
; ModuleID = 'one-kernel-per-module.ll'
source_filename = "one-kernel-per-module.ll"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

$_Z3barIiET_S0_ = comdat any

;CHECK-MODULE0-NOT: @{{.*}}GV{{.*}}
;CHECK-MODULE1-NOT: @{{.*}}GV{{.*}}
;CHECK-MODULE2: @{{.*}}GV{{.*}} = internal addrspace(1) constant [1 x i32] [i32 42], align 4
@_ZL2GV = internal addrspace(1) constant [1 x i32] [i32 42], align 4

; CHECK-MODULE0: define dso_local spir_kernel void @{{.*}}TU0_kernel0{{.*}}
; CHECK-MODULE0-TXT: {{.*}}TU0_kernel0{{.*}}
; CHECK-MODULE1-NOT: define dso_local spir_kernel void @{{.*}}TU0_kernel0{{.*}}
; CHECK-MODULE1-TXT-NOT: {{.*}}TU0_kernel0{{.*}}

; CHECK-MODULE0: call spir_func void @{{.*}}foo{{.*}}()

define dso_local spir_kernel void @TU0_kernel0() #0 {
entry:
  call spir_func void @_Z3foov()
  ret void
}

; CHECK-MODULE0: define dso_local spir_func void @{{.*}}foo{{.*}}()
; CHECK-MODULE1-NOT: define dso_local spir_func void @{{.*}}foo{{.*}}()
; CHECK-MODULE2-NOT: define dso_local spir_func void @{{.*}}foo{{.*}}()

; CHECK-MODULE0: call spir_func i32 @{{.*}}bar{{.*}}(i32 1)

define dso_local spir_func void @_Z3foov() {
entry:
  %a = alloca i32, align 4
  %call = call spir_func i32 @_Z3barIiET_S0_(i32 1)
  %add = add nsw i32 2, %call
  store i32 %add, i32* %a, align 4
  ret void
}

; CHECK-MODULE0: define {{.*}} spir_func i32 @{{.*}}bar{{.*}}(i32 %arg)
; CHECK-MODULE1-NOT: define {{.*}} spir_func i32 @{{.*}}bar{{.*}}(i32 %arg)
; CHECK-MODULE2-NOT: define {{.*}} spir_func i32 @{{.*}}bar{{.*}}(i32 %arg)

; Function Attrs: nounwind
define linkonce_odr dso_local spir_func i32 @_Z3barIiET_S0_(i32 %arg) comdat {
entry:
  %arg.addr = alloca i32, align 4
  store i32 %arg, i32* %arg.addr, align 4
  %0 = load i32, i32* %arg.addr, align 4
  ret i32 %0
}

; CHECK-MODULE0-NOT: define dso_local spir_kernel void @{{.*}}TU0_kernel1{{.*}}()
; CHECK-MODULE0-TXT-NOT: {{.*}}TU0_kernel1{{.*}}
; CHECK-MODULE1: define dso_local spir_kernel void @{{.*}}TU0_kernel1{{.*}}()
; CHECK-MODULE1-TXT: {{.*}}TU0_kernel1{{.*}}
; CHECK-MODULE2-NOT: define dso_local spir_kernel void @{{.*}}TU0_kernel1{{.*}}()
; CHECK-MODULE2-TXT-NOT: {{.*}}TU0_kernel1{{.*}}

; CHECK-MODULE1: call spir_func void @{{.*}}foo1{{.*}}()

define dso_local spir_kernel void @TU0_kernel1() #0 {
entry:
  call spir_func void @_Z4foo1v()
  ret void
}

; CHECK-MODULE0-NOT: define dso_local spir_func void @{{.*}}foo1{{.*}}()
; CHECK-MODULE1: define dso_local spir_func void @{{.*}}foo1{{.*}}()
; CHECK-MODULE2-NOT: define dso_local spir_func void @{{.*}}foo1{{.*}}()

; Function Attrs: nounwind
define dso_local spir_func void @_Z4foo1v() {
entry:
  %a = alloca i32, align 4
  store i32 2, i32* %a, align 4
  ret void
}

; CHECK-MODULE0-NOT: define dso_local spir_kernel void @{{.*}}TU1_kernel{{.*}}()
; CHECK-MODULE0-TXT-NOT: {{.*}}TU1_kernel{{.*}}
; CHECK-MODULE1-NOT: define dso_local spir_kernel void @{{.*}}TU1_kernel{{.*}}()
; CHECK-MODULE1-TXT-NOT: {{.*}}TU1_kernel{{.*}}
; CHECK-MODULE2: define dso_local spir_kernel void @{{.*}}TU1_kernel{{.*}}()
; CHECK-MODULE2-TXT: {{.*}}TU1_kernel{{.*}}

; CHECK-MODULE2: call spir_func void @{{.*}}foo2{{.*}}()

define dso_local spir_kernel void @TU1_kernel() #1 {
entry:
  call spir_func void @_Z4foo2v()
  ret void
}

; CHECK-MODULE0-NOT: define dso_local spir_func void @{{.*}}foo2{{.*}}()
; CHECK-MODULE1-NOT: define dso_local spir_func void @{{.*}}foo2{{.*}}()
; CHECK-MODULE2: define dso_local spir_func void @{{.*}}foo2{{.*}}()

; Function Attrs: nounwind
define dso_local spir_func void @_Z4foo2v() {
entry:
  %a = alloca i32, align 4
; CHECK-MODULE2: %0 = load i32, i32 addrspace(4)* getelementptr inbounds ([1 x i32], [1 x i32] addrspace(4)* addrspacecast ([1 x i32] addrspace(1)* @{{.*}}GV{{.*}} to [1 x i32] addrspace(4)*), i64 0, i64 0), align 4
  %0 = load i32, i32 addrspace(4)* getelementptr inbounds ([1 x i32], [1 x i32] addrspace(4)* addrspacecast ([1 x i32] addrspace(1)* @_ZL2GV to [1 x i32] addrspace(4)*), i64 0, i64 0), align 4
  %add = add nsw i32 4, %0
  store i32 %add, i32* %a, align 4
  ret void
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }
attributes #1 = { "sycl-module-id"="TU2.cpp" }

; Metadata is saved in both modules.
; CHECK: !opencl.spir.version = !{!0, !0}
; CHECK: !spirv.Source = !{!1, !1}

!opencl.spir.version = !{!0, !0}
!spirv.Source = !{!1, !1}

; CHECK; !0 = !{i32 1, i32 2}
; CHECK; !1 = !{i32 4, i32 100000}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
