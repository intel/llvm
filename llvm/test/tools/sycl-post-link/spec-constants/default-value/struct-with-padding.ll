; Test checks that struct with padding is handled correctly.

; RUN: sycl-post-link -split=auto -spec-const=native -S -o %t.table %s -generate-device-image-default-spec-consts
; RUN: cat %t.table | FileCheck %s -check-prefix=CHECK-TABLE
; RUN: cat %t_1.prop | FileCheck %s -check-prefix=CHECK-PROP1
; RUN: cat %t_1.ll | FileCheck %s -check-prefix=CHECK-IR1 --implicit-check-not SpecConstant
; RUN: sycl-post-link -debug-only=SpecConst -split=auto -spec-const=native -S %s -generate-device-image-default-spec-consts 2>&1 | FileCheck %s --check-prefix=CHECK-LOG

; CHECK-TABLE: {{.*}}_0.ll|{{.*}}_0.prop
; CHECK-TABLE: {{.*}}_1.ll|{{.*}}_1.prop

; CHECK-PROP1: specConstsReplacedWithDefault=1|1

; CHECK-IR1: store { float, i32, i8 } { float 0x40091EB860000000, i32 42, i8 8 }, ptr addrspace(4) %a.ascast.i, align 4

; CHECK-LOG: sycl.specialization-constants
; CHECK-LOG:[[UNIQUE_PREFIX:[0-9a-zA-Z]+]]={0, 0, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={1, 4, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={2, 8, 1}
; CHECK-LOG:[[UNIQUE_PREFIX]]={4294967295, 9, 3}
; CHECK-LOG: sycl.specialization-constants-default-values
; CHECK-LOG: {0, 4, 3.140000e+00}
; CHECK-LOG: {4, 4, 42}
; CHECK-LOG: {8, 1, 8}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.A = type <{ float, i32, i8, [3 x i8] }>

@__usid_str = private unnamed_addr constant [28 x i8] c"uida046125e6e1c1f8d____ZL1c\00", align 1
@_ZL1c = internal addrspace(1)  constant { { float, i32, i8 } } { { float, i32, i8 } { float 0x40091EB860000000, i32 42, i8 8 } }, align 4

declare spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(ptr addrspace(4) sret(%struct.A) align 4, ptr addrspace(4) noundef, ptr addrspace(4) noundef, ptr addrspace(4) noundef)

define spir_kernel void @func1() {
entry:
  %a.i = alloca %struct.A, align 4
  %a.ascast.i = addrspacecast %struct.A* %a.i to ptr addrspace(4)
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(ptr addrspace(4) sret(%struct.A) align 4 %a.ascast.i, ptr addrspace(4) noundef addrspacecast (ptr getelementptr inbounds ([28 x i8], ptr @__usid_str, i64 0, i64 0) to ptr addrspace(4)), ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @_ZL1c to ptr addrspace(4)), ptr addrspace(4) noundef null)
  ret void
}
