; Test checks that struct with padding is handled correctly.

; RUN: sycl-post-link -split=auto -spec-const=rt -symbols -S -o %t.table %s -generate-device-image-default-spec-consts
; RUN: cat %t.table | FileCheck %s -check-prefix=CHECK-TABLE -DPATH=%t
; RUN: cat %t_1.prop | FileCheck %s -check-prefix=CHECK-PROP1
; RUN: cat %t_1.ll | FileCheck %s -check-prefix=CHECK-IR1

; CHECK-TABLE: [[PATH]]_0.ll|[[PATH]]_0.prop|[[PATH]]_0.sym
; CHECK-TABLE: [[PATH]]_1.ll|[[PATH]]_1.prop|[[PATH]]_1.sym

; CHECK-PROP1: specConstsReplacedWithDefault=1|1

; CHECK-IR1-NOT: SpecConstant
; CHECK-IR1: store { float, i32, i8 } { float 0x40091EB860000000, i32 42, i8 8 }, { float, i32, i8 } addrspace(4)* %1, align 4

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.A = type <{ float, i32, i8, [3 x i8] }>

@__usid_str = private unnamed_addr constant [28 x i8] c"uida046125e6e1c1f8d____ZL1c\00", align 1
@_ZL1c = internal addrspace(1)  constant { { float, i32, i8 } } { { float, i32, i8 } { float 0x40091EB860000000, i32 42, i8 8 } }, align 4

declare spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(%struct.A addrspace(4)* sret(%struct.A) align 4, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef)

define spir_kernel void @func1() {
entry:
  %a.i = alloca %struct.A, align 4
  %a.ascast.i = addrspacecast %struct.A* %a.i to %struct.A addrspace(4)*
  %0 = bitcast %struct.A* %a.i to i8*
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(%struct.A addrspace(4)* sret(%struct.A) align 4 %a.ascast.i, i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([28 x i8], [28 x i8]* @__usid_str, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast ({ { float, i32, i8 } } addrspace(1)* @_ZL1c to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* noundef null)
  ret void
}
