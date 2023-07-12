; Test checks the content of simple generated device image.

; RUN: sycl-post-link -split=auto -spec-const=rt -symbols -S -o %t.table %s -generate-device-image-default-spec-consts
; RUN: cat %t.table | FileCheck %s -check-prefix=CHECK-TABLE -DPATH=%t
; RUN: cat %t_0.prop | FileCheck %s -check-prefix=CHECK-PROP0
; RUN: cat %t_1.prop | FileCheck %s -check-prefix=CHECK-PROP1
; RUN: cat %t_0.ll | FileCheck %s -check-prefix=CHECK-IR0
; RUN: cat %t_1.ll | FileCheck %s -check-prefix=CHECK-IR1

; CHECK-TABLE: [[PATH]]_0.ll|[[PATH]]_0.prop|[[PATH]]_0.sym
; CHECK-TABLE: [[PATH]]_1.ll|[[PATH]]_1.prop|[[PATH]]_1.sym

; CHECK-PROP0-NOT: defaultSpecConstants=1|1
; CHECK-PROP0-NOT: originalImage

; CHECK-PROP1: defaultSpecConstants=1|1

; CHECK-IR0: call i32 @_Z20__spirv_SpecConstantii
; CHECK-IR0: call %struct.B @_Z29__spirv_SpecConstantCompositeiii_Rstruct.B
; CHECK-IR0: call %struct.A @_Z29__spirv_SpecConstantCompositeistruct.B_Rstruct.A

; CHECK-IR1-NOT: SpecConstant
; CHECK-IR1: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %1, i8 addrspace(4)* align 8 addrspacecast (i8 addrspace(1)* bitcast (%"class.sycl::_V1::specialization_id" addrspace(1)* @_ZL1c to i8 addrspace(1)*) to i8 addrspace(4)*), i64 16, i1 false)

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::specialization_id" = type { %struct.A }
%struct.A = type { i32, %struct.B }
%struct.B = type { i32, i32, i32 }

@__usid_str = private unnamed_addr constant [28 x i8] c"uida046125e6e1c1f8d____ZL1c\00", align 1
@_ZL1c = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { %struct.A { i32 3, %struct.B { i32 3, i32 2, i32 1 } } }, align 4

declare spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(%struct.A addrspace(4)* sret(%struct.A) align 4, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef)

define spir_kernel void @func1() {
entry:
  %a.i = alloca %struct.A, align 4
  %a.ascast.i = addrspacecast %struct.A* %a.i to %struct.A addrspace(4)*
  %0 = bitcast %struct.A* %a.i to i8*
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(%struct.A addrspace(4)* sret(%struct.A) align 4 %a.ascast.i, i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([28 x i8], [28 x i8]* @__usid_str, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast (%"class.sycl::_V1::specialization_id" addrspace(1)* @_ZL1c to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* noundef null)
  ret void
}
