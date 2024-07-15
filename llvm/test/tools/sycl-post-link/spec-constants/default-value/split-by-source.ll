; Test checks generation of device images for splitted kernels by source.

; RUN: sycl-post-link -properties -split=source -o %t.table %s -generate-device-image-default-spec-consts
; RUN: cat %t.table | FileCheck %s -check-prefix=CHECK-TABLE
; RUN: cat %t_0.prop | FileCheck %s -check-prefix=CHECK-PROP0
; RUN: cat %t_1.prop | FileCheck %s -check-prefix=CHECK-PROP1
; RUN: cat %t_2.prop | FileCheck %s -check-prefix=CHECK-PROP2
; RUN: cat %t_3.prop | FileCheck %s -check-prefix=CHECK-PROP3

; CHECK-TABLE: {{.*}}_0.bc|{{.*}}_0.prop
; CHECK-TABLE: {{.*}}_1.bc|{{.*}}_1.prop
; CHECK-TABLE: {{.*}}_2.bc|{{.*}}_2.prop
; CHECK-TABLE: {{.*}}_3.bc|{{.*}}_3.prop

; CHECK-PROP0-NOT: specConstsReplacedWithDefault=1|1

; CHECK-PROP1: specConstsReplacedWithDefault=1|1

; CHECK-PROP2-NOT: specConstsReplacedWithDefault=1|1

; CHECK-PROP3: specConstsReplacedWithDefault=1|1

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::specialization_id" = type { %struct.A }
%struct.A = type { i32, %struct.B }
%struct.B = type { i32, i32, i32 }

@__usid_str = private unnamed_addr constant [28 x i8] c"uida046125e6e1c1f8d____ZL1c\00", align 1
@_ZL1c = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { %struct.A { i32 3, %struct.B { i32 3, i32 2, i32 1 } } }, align 4

declare spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(%struct.A addrspace(4)* sret(%struct.A) align 4, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef)

define spir_kernel void @kernel1() #0 {
entry:
  %a.i = alloca %struct.A, align 4
  %a.ascast.i = addrspacecast %struct.A* %a.i to %struct.A addrspace(4)*
  %0 = bitcast %struct.A* %a.i to i8*
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(%struct.A addrspace(4)* sret(%struct.A) align 4 %a.ascast.i, i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([28 x i8], [28 x i8]* @__usid_str, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast (%"class.sycl::_V1::specialization_id" addrspace(1)* @_ZL1c to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* noundef null)
  ret void
}

define spir_kernel void @kernel2() #1 {
entry:
  %a.i = alloca %struct.A, align 4
  %a.ascast.i = addrspacecast %struct.A* %a.i to %struct.A addrspace(4)*
  %0 = bitcast %struct.A* %a.i to i8*
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(%struct.A addrspace(4)* sret(%struct.A) align 4 %a.ascast.i, i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([28 x i8], [28 x i8]* @__usid_str, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast (%"class.sycl::_V1::specialization_id" addrspace(1)* @_ZL1c to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* noundef null)
  ret void
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }
attributes #1 = { "sycl-module-id"="TU2.cpp" }
