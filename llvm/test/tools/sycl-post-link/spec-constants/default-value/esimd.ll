; Test checks generation of device image of esimd kernel.

; RUN: sycl-post-link -properties -split=auto -split-esimd -lower-esimd -O2 -spec-const=native -o %t.table %s -generate-device-image-default-spec-consts
; RUN: FileCheck %s -input-file=%t.table -check-prefix=CHECK-TABLE
; RUN: FileCheck %s -input-file=%t_1.prop -check-prefix=CHECK-PROP
; RUN: FileCheck %s -input-file=%t_esimd_1.prop -check-prefix=CHECK-ESIMD-PROP
; RUN: %if asserts %{ sycl-post-link -properties -debug-only=SpecConst -split=auto -split-esimd -lower-esimd -O2 -spec-const=native %s -generate-device-image-default-spec-consts 2>&1 | FileCheck %s --check-prefix=CHECK-LOG %}

; CHECK-TABLE: {{.*}}_esimd_0.bc|{{.*}}_esimd_0.prop
; CHECK-TABLE: {{.*}}_0.bc|{{.*}}_0.prop
; CHECK-TABLE: {{.*}}_esimd_1.bc|{{.*}}_esimd_1.prop
; CHECK-TABLE: {{.*}}_1.bc|{{.*}}_1.prop

; CHECK-PROP: specConstsReplacedWithDefault=1|1

; CHECK-ESIMD-PROP: isEsimdImage=1|1
; CHECK-ESIMD-PROP: specConstsReplacedWithDefault=1|1
; CHECK-LOG: sycl.specialization-constants
; CHECK-LOG:[[UNIQUE_PREFIX:[0-9a-zA-Z]+]]={0, 0, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={1, 4, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={2, 8, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={3, 12, 4}
; CHECK-LOG: sycl.specialization-constants-default-values
; CHECK-LOG: {0, 4, 3}
; CHECK-LOG: {4, 4, 3}
; CHECK-LOG: {8, 4, 2}
; CHECK-LOG: {12, 4, 1}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::specialization_id" = type { %struct.A }
%struct.A = type { i32, %struct.B }
%struct.B = type { i32, i32, i32 }

@__usid_str = private unnamed_addr constant [28 x i8] c"uida046125e6e1c1f8d____ZL1c\00", align 1
@_ZL1c = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { %struct.A { i32 3, %struct.B { i32 3, i32 2, i32 1 } } }, align 4

declare spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(%struct.A addrspace(4)* sret(%struct.A) align 4, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef)

define spir_kernel void @func1() !kernel_arg_buffer_location !7 !sycl_kernel_omit_args !8 {
entry:
  %a.i = alloca %struct.A, align 4
  %a.ascast.i = addrspacecast %struct.A* %a.i to %struct.A addrspace(4)*
  %0 = bitcast %struct.A* %a.i to i8*
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(%struct.A addrspace(4)* sret(%struct.A) align 4 %a.ascast.i, i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([28 x i8], [28 x i8]* @__usid_str, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast (%"class.sycl::_V1::specialization_id" addrspace(1)* @_ZL1c to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* noundef null)
  ret void
}

define spir_kernel void @func2(i8 addrspace(1)* noundef align 1 %_arg__specialization_constants_buffer) !sycl_explicit_simd !1 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 !kernel_arg_accessor_ptr !6 {
entry:
  %a.i = alloca %struct.A, align 4
  %a.ascast.i = addrspacecast %struct.A* %a.i to %struct.A addrspace(4)*
  %0 = bitcast %struct.A* %a.i to i8*
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(%struct.A addrspace(4)* sret(%struct.A) align 4 %a.ascast.i, i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([28 x i8], [28 x i8]* @__usid_str, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast (%"class.sycl::_V1::specialization_id" addrspace(1)* @_ZL1c to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* noundef null)
  ret void
}

!1 = !{}
!2 = !{i32 1}
!3 = !{!"none"}
!4 = !{!"char*"}
!5 = !{!""}
!6 = !{i1 false}
!7 = !{i32 -1}
!8 = !{i1 true}
