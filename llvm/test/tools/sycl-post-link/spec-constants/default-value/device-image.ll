; Test checks the content of simple generated device image.
; It checks scalar, sret and "return by value" versions of SpecConstant functions.
; Also test checks generated symbols.

; RUN: sycl-post-link -split=auto -spec-const=native -symbols -S -o %t.table %s -generate-device-image-default-spec-consts
; RUN: FileCheck %s -input-file %t.table -check-prefix=CHECK-TABLE
; RUN: FileCheck %s -input-file %t_0.prop -check-prefix=CHECK-PROP0
; RUN: FileCheck %s -input-file %t_1.prop -check-prefix=CHECK-PROP1
; RUN: FileCheck %s -input-file %t_0.ll -check-prefix=CHECK-IR0
; RUN: FileCheck %s -input-file %t_1.ll -check-prefix=CHECK-IR1 --implicit-check-not "SpecConstant"
; RUN: FileCheck %s -input-file %t_0.sym -check-prefix=CHECK-SYM0
; RUN: FileCheck %s -input-file %t_1.sym -check-prefix=CHECK-SYM1

; CHECK-TABLE: {{.*}}_0.ll|{{.*}}_0.prop|{{.*}}_0.sym
; CHECK-TABLE: {{.*}}_1.ll|{{.*}}_1.prop|{{.*}}_1.sym

; CHECK-PROP0-NOT: specConstsReplacedWithDefault=1|1

; CHECK-PROP1: specConstsReplacedWithDefault=1|1

; CHECK-IR0: call i32 @_Z20__spirv_SpecConstantii
; CHECK-IR0: call %struct.B @_Z29__spirv_SpecConstantCompositeiii_Rstruct.B
; CHECK-IR0: call %struct.A @_Z29__spirv_SpecConstantCompositeistruct.B_Rstruct.A

; CHECK-IR1: store %struct.A { i32 3, %struct.B { i32 3, i32 2, i32 1 } }, %struct.A addrspace(4)* %a.ascast.i, align 4

; Check that %scalar value has been replaced by global value.
; CHECK-IR1-NOT: %scalar = call
; CHECK-IR1: %scalar2 = add i32 123, 1

; Check that %returned_spec_const value has been replaced by global value.
; CHECK-IR1-NOT: %returned_spec_const = call
; CHECK-IR1: %sc.e = extractvalue %struct.C { i32 1 }, 0

; CHECK-SYM0: kernel
; CHECK-SYM0-EMPTY:

; CHECK-SYM1: kernel
; CHECK-SYM1-EMPTY:

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::specialization_id" = type { %struct.A }
%"class.sycl::_V1::specialization_id.2" = type { i32 }
%"class.sycl::_V1::specialization_id.3" = type { %struct.C }
%struct.A = type { i32, %struct.B }
%struct.B = type { i32, i32, i32 }
%struct.C = type { i32 }

@__usid_str = private unnamed_addr constant [28 x i8] c"uida046125e6e1c1f8d____ZL1c\00", align 1
@__usid_str.1 = private unnamed_addr constant [33 x i8] c"uidcac21ed8fab7d507____ZL6valueS\00", align 1
@__usid_str.2 = private unnamed_addr constant [28 x i8] c"uida046125e6e1c1f8f____ZL1b\00", align 1
@_ZL1c = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { %struct.A { i32 3, %struct.B { i32 3, i32 2, i32 1 } } }, align 4
@_ZL6valueS = internal addrspace(1) constant %"class.sycl::_V1::specialization_id.2" { i32 123 }, align 4
@_ZL1b = internal addrspace(1) constant %"class.sycl::_V1::specialization_id.3" { %struct.C { i32 1 } }, align 4

; This constant checks `zeroinitializer` field.
@_ZL1d = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { %struct.A { i32 3, %struct.B zeroinitializer } }, align 4

declare spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(%struct.A addrspace(4)* sret(%struct.A) align 4, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef)

declare spir_func %struct.C @_Z40__sycl_getComposite2020SpecConstantValueI1CET_PKcPKvS5_(i8 addrspace(4)* noundef, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef)

declare dso_local spir_func noundef i32 @_Z37__sycl_getScalar2020SpecConstantValueIiET_PKcPKvS4_(i8 addrspace(4)* noundef, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef)

; Function for testing symbol generation
define spir_func void @func() {
  ret void
}

define spir_kernel void @kernel() {
entry:
  %a.i = alloca %struct.A, align 4
  %a.ascast.i = addrspacecast %struct.A* %a.i to %struct.A addrspace(4)*
  %0 = bitcast %struct.A* %a.i to i8*
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(%struct.A addrspace(4)* sret(%struct.A) align 4 %a.ascast.i, i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([28 x i8], [28 x i8]* @__usid_str, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast (%"class.sycl::_V1::specialization_id" addrspace(1)* @_ZL1c to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* noundef null)
  %scalar = call spir_func i32 @_Z37__sycl_getScalar2020SpecConstantValueIiET_PKcPKvS4_(i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([33 x i8], [33 x i8]* @__usid_str.1, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast (%"class.sycl::_V1::specialization_id.2" addrspace(1)* @_ZL6valueS to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* noundef null)
  %scalar2 = add i32 %scalar, 1

  %returned_spec_const = call spir_func %struct.C @_Z40__sycl_getComposite2020SpecConstantValueI1CET_PKcPKvS5_(i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([28 x i8], [28 x i8]* @__usid_str.2, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast (%"class.sycl::_V1::specialization_id.3" addrspace(1)* @_ZL1b to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* noundef null)
  %sc.e = extractvalue %struct.C %returned_spec_const, 0
  %scalar3 = add i32 %sc.e, 1

  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI1AET_PKcPKvS5_(%struct.A addrspace(4)* sret(%struct.A) align 4 %a.ascast.i, i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([28 x i8], [28 x i8]* @__usid_str, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast (%"class.sycl::_V1::specialization_id" addrspace(1)* @_ZL1d to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* noundef null)


  call void @func()
  ret void
}
