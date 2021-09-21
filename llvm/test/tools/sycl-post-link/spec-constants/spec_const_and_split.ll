; This test checks that the post-link tool works correctly when both
; device code splitting and specialization constant processing are
; requested.
;
; Additionally this test checks that sycl-post-link creates specialization
; constant property correctly: i.e. it doesn't put all specialization constants
; into properties of each device image

; RUN: sycl-post-link -split=kernel -spec-const=rt -S %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.ll --check-prefixes CHECK-IR0
; RUN: FileCheck %s -input-file=%t.files_1.ll --check-prefixes CHECK-IR1
; RUN: FileCheck %s -input-file=%t.files_2.ll --check-prefixes CHECK-IR2
; RUN: FileCheck %s -input-file=%t.files_0.prop --check-prefixes CHECK-PROP0
; RUN: FileCheck %s -input-file=%t.files_1.prop --check-prefixes CHECK-PROP1
; RUN: FileCheck %s -input-file=%t.files_2.prop --check-prefixes CHECK-PROP2

@SCSymID = private unnamed_addr constant [10 x i8] c"SpecConst\00", align 1
@SCSymID2 = private unnamed_addr constant [11 x i8] c"SpecConst2\00", align 1

declare dso_local spir_func zeroext i1 @_Z33__sycl_getScalarSpecConstantValueIbET_PKc(i8 addrspace(4)*)

define dso_local spir_kernel void @KERNEL_AAA() #0 {
  %1 = call spir_func zeroext i1 @_Z33__sycl_getScalarSpecConstantValueIbET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([10 x i8], [10 x i8]* @SCSymID, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK-IR0: %{{[0-9]+}} = call i1 @_Z20__spirv_SpecConstantib(i32 1, i1 false)
  %2 = call spir_func zeroext i1 @_Z33__sycl_getScalarSpecConstantValueIbET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([11 x i8], [11 x i8]* @SCSymID2, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK-IR0: %{{[0-9]+}} = call i1 @_Z20__spirv_SpecConstantib(i32 0, i1 false)
  ret void
}

define dso_local spir_kernel void @KERNEL_BBB() #0 {
  %1 = call spir_func zeroext i1 @_Z33__sycl_getScalarSpecConstantValueIbET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([10 x i8], [10 x i8]* @SCSymID, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK-IR1: %{{[0-9]+}} = call i1 @_Z20__spirv_SpecConstantib(i32 0, i1 false)
  ret void
}

define dso_local spir_kernel void @KERNEL_CCC() #0 {
; CHECK-IR2: define{{.*}}spir_kernel void @KERNEL_CCC
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }

; CHECK-IR0: !sycl.specialization-constants = !{![[#MD0:]], ![[#MD1:]]}
; CHECK-IR0: ![[#MD0:]] = !{!"SpecConst2", i32 0, i32 0, i32 1}
; CHECK-IR0: ![[#MD1:]] = !{!"SpecConst", i32 1, i32 0, i32 1}
;
; CHECK-IR1: !sycl.specialization-constants = !{![[#MD0:]]}
; CHECK-IR1: ![[#MD0:]] = !{!"SpecConst", i32 0, i32 0, i32 1}
;
; CHECK-IR2: !sycl.specialization-constants = !{}

; CHECK-PROP0: [SYCL/specialization constants]
; CHECK-PROP0-DAG: SpecConst=2|
; CHECK-PROP0-DAG: SpecConst2=2|
;
; CHECK-PROP1: [SYCL/specialization constants]
; CHECK-PROP1: SpecConst=2|
; CHECK-PROP1-NOT: SpecConst2
;
; CHECK-PROP2: [SYCL/specialization constants]
; CHECK-PROP2-EMPTY:
