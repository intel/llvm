; This test checks that the post-link tool works correctly when both
; device code splitting and specialization constant processing are
; requested.
;
; RUN: sycl-post-link -split=kernel -spec-const=rt -S %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.ll --check-prefixes CHECK0,CHECK
; RUN: FileCheck %s -input-file=%t.files_1.ll --check-prefixes CHECK1,CHECK

@SCSymID = private unnamed_addr constant [10 x i8] c"SpecConst\00", align 1
; CHECK-NOT: @SCSymID

declare dso_local spir_func zeroext i1 @_Z33__sycl_getScalarSpecConstantValueIbET_PKc(i8 addrspace(4)*)

define dso_local spir_kernel void @KERNEL_AAA() {
  %1 = call spir_func zeroext i1 @_Z33__sycl_getScalarSpecConstantValueIbET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([10 x i8], [10 x i8]* @SCSymID, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK0: %{{[0-9]+}} = call i1 @_Z20__spirv_SpecConstantib(i32 0, i1 false)
  ret void
}

define dso_local spir_kernel void @KERNEL_BBB() {
  %1 = call spir_func zeroext i1 @_Z33__sycl_getScalarSpecConstantValueIbET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([10 x i8], [10 x i8]* @SCSymID, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK1: %{{[0-9]+}} = call i1 @_Z20__spirv_SpecConstantib(i32 0, i1 false)
  ret void
}

; CHECK: !sycl.specialization-constants = !{![[#MD:]]}

; CHECK: ![[#MD:]] = !{!"SpecConst", i32 0, i32 0, i32 1}
