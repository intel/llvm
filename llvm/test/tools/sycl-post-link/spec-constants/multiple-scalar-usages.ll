; This test checks that IDs assigned to spec constants are correct, i.e. if some
; spec constant is accessed twice, then metadata for both accesses should point
; to the same ID

; RUN: sycl-post-link -spec-const=rt --ir-output-only %s -S -o - \
; RUN: | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown-sycldevice"

%"spec_constant" = type { i8 }

@SCSymID = private unnamed_addr constant [10 x i8] c"SpecConst\00", align 1
@SCSymID1 = private unnamed_addr constant [11 x i8] c"SpecConst1\00", align 1

declare dso_local spir_func float @_Z33__sycl_getScalarSpecConstantValueIfET_PKc(i8 addrspace(4)*)

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @Kernel() {
  %1 = call spir_func float @_Z33__sycl_getScalarSpecConstantValueIfET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([10 x i8], [10 x i8]* @SCSymID, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK: call float @_Z20__spirv_SpecConstantif(i32 0, {{.*}})
  ret void
}

; Function Attrs: norecurse
define dso_local spir_func float @foo_float(%"spec_constant" addrspace(4)* nocapture readnone dereferenceable(1) %0) local_unnamed_addr #3 {
  %2 = tail call spir_func float @_Z33__sycl_getScalarSpecConstantValueIfET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([11 x i8], [11 x i8]* @SCSymID1, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK: call float @_Z20__spirv_SpecConstantif(i32 1, {{.*}})
  ret float %2
}

; Function Attrs: norecurse
define dso_local spir_func float @foo_float2(%"spec_constant" addrspace(4)* nocapture readnone dereferenceable(1) %0) local_unnamed_addr #3 {
  %2 = tail call spir_func float @_Z33__sycl_getScalarSpecConstantValueIfET_PKc(i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([10 x i8], [10 x i8]* @SCSymID, i64 0, i64 0) to i8 addrspace(4)*))
; CHECK: call float @_Z20__spirv_SpecConstantif(i32 0, {{.*}})
  ret float %2
}

; CHECK: !sycl.specialization-constants = !{![[#MD1:]], ![[#MD2:]]}
; CHECK: ![[#MD1]] = !{!"SpecConst", i32 0, i32 0, i32 4}
; CHECK: ![[#MD2]] = !{!"SpecConst1", i32 1, i32 0, i32 4}
