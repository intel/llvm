; RUN: sycl-post-link -spec-const=rt --ir-output-only %s -S -o - \
; RUN: | FileCheck %s --implicit-check-not "call {{.*}} __sycl_getCompositeSpecConstantValue"
;
; This test is intended to check that sycl-post-link tool is capable of handling
; situations when the same composite specialization constants is used more than
; once. Unlike multiple-composite-spec-const-usages.ll test, this is a real life
; LLVM IR example
;
; CHECK-LABEL: @_ZTSN4test8kernel_tIfEE
; CHECK: %[[#X1:]] = call float @_Z20__spirv_SpecConstantif(i32 0, float 0
; CHECK: %[[#Y1:]] = call float @_Z20__spirv_SpecConstantif(i32 1, float 0
; CHECK: call {{.*}} @_Z29__spirv_SpecConstantCompositeff(float %[[#X1]], float %[[#Y1]])
; CHECK-LABEL: @_ZTSN4test8kernel_tIiEE
; CHECK: %[[#X2:]] = call float @_Z20__spirv_SpecConstantif(i32 0, float 0
; CHECK: %[[#Y2:]] = call float @_Z20__spirv_SpecConstantif(i32 1, float 0
; CHECK: call {{.*}} @_Z29__spirv_SpecConstantCompositeff(float %[[#X2]], float %[[#Y2]])

; CHECK: !sycl.specialization-constants = !{![[#ID:]]

; CHECK: ![[#ID]] = !{!"_ZTS11sc_kernel_t", i32 0, i32 0, i32 4, i32 1, i32 4, i32 4}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%"struct._ZTSN4test5pod_tE.test::pod_t" = type { float, float }

$_ZTSN4test8kernel_tIfEE = comdat any

$_ZTSN4test8kernel_tIiEE = comdat any

@__builtin_unique_stable_name._ZNK2cl4sycl3ext6oneapi12experimental13spec_constantIN4test5pod_tE11sc_kernel_tE3getIS5_EENSt9enable_ifIXaasr3std8is_classIT_EE5valuesr3std6is_podISA_EE5valueESA_E4typeEv = private unnamed_addr addrspace(1) constant [18 x i8] c"_ZTS11sc_kernel_t\00", align 1

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSN4test8kernel_tIfEE() local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
  %ref.tmp.i = alloca %"struct._ZTSN4test5pod_tE.test::pod_t", align 4
  %0 = bitcast %"struct._ZTSN4test5pod_tE.test::pod_t"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #3
  %1 = addrspacecast %"struct._ZTSN4test5pod_tE.test::pod_t"* %ref.tmp.i to %"struct._ZTSN4test5pod_tE.test::pod_t" addrspace(4)*
  call spir_func void @_Z36__sycl_getCompositeSpecConstantValueIN4test5pod_tEET_PKc(%"struct._ZTSN4test5pod_tE.test::pod_t" addrspace(4)* sret(%"struct._ZTSN4test5pod_tE.test::pod_t") align 4 %1, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([18 x i8], [18 x i8] addrspace(1)* @__builtin_unique_stable_name._ZNK2cl4sycl3ext6oneapi12experimental13spec_constantIN4test5pod_tE11sc_kernel_tE3getIS5_EENSt9enable_ifIXaasr3std8is_classIT_EE5valuesr3std6is_podISA_EE5valueESA_E4typeEv, i64 0, i64 0) to i8 addrspace(4)*)) #4
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #3
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z36__sycl_getCompositeSpecConstantValueIN4test5pod_tEET_PKc(%"struct._ZTSN4test5pod_tE.test::pod_t" addrspace(4)* sret(%"struct._ZTSN4test5pod_tE.test::pod_t") align 4, i8 addrspace(4)*) local_unnamed_addr #2

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSN4test8kernel_tIiEE() local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
  %ref.tmp.i = alloca %"struct._ZTSN4test5pod_tE.test::pod_t", align 4
  %0 = bitcast %"struct._ZTSN4test5pod_tE.test::pod_t"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #3
  %1 = addrspacecast %"struct._ZTSN4test5pod_tE.test::pod_t"* %ref.tmp.i to %"struct._ZTSN4test5pod_tE.test::pod_t" addrspace(4)*
  call spir_func void @_Z36__sycl_getCompositeSpecConstantValueIN4test5pod_tEET_PKc(%"struct._ZTSN4test5pod_tE.test::pod_t" addrspace(4)* sret(%"struct._ZTSN4test5pod_tE.test::pod_t") align 4 %1, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([18 x i8], [18 x i8] addrspace(1)* @__builtin_unique_stable_name._ZNK2cl4sycl3ext6oneapi12experimental13spec_constantIN4test5pod_tE11sc_kernel_tE3getIS5_EENSt9enable_ifIXaasr3std8is_classIT_EE5valuesr3std6is_podISA_EE5valueESA_E4typeEv, i64 0, i64 0) to i8 addrspace(4)*)) #4
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #3
  ret void
}

attributes #0 = { convergent norecurse "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="repro-1.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { convergent "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { convergent }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 12.0.0 (/data/github.com/intel/llvm/clang 9b7086f7cef079b80ac5e137394f8d77d5d49c3e)"}
!4 = !{}
