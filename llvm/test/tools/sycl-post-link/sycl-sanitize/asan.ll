; This test checks that the post-link tool properly generates "asanUsed=1"
; in [SYCL/misc properties]

; RUN: sycl-post-link -split=kernel -symbols -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop
; CHECK: [SYCL/misc properties]
; CHECK: asanUsed=1

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.AssertHappened.10 = type { i32, [257 x i8], [257 x i8], [129 x i8], i32, i64, i64, i64, i64, i64, i64 }
%"class.sycl::_V1::range.12" = type { %"class.sycl::_V1::detail::array.11" }
%"class.sycl::_V1::detail::array.11" = type { [2 x i64] }
%"class.sycl::_V1::detail::RoundedRangeIDGenerator.15" = type <{ %"class.sycl::_V1::range.12", %"class.sycl::_V1::range.12", %"class.sycl::_V1::range.12", %"class.sycl::_V1::range.12", i8, [7 x i8] }>
%"class.sycl::_V1::detail::RoundedRangeKernel.17" = type <{ %"class.sycl::_V1::range.12", %class.anon.16, [7 x i8] }>
%class.anon.16 = type { i8 }
%struct.__devicelib_div_t_32.13 = type { i32, i32 }
%struct.__devicelib_div_t_64.14 = type { i64, i64 }

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @asan.module_ctor, ptr null }]
@__spirv_BuiltInGlobalSize = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@SPIR_AssertHappenedMem = linkonce_odr dso_local addrspace(1) global %struct.AssertHappened.10 zeroinitializer, align 8
@__spirv_BuiltInWorkgroupId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalLinearId = external dso_local local_unnamed_addr addrspace(1) constant i64, align 8
@__spirv_BuiltInWorkgroupSize = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: nounwind uwtable
define internal void @asan.module_ctor() #0 {
  call void @__asan_init()
  call void @__asan_version_mismatch_check_v8()
  ret void
}

declare void @__asan_init()

declare void @__asan_version_mismatch_check_v8()

; Function Attrs: mustprogress norecurse nounwind sanitize_address uwtable
define weak_odr dso_local spir_kernel void @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZ4mainENKUlRNS0_7handlerEE_clES4_E9TheKernelEE(ptr noundef byval(%"class.sycl::_V1::range.12") align 8 %_arg_UserRange) local_unnamed_addr #1 {
entry:
  ret void
}

attributes #0 = { nounwind uwtable "frame-pointer"="all" "sycl-optlevel"="2" }
attributes #1 = { mustprogress norecurse nounwind sanitize_address uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test.cpp" "sycl-optlevel"="2" "uniform-work-group-size"="true" }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!llvm.ident = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!llvm.module.flags = !{!3, !4, !5}
!sycl.specialization-constants = !{}
!sycl.specialization-constants-default-values = !{}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 18.0.0 (https://github.com/AllanZyne/llvm.git 97c052ed8efa30f750dacf8d89e8e64743ec03f7)"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i32 8544724}
!7 = !{i32 -1}
!8 = !{}
!9 = !{i1 false, i1 true}
