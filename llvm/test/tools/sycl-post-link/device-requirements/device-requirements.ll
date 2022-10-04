; Original code:
; #include <sycl/sycl.hpp>
; [[__sycl_detail__::__uses_aspects__(sycl::aspect::fp64, sycl::aspect::cpu)]] void foo() {}
; [[__sycl_detail__::__uses_aspects__(sycl::aspect::queue_profiling, sycl::aspect::cpu, sycl::aspect::image)]] void bar() {
;   foo();
; }
; int main() {
;   sycl::queue q;
;   q.submit([&](sycl::handler &cgh) {
;     cgh.single_task([=]() { bar(); });
;   });
; }

; RUN: sycl-post-link -split=auto -o %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_2.prop --check-prefix CHECK-PROP

; CHECK-PROP: [SYCL/device requirements]
; CHECK-PROP-NEXT: aspects=2|ACAAAAAAAAQAAAAAGAAAAkAAAAADAAAA

source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%class.anon = type { i8 }

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_ = comdat any

@__spirv_BuiltInWorkgroupId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalLinearId = external dso_local local_unnamed_addr addrspace(1) constant i64, align 8
@__spirv_BuiltInWorkgroupSize = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent mustprogress noinline norecurse optnone
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_() #0 comdat !kernel_arg_buffer_location !6 {
entry:
  call void @__itt_offload_wi_start_wrapper()
  %__SYCLKernel = alloca %class.anon, align 1
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(%class.anon addrspace(4)* noundef align 1 dereferenceable_or_null(1) %__SYCLKernel.ascast) #7
  call void @__itt_offload_wi_finish_wrapper()
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(%class.anon addrspace(4)* noundef align 1 dereferenceable_or_null(1) %this) #1 align 2 {
entry:
  %this.addr = alloca %class.anon addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %class.anon addrspace(4)** %this.addr to %class.anon addrspace(4)* addrspace(4)*
  store %class.anon addrspace(4)* %this, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %class.anon addrspace(4)*, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  call spir_func void @_Z3barv() #7
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define dso_local spir_func void @_Z3barv() #1 !intel_used_aspects !7 {
entry:
  call spir_func void @_Z3foov() #7
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_func void @_Z3foov() #2 !intel_used_aspects !8 {
entry:
  ret void
}

; Function Attrs: alwaysinline convergent mustprogress norecurse
define weak dso_local spir_func void @__itt_offload_wi_start_wrapper() #3 {
entry:
  %GroupID = alloca [3 x i64], align 8
  %call.i = tail call spir_func signext i8 @__spirv_SpecConstant(i32 noundef -9145239, i8 noundef signext 0) #7
  %cmp.i.not = icmp eq i8 %call.i, 0
  br i1 %cmp.i.not, label %return, label %if.end

if.end:                                           ; preds = %entry
  %0 = bitcast [3 x i64]* %GroupID to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #8
  %arrayinit.begin5 = getelementptr inbounds [3 x i64], [3 x i64]* %GroupID, i64 0, i64 0
  %arrayinit.begin = addrspacecast i64* %arrayinit.begin5 to i64 addrspace(4)*
  %1 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupId to <3 x i64> addrspace(4)*), align 32
  %2 = extractelement <3 x i64> %1, i64 0
  store i64 %2, i64 addrspace(4)* %arrayinit.begin, align 8, !tbaa !9
  %arrayinit.element6 = getelementptr inbounds [3 x i64], [3 x i64]* %GroupID, i64 0, i64 1
  %arrayinit.element = addrspacecast i64* %arrayinit.element6 to i64 addrspace(4)*
  %3 = extractelement <3 x i64> %1, i64 1
  store i64 %3, i64 addrspace(4)* %arrayinit.element, align 8, !tbaa !9
  %arrayinit.element17 = getelementptr inbounds [3 x i64], [3 x i64]* %GroupID, i64 0, i64 2
  %arrayinit.element1 = addrspacecast i64* %arrayinit.element17 to i64 addrspace(4)*
  %4 = extractelement <3 x i64> %1, i64 2
  store i64 %4, i64 addrspace(4)* %arrayinit.element1, align 8, !tbaa !9
  %5 = load i64, i64 addrspace(4)* addrspacecast (i64 addrspace(1)* @__spirv_BuiltInGlobalLinearId to i64 addrspace(4)*), align 8, !tbaa !9
  %6 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupSize to <3 x i64> addrspace(4)*), align 32
  %7 = extractelement <3 x i64> %6, i64 0
  %8 = extractelement <3 x i64> %6, i64 1
  %mul = mul i64 %7, %8
  %9 = extractelement <3 x i64> %6, i64 2
  %mul2 = mul i64 %mul, %9
  %conv = trunc i64 %mul2 to i32
  call spir_func void @__itt_offload_wi_start_stub(i64 addrspace(4)* noundef %arrayinit.begin, i64 noundef %5, i32 noundef %conv) #7
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #8
  br label %return

return:                                           ; preds = %entry, %if.end
  ret void
}

; Function Attrs: convergent
declare extern_weak dso_local spir_func signext i8 @__spirv_SpecConstant(i32 noundef, i8 noundef signext) local_unnamed_addr #4

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #5

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #5

; Function Attrs: alwaysinline convergent mustprogress norecurse
define weak dso_local spir_func void @__itt_offload_wi_finish_wrapper() #3 {
entry:
  %GroupID = alloca [3 x i64], align 8
  %call.i = tail call spir_func signext i8 @__spirv_SpecConstant(i32 noundef -9145239, i8 noundef signext 0) #7
  %cmp.i.not = icmp eq i8 %call.i, 0
  br i1 %cmp.i.not, label %return, label %if.end

if.end:                                           ; preds = %entry
  %0 = bitcast [3 x i64]* %GroupID to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #8
  %arrayinit.begin3 = getelementptr inbounds [3 x i64], [3 x i64]* %GroupID, i64 0, i64 0
  %arrayinit.begin = addrspacecast i64* %arrayinit.begin3 to i64 addrspace(4)*
  %1 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupId to <3 x i64> addrspace(4)*), align 32
  %2 = extractelement <3 x i64> %1, i64 0
  store i64 %2, i64 addrspace(4)* %arrayinit.begin, align 8, !tbaa !9
  %arrayinit.element4 = getelementptr inbounds [3 x i64], [3 x i64]* %GroupID, i64 0, i64 1
  %arrayinit.element = addrspacecast i64* %arrayinit.element4 to i64 addrspace(4)*
  %3 = extractelement <3 x i64> %1, i64 1
  store i64 %3, i64 addrspace(4)* %arrayinit.element, align 8, !tbaa !9
  %arrayinit.element15 = getelementptr inbounds [3 x i64], [3 x i64]* %GroupID, i64 0, i64 2
  %arrayinit.element1 = addrspacecast i64* %arrayinit.element15 to i64 addrspace(4)*
  %4 = extractelement <3 x i64> %1, i64 2
  store i64 %4, i64 addrspace(4)* %arrayinit.element1, align 8, !tbaa !9
  %5 = load i64, i64 addrspace(4)* addrspacecast (i64 addrspace(1)* @__spirv_BuiltInGlobalLinearId to i64 addrspace(4)*), align 8, !tbaa !9
  call spir_func void @__itt_offload_wi_finish_stub(i64 addrspace(4)* noundef %arrayinit.begin, i64 noundef %5) #7
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #8
  br label %return

return:                                           ; preds = %entry, %if.end
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define weak dso_local spir_func void @__itt_offload_wi_start_stub(i64 addrspace(4)* noundef %group_id, i64 noundef %wi_id, i32 noundef %wg_size) local_unnamed_addr #6 {
entry:
  %group_id.addr = alloca i64 addrspace(4)*, align 8
  %wi_id.addr = alloca i64, align 8
  %wg_size.addr = alloca i32, align 4
  %group_id.addr.ascast = addrspacecast i64 addrspace(4)** %group_id.addr to i64 addrspace(4)* addrspace(4)*
  %wi_id.addr.ascast = addrspacecast i64* %wi_id.addr to i64 addrspace(4)*
  %wg_size.addr.ascast = addrspacecast i32* %wg_size.addr to i32 addrspace(4)*
  store i64 addrspace(4)* %group_id, i64 addrspace(4)* addrspace(4)* %group_id.addr.ascast, align 8, !tbaa !13
  store i64 %wi_id, i64 addrspace(4)* %wi_id.addr.ascast, align 8, !tbaa !9
  store i32 %wg_size, i32 addrspace(4)* %wg_size.addr.ascast, align 4, !tbaa !15
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define weak dso_local spir_func void @__itt_offload_wi_finish_stub(i64 addrspace(4)* noundef %group_id, i64 noundef %wi_id) local_unnamed_addr #6 {
entry:
  %group_id.addr = alloca i64 addrspace(4)*, align 8
  %wi_id.addr = alloca i64, align 8
  %group_id.addr.ascast = addrspacecast i64 addrspace(4)** %group_id.addr to i64 addrspace(4)* addrspace(4)*
  %wi_id.addr.ascast = addrspacecast i64* %wi_id.addr to i64 addrspace(4)*
  store i64 addrspace(4)* %group_id, i64 addrspace(4)* addrspace(4)* %group_id.addr.ascast, align 8, !tbaa !13
  store i64 %wi_id, i64 addrspace(4)* %wi_id.addr.ascast, align 8, !tbaa !9
  ret void
}

attributes #0 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="main.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { alwaysinline convergent mustprogress norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/localdisk2/nkornev/llvm/libdevice/itt_compiler_wrappers.cpp" }
attributes #4 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { argmemonly nocallback nofree nosync nounwind willreturn }
attributes #6 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/localdisk2/nkornev/llvm/libdevice/itt_stubs.cpp" }
attributes #7 = { convergent }
attributes #8 = { nounwind }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!llvm.ident = !{!2, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
!llvm.module.flags = !{!4, !5}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 16.0.0"}
!3 = !{!"clang version 16.0.0"}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{}
!7 = !{i32 12, i32 1, i32 9}
!8 = !{i32 6, i32 1}
!9 = !{!10, !10, i64 0}
!10 = !{!"long", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C++ TBAA"}
!13 = !{!14, !14, i64 0}
!14 = !{!"any pointer", !11, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !11, i64 0}
