; This test checks that the post-link tool properly generates "asanUsed=1"
; in [SYCL/misc properties]

; RUN: sycl-post-link -split=kernel -symbols -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop
; CHECK: [SYCL/misc properties]
; CHECK: asanUsed=1

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%class.DeviceGlobal = type { i64 }

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E4Test = comdat any

@__spirv_BuiltInGlobalLinearId = external dso_local local_unnamed_addr addrspace(1) constant i64, align 8
@__spirv_BuiltInWorkgroupSize = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__DeviceType = dso_local local_unnamed_addr addrspace(1) global %class.DeviceGlobal zeroinitializer, align 8 #0
@__AsanShadowMemoryGlobalStart = dso_local local_unnamed_addr addrspace(1) global %class.DeviceGlobal zeroinitializer, align 8 #1
@__AsanShadowMemoryGlobalEnd = dso_local local_unnamed_addr addrspace(1) global %class.DeviceGlobal zeroinitializer, align 8 #2
@__spirv_BuiltInWorkgroupId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__AsanShadowMemoryLocalStart = dso_local local_unnamed_addr addrspace(1) global %class.DeviceGlobal zeroinitializer, align 8 #3
@__AsanShadowMemoryLocalEnd = dso_local local_unnamed_addr addrspace(1) global %class.DeviceGlobal zeroinitializer, align 8 #4
@__DeviceSanitizerReportMem = dso_local addrspace(1) global { { i32, [257 x i8], [257 x i8], i32, i64, i64, i64, i64, i64, i64, i8, i32, i32, i32, i8 } } zeroinitializer, align 8 #5

; Function Attrs: mustprogress norecurse nounwind sanitize_address uwtable
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E4Test() local_unnamed_addr #6 comdat !srcloc !7 !kernel_arg_buffer_location !8 !sycl_fixed_targets !8 !sycl_kernel_omit_args !8 {
entry:
  call spir_func void @__itt_offload_wi_start_wrapper()
  call spir_func void @__itt_offload_wi_finish_wrapper()
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #7

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #7

; Function Attrs: convergent nounwind
declare dso_local spir_func signext i8 @__spirv_SpecConstant(i32 noundef, i8 noundef signext) local_unnamed_addr #8

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define dso_local spir_func void @__itt_offload_wi_start_wrapper() #9 !srcloc !9 {
entry:
  %GroupID = alloca [3 x i64], align 8
  %call.i = tail call spir_func signext i8 @__spirv_SpecConstant(i32 noundef -9145239, i8 noundef signext 0) #11
  %cmp.i.not = icmp eq i8 %call.i, 0
  br i1 %cmp.i.not, label %return, label %if.end

if.end:                                           ; preds = %entry
  %GroupID.ascast = addrspacecast ptr %GroupID to ptr addrspace(4)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %GroupID) #12
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  %1 = extractelement <3 x i64> %0, i64 0
  store i64 %1, ptr %GroupID, align 8, !tbaa !10
  %arrayinit.element = getelementptr inbounds i8, ptr %GroupID, i64 8
  %2 = extractelement <3 x i64> %0, i64 1
  store i64 %2, ptr %arrayinit.element, align 8, !tbaa !10
  %arrayinit.element1 = getelementptr inbounds i8, ptr %GroupID, i64 16
  %3 = extractelement <3 x i64> %0, i64 2
  store i64 %3, ptr %arrayinit.element1, align 8, !tbaa !10
  %4 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalLinearId, align 8, !tbaa !10
  %5 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, align 32
  %6 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, i64 8), align 8
  %mul = mul i64 %5, %6
  %7 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, i64 16), align 16
  %mul2 = mul i64 %mul, %7
  %conv = trunc i64 %mul2 to i32
  call spir_func void @__itt_offload_wi_start_stub(ptr addrspace(4) noundef %GroupID.ascast, i64 noundef %4, i32 noundef %conv) #11
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %GroupID) #12
  br label %return

return:                                           ; preds = %if.end, %entry
  ret void
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define dso_local spir_func void @__itt_offload_wi_finish_wrapper() #9 !srcloc !14 {
entry:
  %GroupID = alloca [3 x i64], align 8
  %call.i = tail call spir_func signext i8 @__spirv_SpecConstant(i32 noundef -9145239, i8 noundef signext 0) #11
  %cmp.i.not = icmp eq i8 %call.i, 0
  br i1 %cmp.i.not, label %return, label %if.end

if.end:                                           ; preds = %entry
  %GroupID.ascast = addrspacecast ptr %GroupID to ptr addrspace(4)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %GroupID) #12
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  %1 = extractelement <3 x i64> %0, i64 0
  store i64 %1, ptr %GroupID, align 8, !tbaa !10
  %arrayinit.element = getelementptr inbounds i8, ptr %GroupID, i64 8
  %2 = extractelement <3 x i64> %0, i64 1
  store i64 %2, ptr %arrayinit.element, align 8, !tbaa !10
  %arrayinit.element1 = getelementptr inbounds i8, ptr %GroupID, i64 16
  %3 = extractelement <3 x i64> %0, i64 2
  store i64 %3, ptr %arrayinit.element1, align 8, !tbaa !10
  %4 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalLinearId, align 8, !tbaa !10
  call spir_func void @__itt_offload_wi_finish_stub(ptr addrspace(4) noundef %GroupID.ascast, i64 noundef %4) #11
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %GroupID) #12
  br label %return

return:                                           ; preds = %if.end, %entry
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_func void @__itt_offload_wi_start_stub(ptr addrspace(4) noundef %group_id, i64 noundef %wi_id, i32 noundef %wg_size) local_unnamed_addr #10 !srcloc !15 {
entry:
  %group_id.addr = alloca ptr addrspace(4), align 8
  %wi_id.addr = alloca i64, align 8
  %wg_size.addr = alloca i32, align 4
  %group_id.addr.ascast = addrspacecast ptr %group_id.addr to ptr addrspace(4)
  %wi_id.addr.ascast = addrspacecast ptr %wi_id.addr to ptr addrspace(4)
  %wg_size.addr.ascast = addrspacecast ptr %wg_size.addr to ptr addrspace(4)
  store ptr addrspace(4) %group_id, ptr addrspace(4) %group_id.addr.ascast, align 8, !tbaa !16
  store i64 %wi_id, ptr addrspace(4) %wi_id.addr.ascast, align 8, !tbaa !10
  store i32 %wg_size, ptr addrspace(4) %wg_size.addr.ascast, align 4, !tbaa !18
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_func void @__itt_offload_wi_finish_stub(ptr addrspace(4) noundef %group_id, i64 noundef %wi_id) local_unnamed_addr #10 !srcloc !20 {
entry:
  %group_id.addr = alloca ptr addrspace(4), align 8
  %wi_id.addr = alloca i64, align 8
  %group_id.addr.ascast = addrspacecast ptr %group_id.addr to ptr addrspace(4)
  %wi_id.addr.ascast = addrspacecast ptr %wi_id.addr to ptr addrspace(4)
  store ptr addrspace(4) %group_id, ptr addrspace(4) %group_id.addr.ascast, align 8, !tbaa !16
  store i64 %wi_id, ptr addrspace(4) %wi_id.addr.ascast, align 8, !tbaa !10
  ret void
}

attributes #0 = { "sycl-unique-id"="_Z12__DeviceType" }
attributes #1 = { "sycl-unique-id"="_Z29__AsanShadowMemoryGlobalStart" }
attributes #2 = { "sycl-unique-id"="_Z27__AsanShadowMemoryGlobalEnd" }
attributes #3 = { "sycl-unique-id"="_Z28__AsanShadowMemoryLocalStart" }
attributes #4 = { "sycl-unique-id"="_Z26__AsanShadowMemoryLocalEnd" }
attributes #5 = { "sycl-unique-id"="_Z26__DeviceSanitizerReportMem" }
attributes #6 = { mustprogress norecurse nounwind sanitize_address uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test.cpp" "sycl-optlevel"="2" "sycl-single-task" "uniform-work-group-size"="true" }
attributes #7 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #8 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #9 = { alwaysinline convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/export/users/maosuzha/ics_workspace/syclos/libdevice/itt_compiler_wrappers.cpp" "sycl-optlevel"="2" }
attributes #10 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/export/users/maosuzha/ics_workspace/syclos/libdevice/itt_stubs.cpp" "sycl-optlevel"="2" }
attributes #11 = { convergent nounwind }
attributes #12 = { nounwind }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!llvm.ident = !{!2, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !2}
!llvm.module.flags = !{!4, !5, !6}
!sycl.specialization-constants = !{}
!sycl.specialization-constants-default-values = !{}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 19.0.0git (https://github.com/intel/llvm.git c4308cc8751c15934d154a9a9d5cac8c31a7743a)"}
!3 = !{!"clang version 19.0.0git (https://github.com/intel/llvm.git 55c6d2b3751e3b59e9aaf3972f375c33dc0b9d8b)"}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{i32 5141640}
!8 = !{}
!9 = !{i32 442}
!10 = !{!11, !11, i64 0}
!11 = !{!"long", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C++ TBAA"}
!14 = !{i32 1030}
!15 = !{i32 462}
!16 = !{!17, !17, i64 0}
!17 = !{!"any pointer", !12, i64 0}
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !12, i64 0}
!20 = !{i32 592}
