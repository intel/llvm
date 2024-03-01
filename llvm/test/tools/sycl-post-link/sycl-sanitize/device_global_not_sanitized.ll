; RUN: sycl-post-link -O0 --device-globals -ir-output-only -S %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%class.anon = type { i8 }
%"class.sycl::_V1::ext::oneapi::experimental::detail::device_global_base" = type { [4 x i32] }

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E4Test = comdat any

$_ZN4sycl3_V13ext6oneapi12experimental13device_globalIA4_iNS3_10propertiesISt5tupleIJNS3_14property_valueINS3_22device_image_scope_keyEJEEENS8_INS3_15host_access_keyEJSt17integral_constantINS3_16host_access_enumELSD_2EEEEEEEEEEixIS5_EERNSt16remove_referenceIDTixclsr3stdE7declvalIT_EEclL_ZSt7declvalIlEDTcl9__declvalISL_ELi0EEEvEEEE4typeEl = comdat any

$_ZN4sycl3_V13ext6oneapi12experimental6detail18device_global_baseIA4_iNS3_10propertiesISt5tupleIJNS3_14property_valueINS3_22device_image_scope_keyEJEEENS9_INS3_15host_access_keyEJSt17integral_constantINS3_16host_access_enumELSE_2EEEEEEEEEvE7get_ptrEv = comdat any

@dev_global = dso_local addrspace(1) global { [4 x i32] } zeroinitializer, align 4, !spirv.Decorations !0 #0
@__spirv_BuiltInWorkgroupId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalLinearId = external dso_local local_unnamed_addr addrspace(1) constant i64, align 8
@__spirv_BuiltInWorkgroupSize = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; CHECK-NOT: @__AsanDeviceGlobalCount
; CHECK-NOT: @__AsanDeviceGlobalMetadata

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E4Test() #1 comdat !srcloc !8 !kernel_arg_buffer_location !9 !sycl_fixed_targets !9 {
entry:
  call spir_func void @__itt_offload_wi_start_wrapper()
  %__SYCLKernel = alloca %class.anon, align 1
  %__SYCLKernel.ascast = addrspacecast ptr %__SYCLKernel to ptr addrspace(4)
  call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(ptr addrspace(4) noundef align 1 dereferenceable_or_null(1) %__SYCLKernel.ascast) #7
  call spir_func void @__itt_offload_wi_finish_wrapper()
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(ptr addrspace(4) noundef align 1 dereferenceable_or_null(1) %this) #2 align 2 !srcloc !8 {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast, align 8
  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast, align 8
  %call = call spir_func noundef align 4 dereferenceable(4) ptr addrspace(4) @_ZN4sycl3_V13ext6oneapi12experimental13device_globalIA4_iNS3_10propertiesISt5tupleIJNS3_14property_valueINS3_22device_image_scope_keyEJEEENS8_INS3_15host_access_keyEJSt17integral_constantINS3_16host_access_enumELSD_2EEEEEEEEEEixIS5_EERNSt16remove_referenceIDTixclsr3stdE7declvalIT_EEclL_ZSt7declvalIlEDTcl9__declvalISL_ELi0EEEvEEEE4typeEl(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) addrspacecast (ptr addrspace(1) @dev_global to ptr addrspace(4)), i64 noundef 0) #7
  store i32 42, ptr addrspace(4) %call, align 4
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func noundef align 4 dereferenceable(4) ptr addrspace(4) @_ZN4sycl3_V13ext6oneapi12experimental13device_globalIA4_iNS3_10propertiesISt5tupleIJNS3_14property_valueINS3_22device_image_scope_keyEJEEENS8_INS3_15host_access_keyEJSt17integral_constantINS3_16host_access_enumELSD_2EEEEEEEEEEixIS5_EERNSt16remove_referenceIDTixclsr3stdE7declvalIT_EEclL_ZSt7declvalIlEDTcl9__declvalISL_ELi0EEEvEEEE4typeEl(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) %this, i64 noundef %idx) #2 comdat align 2 !srcloc !10 {
entry:
  %retval = alloca ptr addrspace(4), align 8
  %this.addr = alloca ptr addrspace(4), align 8
  %idx.addr = alloca i64, align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  %idx.addr.ascast = addrspacecast ptr %idx.addr to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast, align 8
  store i64 %idx, ptr addrspace(4) %idx.addr.ascast, align 8
  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast, align 8
  %call = call spir_func noundef ptr addrspace(4) @_ZN4sycl3_V13ext6oneapi12experimental6detail18device_global_baseIA4_iNS3_10propertiesISt5tupleIJNS3_14property_valueINS3_22device_image_scope_keyEJEEENS9_INS3_15host_access_keyEJSt17integral_constantINS3_16host_access_enumELSE_2EEEEEEEEEvE7get_ptrEv(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) %this1) #7
  %0 = load i64, ptr addrspace(4) %idx.addr.ascast, align 8
  %arrayidx = getelementptr inbounds [4 x i32], ptr addrspace(4) %call, i64 0, i64 %0
  ret ptr addrspace(4) %arrayidx
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func noundef ptr addrspace(4) @_ZN4sycl3_V13ext6oneapi12experimental6detail18device_global_baseIA4_iNS3_10propertiesISt5tupleIJNS3_14property_valueINS3_22device_image_scope_keyEJEEENS9_INS3_15host_access_keyEJSt17integral_constantINS3_16host_access_enumELSE_2EEEEEEEEEvE7get_ptrEv(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) %this) #2 comdat align 2 !srcloc !11 {
entry:
  %retval = alloca ptr addrspace(4), align 8
  %this.addr = alloca ptr addrspace(4), align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast, align 8
  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast, align 8
  %val = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::experimental::detail::device_global_base", ptr addrspace(4) %this1, i32 0, i32 0
  ret ptr addrspace(4) %val
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define dso_local spir_func void @__itt_offload_wi_start_wrapper() #3 !srcloc !12 {
entry:
  %GroupID = alloca [3 x i64], align 8
  %call.i = tail call spir_func signext i8 @__spirv_SpecConstant(i32 noundef -9145239, i8 noundef signext 0) #7
  %cmp.i.not = icmp eq i8 %call.i, 0
  br i1 %cmp.i.not, label %return, label %if.end

if.end:                                           ; preds = %entry
  %GroupID.ascast = addrspacecast ptr %GroupID to ptr addrspace(4)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %GroupID) #8
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  %1 = extractelement <3 x i64> %0, i64 0
  store i64 %1, ptr %GroupID, align 8, !tbaa !13
  %arrayinit.element = getelementptr inbounds i8, ptr %GroupID, i64 8
  %2 = extractelement <3 x i64> %0, i64 1
  store i64 %2, ptr %arrayinit.element, align 8, !tbaa !13
  %arrayinit.element1 = getelementptr inbounds i8, ptr %GroupID, i64 16
  %3 = extractelement <3 x i64> %0, i64 2
  store i64 %3, ptr %arrayinit.element1, align 8, !tbaa !13
  %4 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalLinearId, align 8, !tbaa !13
  %5 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, align 32
  %6 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, i64 8), align 8
  %mul = mul i64 %5, %6
  %7 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, i64 16), align 16
  %mul2 = mul i64 %mul, %7
  %conv = trunc i64 %mul2 to i32
  call spir_func void @__itt_offload_wi_start_stub(ptr addrspace(4) noundef %GroupID.ascast, i64 noundef %4, i32 noundef %conv) #7
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %GroupID) #8
  br label %return

return:                                           ; preds = %entry, %if.end
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local spir_func signext i8 @__spirv_SpecConstant(i32 noundef, i8 noundef signext) local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #5

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define dso_local spir_func void @__itt_offload_wi_finish_wrapper() #3 !srcloc !17 {
entry:
  %GroupID = alloca [3 x i64], align 8
  %call.i = tail call spir_func signext i8 @__spirv_SpecConstant(i32 noundef -9145239, i8 noundef signext 0) #7
  %cmp.i.not = icmp eq i8 %call.i, 0
  br i1 %cmp.i.not, label %return, label %if.end

if.end:                                           ; preds = %entry
  %GroupID.ascast = addrspacecast ptr %GroupID to ptr addrspace(4)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %GroupID) #8
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  %1 = extractelement <3 x i64> %0, i64 0
  store i64 %1, ptr %GroupID, align 8, !tbaa !13
  %arrayinit.element = getelementptr inbounds i8, ptr %GroupID, i64 8
  %2 = extractelement <3 x i64> %0, i64 1
  store i64 %2, ptr %arrayinit.element, align 8, !tbaa !13
  %arrayinit.element1 = getelementptr inbounds i8, ptr %GroupID, i64 16
  %3 = extractelement <3 x i64> %0, i64 2
  store i64 %3, ptr %arrayinit.element1, align 8, !tbaa !13
  %4 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalLinearId, align 8, !tbaa !13
  call spir_func void @__itt_offload_wi_finish_stub(ptr addrspace(4) noundef %GroupID.ascast, i64 noundef %4) #7
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %GroupID) #8
  br label %return

return:                                           ; preds = %entry, %if.end
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_func void @__itt_offload_wi_start_stub(ptr addrspace(4) noundef %group_id, i64 noundef %wi_id, i32 noundef %wg_size) local_unnamed_addr #6 !srcloc !18 {
entry:
  %group_id.addr = alloca ptr addrspace(4), align 8
  %wi_id.addr = alloca i64, align 8
  %wg_size.addr = alloca i32, align 4
  %group_id.addr.ascast = addrspacecast ptr %group_id.addr to ptr addrspace(4)
  %wi_id.addr.ascast = addrspacecast ptr %wi_id.addr to ptr addrspace(4)
  %wg_size.addr.ascast = addrspacecast ptr %wg_size.addr to ptr addrspace(4)
  store ptr addrspace(4) %group_id, ptr addrspace(4) %group_id.addr.ascast, align 8, !tbaa !19
  store i64 %wi_id, ptr addrspace(4) %wi_id.addr.ascast, align 8, !tbaa !13
  store i32 %wg_size, ptr addrspace(4) %wg_size.addr.ascast, align 4, !tbaa !21
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_func void @__itt_offload_wi_finish_stub(ptr addrspace(4) noundef %group_id, i64 noundef %wi_id) local_unnamed_addr #6 !srcloc !23 {
entry:
  %group_id.addr = alloca ptr addrspace(4), align 8
  %wi_id.addr = alloca i64, align 8
  %group_id.addr.ascast = addrspacecast ptr %group_id.addr to ptr addrspace(4)
  %wi_id.addr.ascast = addrspacecast ptr %wi_id.addr to ptr addrspace(4)
  store ptr addrspace(4) %group_id, ptr addrspace(4) %group_id.addr.ascast, align 8, !tbaa !19
  store i64 %wi_id, ptr addrspace(4) %wi_id.addr.ascast, align 8, !tbaa !13
  ret void
}

attributes #0 = { "sycl-device-global-size"="16" "sycl-device-image-scope" "sycl-host-access"="2" "sycl-unique-id"="_Z10dev_global" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="device_global_image_scope.cpp" "sycl-optlevel"="0" "sycl-single-task" "uniform-work-group-size"="true" }
attributes #2 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-optlevel"="0" }
attributes #3 = { alwaysinline convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/export/users/maosuzha/ics_workspace/syclos/libdevice/itt_compiler_wrappers.cpp" "sycl-optlevel"="2" }
attributes #4 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/export/users/maosuzha/ics_workspace/syclos/libdevice/itt_stubs.cpp" "sycl-optlevel"="2" }
attributes #7 = { convergent nounwind }
attributes #8 = { nounwind }

!opencl.spir.version = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!spirv.Source = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
!llvm.ident = !{!4, !5, !5, !5, !5, !5, !5, !4, !5, !5, !5, !5, !5, !5, !5, !5, !4, !5, !5, !5, !5}
!llvm.module.flags = !{!6, !7}

!0 = !{!1}
!1 = !{i32 6147, i32 2, !"_Z10dev_global"}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 19.0.0git (https://github.com/intel/llvm.git 5ec9cce34290bb937846e7ee81dad0100e2dba17)"}
!5 = !{!"clang version 19.0.0git (https://github.com/intel/llvm.git 55c6d2b3751e3b59e9aaf3972f375c33dc0b9d8b)"}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{i32 5139140}
!9 = !{}
!10 = !{i32 3766433}
!11 = !{i32 3763433}
!12 = !{i32 442}
!13 = !{!14, !14, i64 0}
!14 = !{!"long", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C++ TBAA"}
!17 = !{i32 1030}
!18 = !{i32 462}
!19 = !{!20, !20, i64 0}
!20 = !{!"any pointer", !15, i64 0}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !15, i64 0}
!23 = !{i32 592}
