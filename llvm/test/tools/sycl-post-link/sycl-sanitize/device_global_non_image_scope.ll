; RUN: sycl-post-link -O0 --device-globals -ir-output-only -S %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%class.DeviceGlobal = type { i64 }
%class.anon = type { i8 }
%"class.sycl::_V1::ext::oneapi::experimental::detail::device_global_base" = type { ptr addrspace(1), [4 x i32] }

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E4Test = comdat any

$_ZN4sycl3_V13ext6oneapi12experimental13device_globalIA4_iNS3_10propertiesISt5tupleIJEEEEEixIS5_EERNSt16remove_referenceIDTixclsr3stdE7declvalIT_EEclL_ZSt7declvalIlEDTcl9__declvalISD_ELi0EEEvEEEE4typeEl = comdat any

$_ZN4sycl3_V13ext6oneapi12experimental6detail18device_global_baseIA4_iNS3_10propertiesISt5tupleIJEEEEvE7get_ptrEv = comdat any

$_Z28__spirv_GlobalInvocationId_xv = comdat any

$_Z28__spirv_GlobalInvocationId_yv = comdat any

$_Z28__spirv_GlobalInvocationId_zv = comdat any

$_Z27__spirv_LocalInvocationId_xv = comdat any

$_Z27__spirv_LocalInvocationId_yv = comdat any

$_Z27__spirv_LocalInvocationId_zv = comdat any

@llvm.used = appending global [1 x ptr] [ptr @asan.module_ctor], section "llvm.metadata"
@dev_global = dso_local addrspace(1) global { ptr addrspace(1), [4 x i32] } zeroinitializer, align 8, !spirv.Decorations !0 #0
@__asan_func = internal addrspace(2) constant [107 x i8] c"main::'lambda'(sycl::_V1::handler&)::operator()(sycl::_V1::handler&) const::'lambda'()::operator()() const\00"
@__asan_func.1 = internal addrspace(2) constant [274 x i8] c"std::remove_reference<decltype((std::declval<int [4]>())[decltype(__declval<int [4]>(0)) std::declval<long>()()])>::type& sycl::_V1::ext::oneapi::experimental::device_global<int [4], sycl::_V1::ext::oneapi::experimental::properties<std::tuple<>>>::operator[]<int [4]>(long)\00"
@__asan_func.2 = internal addrspace(2) constant [155 x i8] c"sycl::_V1::ext::oneapi::experimental::detail::device_global_base<int [4], sycl::_V1::ext::oneapi::experimental::properties<std::tuple<>>, void>::get_ptr()\00"
@__spirv_BuiltInGlobalLinearId = external dso_local local_unnamed_addr addrspace(1) constant i64, align 8
@__spirv_BuiltInWorkgroupSize = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__DeviceType = dso_local local_unnamed_addr addrspace(1) global %class.DeviceGlobal zeroinitializer, align 8 #1
@_ZL23__unsupport_device_type = internal addrspace(2) constant [34 x i8] c"ERROR: Unsupport device type: %d\0A\00", align 1
@__AsanShadowMemoryGlobalStart = dso_local local_unnamed_addr addrspace(1) global %class.DeviceGlobal zeroinitializer, align 8 #2
@__AsanShadowMemoryGlobalEnd = dso_local local_unnamed_addr addrspace(1) global %class.DeviceGlobal zeroinitializer, align 8 #3
@_ZL28__global_shadow_out_of_bound = internal addrspace(2) constant [68 x i8] c"ERROR: Global shadow memory out-of-bound (ptr: %p -> %p, base: %p)\0A\00", align 1
@__spirv_BuiltInWorkgroupId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInNumWorkgroups = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__AsanShadowMemoryLocalStart = dso_local local_unnamed_addr addrspace(1) global %class.DeviceGlobal zeroinitializer, align 8 #4
@__AsanShadowMemoryLocalEnd = dso_local local_unnamed_addr addrspace(1) global %class.DeviceGlobal zeroinitializer, align 8 #5
@_ZL27__local_shadow_out_of_bound = internal addrspace(2) constant [75 x i8] c"ERROR: Local shadow memory out-of-bound (ptr: %p -> %p, wg: %d, base: %p)\0A\00", align 1
@__DeviceSanitizerReportMem = dso_local addrspace(1) global { { i32, [257 x i8], [257 x i8], i32, i64, i64, i64, i64, i64, i64, i8, i32, i32, i32, i8 } } zeroinitializer, align 8 #6
@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; CHECK-NOT: @__AsanDeviceGlobalCount
; CHECK-NOT: @__AsanDeviceGlobalMetadata

; Function Attrs: nounwind uwtable
define internal void @asan.module_ctor() #7 {
  call void @__asan_init()
  call void @__asan_version_mismatch_check_v8()
  ret void
}

declare void @__asan_init()

declare void @__asan_version_mismatch_check_v8()

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone sanitize_address uwtable
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E4Test() #8 comdat !srcloc !9 !kernel_arg_buffer_location !10 !sycl_fixed_targets !10 {
entry:
  call spir_func void @__itt_offload_wi_start_wrapper()
  %__SYCLKernel = alloca %class.anon, align 1
  %__SYCLKernel.ascast = addrspacecast ptr %__SYCLKernel to ptr addrspace(4)
  call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(ptr addrspace(4) noundef align 1 dereferenceable_or_null(1) %__SYCLKernel.ascast) #20
  call spir_func void @__itt_offload_wi_finish_wrapper()
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone sanitize_address uwtable
define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(ptr addrspace(4) noundef align 1 dereferenceable_or_null(1) %this) #9 align 2 !srcloc !9 {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  %0 = ptrtoint ptr addrspace(4) %this.addr.ascast to i64
  call void @__asan_store8(i64 %0, i32 4, ptr addrspace(2) null, i32 0, ptr addrspace(2) @__asan_func)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast, align 8
  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast, align 8
  %call = call spir_func noundef align 4 dereferenceable(4) ptr addrspace(4) @_ZN4sycl3_V13ext6oneapi12experimental13device_globalIA4_iNS3_10propertiesISt5tupleIJEEEEEixIS5_EERNSt16remove_referenceIDTixclsr3stdE7declvalIT_EEclL_ZSt7declvalIlEDTcl9__declvalISD_ELi0EEEvEEEE4typeEl(ptr addrspace(4) noundef align 8 dereferenceable_or_null(24) addrspacecast (ptr addrspace(1) @dev_global to ptr addrspace(4)), i64 noundef 0) #20
  %1 = ptrtoint ptr addrspace(4) %call to i64
  call void @__asan_store4(i64 %1, i32 4, ptr addrspace(2) null, i32 0, ptr addrspace(2) @__asan_func)
  store i32 42, ptr addrspace(4) %call, align 4
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone sanitize_address uwtable
define linkonce_odr dso_local spir_func noundef align 4 dereferenceable(4) ptr addrspace(4) @_ZN4sycl3_V13ext6oneapi12experimental13device_globalIA4_iNS3_10propertiesISt5tupleIJEEEEEixIS5_EERNSt16remove_referenceIDTixclsr3stdE7declvalIT_EEclL_ZSt7declvalIlEDTcl9__declvalISD_ELi0EEEvEEEE4typeEl(ptr addrspace(4) noundef align 8 dereferenceable_or_null(24) %this, i64 noundef %idx) #9 comdat align 2 !srcloc !11 {
entry:
  %retval = alloca ptr addrspace(4), align 8
  %this.addr = alloca ptr addrspace(4), align 8
  %idx.addr = alloca i64, align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  %idx.addr.ascast = addrspacecast ptr %idx.addr to ptr addrspace(4)
  %0 = ptrtoint ptr addrspace(4) %this.addr.ascast to i64
  call void @__asan_store8(i64 %0, i32 4, ptr addrspace(2) null, i32 0, ptr addrspace(2) @__asan_func.1)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast, align 8
  %1 = ptrtoint ptr addrspace(4) %idx.addr.ascast to i64
  call void @__asan_store8(i64 %1, i32 4, ptr addrspace(2) null, i32 0, ptr addrspace(2) @__asan_func.1)
  store i64 %idx, ptr addrspace(4) %idx.addr.ascast, align 8
  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast, align 8
  %call = call spir_func noundef ptr addrspace(1) @_ZN4sycl3_V13ext6oneapi12experimental6detail18device_global_baseIA4_iNS3_10propertiesISt5tupleIJEEEEvE7get_ptrEv(ptr addrspace(4) noundef align 8 dereferenceable_or_null(24) %this1) #20
  %2 = ptrtoint ptr addrspace(4) %idx.addr.ascast to i64
  call void @__asan_load8(i64 %2, i32 4, ptr addrspace(2) null, i32 0, ptr addrspace(2) @__asan_func.1)
  %3 = load i64, ptr addrspace(4) %idx.addr.ascast, align 8
  %arrayidx = getelementptr inbounds [4 x i32], ptr addrspace(1) %call, i64 0, i64 %3
  %arrayidx.ascast = addrspacecast ptr addrspace(1) %arrayidx to ptr addrspace(4)
  ret ptr addrspace(4) %arrayidx.ascast
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone sanitize_address uwtable
define linkonce_odr dso_local spir_func noundef ptr addrspace(1) @_ZN4sycl3_V13ext6oneapi12experimental6detail18device_global_baseIA4_iNS3_10propertiesISt5tupleIJEEEEvE7get_ptrEv(ptr addrspace(4) noundef align 8 dereferenceable_or_null(24) %this) #9 comdat align 2 !srcloc !12 {
entry:
  %retval = alloca ptr addrspace(1), align 8
  %this.addr = alloca ptr addrspace(4), align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  %0 = ptrtoint ptr addrspace(4) %this.addr.ascast to i64
  call void @__asan_store8(i64 %0, i32 4, ptr addrspace(2) null, i32 0, ptr addrspace(2) @__asan_func.2)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast, align 8
  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast, align 8
  %usmptr = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::experimental::detail::device_global_base", ptr addrspace(4) %this1, i32 0, i32 0
  %1 = ptrtoint ptr addrspace(4) %usmptr to i64
  call void @__asan_load8(i64 %1, i32 4, ptr addrspace(2) null, i32 0, ptr addrspace(2) @__asan_func.2)
  %2 = load ptr addrspace(1), ptr addrspace(4) %usmptr, align 8
  ret ptr addrspace(1) %2
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define dso_local spir_func void @__itt_offload_wi_start_wrapper() #10 !srcloc !13 {
entry:
  %GroupID = alloca [3 x i64], align 8
  %call.i = tail call spir_func signext i8 @__spirv_SpecConstant(i32 noundef -9145239, i8 noundef signext 0) #20
  %cmp.i.not = icmp eq i8 %call.i, 0
  br i1 %cmp.i.not, label %return, label %if.end

if.end:                                           ; preds = %entry
  %GroupID.ascast = addrspacecast ptr %GroupID to ptr addrspace(4)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %GroupID) #21
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  %1 = extractelement <3 x i64> %0, i64 0
  store i64 %1, ptr %GroupID, align 8, !tbaa !14
  %arrayinit.element = getelementptr inbounds i8, ptr %GroupID, i64 8
  %2 = extractelement <3 x i64> %0, i64 1
  store i64 %2, ptr %arrayinit.element, align 8, !tbaa !14
  %arrayinit.element1 = getelementptr inbounds i8, ptr %GroupID, i64 16
  %3 = extractelement <3 x i64> %0, i64 2
  store i64 %3, ptr %arrayinit.element1, align 8, !tbaa !14
  %4 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalLinearId, align 8, !tbaa !14
  %5 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, align 32
  %6 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, i64 8), align 8
  %mul = mul i64 %5, %6
  %7 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInWorkgroupSize, i64 16), align 16
  %mul2 = mul i64 %mul, %7
  %conv = trunc i64 %mul2 to i32
  call spir_func void @__itt_offload_wi_start_stub(ptr addrspace(4) noundef %GroupID.ascast, i64 noundef %4, i32 noundef %conv) #20
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %GroupID) #21
  br label %return

return:                                           ; preds = %entry, %if.end
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local spir_func signext i8 @__spirv_SpecConstant(i32 noundef, i8 noundef signext) local_unnamed_addr #11

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #12

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #12

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define dso_local spir_func void @__itt_offload_wi_finish_wrapper() #10 !srcloc !18 {
entry:
  %GroupID = alloca [3 x i64], align 8
  %call.i = tail call spir_func signext i8 @__spirv_SpecConstant(i32 noundef -9145239, i8 noundef signext 0) #20
  %cmp.i.not = icmp eq i8 %call.i, 0
  br i1 %cmp.i.not, label %return, label %if.end

if.end:                                           ; preds = %entry
  %GroupID.ascast = addrspacecast ptr %GroupID to ptr addrspace(4)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %GroupID) #21
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  %1 = extractelement <3 x i64> %0, i64 0
  store i64 %1, ptr %GroupID, align 8, !tbaa !14
  %arrayinit.element = getelementptr inbounds i8, ptr %GroupID, i64 8
  %2 = extractelement <3 x i64> %0, i64 1
  store i64 %2, ptr %arrayinit.element, align 8, !tbaa !14
  %arrayinit.element1 = getelementptr inbounds i8, ptr %GroupID, i64 16
  %3 = extractelement <3 x i64> %0, i64 2
  store i64 %3, ptr %arrayinit.element1, align 8, !tbaa !14
  %4 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalLinearId, align 8, !tbaa !14
  call spir_func void @__itt_offload_wi_finish_stub(ptr addrspace(4) noundef %GroupID.ascast, i64 noundef %4) #20
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %GroupID) #21
  br label %return

return:                                           ; preds = %entry, %if.end
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_func void @__itt_offload_wi_start_stub(ptr addrspace(4) noundef %group_id, i64 noundef %wi_id, i32 noundef %wg_size) local_unnamed_addr #13 !srcloc !19 {
entry:
  %group_id.addr = alloca ptr addrspace(4), align 8
  %wi_id.addr = alloca i64, align 8
  %wg_size.addr = alloca i32, align 4
  %group_id.addr.ascast = addrspacecast ptr %group_id.addr to ptr addrspace(4)
  %wi_id.addr.ascast = addrspacecast ptr %wi_id.addr to ptr addrspace(4)
  %wg_size.addr.ascast = addrspacecast ptr %wg_size.addr to ptr addrspace(4)
  store ptr addrspace(4) %group_id, ptr addrspace(4) %group_id.addr.ascast, align 8, !tbaa !20
  store i64 %wi_id, ptr addrspace(4) %wi_id.addr.ascast, align 8, !tbaa !14
  store i32 %wg_size, ptr addrspace(4) %wg_size.addr.ascast, align 4, !tbaa !22
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_func void @__itt_offload_wi_finish_stub(ptr addrspace(4) noundef %group_id, i64 noundef %wi_id) local_unnamed_addr #13 !srcloc !24 {
entry:
  %group_id.addr = alloca ptr addrspace(4), align 8
  %wi_id.addr = alloca i64, align 8
  %group_id.addr.ascast = addrspacecast ptr %group_id.addr to ptr addrspace(4)
  %wi_id.addr.ascast = addrspacecast ptr %wi_id.addr to ptr addrspace(4)
  store ptr addrspace(4) %group_id, ptr addrspace(4) %group_id.addr.ascast, align 8, !tbaa !20
  store i64 %wi_id, ptr addrspace(4) %wi_id.addr.ascast, align 8, !tbaa !14
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define weak dso_local spir_func void @__asan_store4(i64 noundef %addr, i32 noundef %as, ptr addrspace(2) noundef %file, i32 noundef %line, ptr addrspace(2) noundef %func) #14 !srcloc !25 {
entry:
  %call.i = tail call spir_func noundef i64 @_ZN12_GLOBAL__N_111MemToShadowEmi(i64 noundef %addr, i32 noundef %as) #20
  %tobool.not.i = icmp eq i64 %call.i, 0
  br i1 %tobool.not.i, label %if.end, label %if.then.i

if.then.i:                                        ; preds = %entry
  %0 = inttoptr i64 %call.i to ptr addrspace(4)
  %1 = load i8, ptr addrspace(4) %0, align 1, !tbaa !26
  %tobool1.not.i = icmp eq i8 %1, 0
  br i1 %tobool1.not.i, label %if.end, label %_Z26__asan_address_is_poisonedmim.exit

_Z26__asan_address_is_poisonedmim.exit:           ; preds = %if.then.i
  %2 = trunc i64 %addr to i32
  %3 = shl i32 %2, 24
  %4 = and i32 %3, 117440512
  %sext.i = add nuw nsw i32 %4, 50331648
  %conv3.i = lshr exact i32 %sext.i, 24
  %conv4.i = sext i8 %1 to i32
  %cmp.i.not = icmp slt i32 %conv3.i, %conv4.i
  br i1 %cmp.i.not, label %if.end, label %if.then

if.then:                                          ; preds = %_Z26__asan_address_is_poisonedmim.exit
  tail call spir_func void @_Z26__asan_report_access_errormimbmPU3AS2KciS0_b(i64 noundef %addr, i32 noundef %as, i64 noundef 4, i1 noundef zeroext true, i64 noundef %addr, ptr addrspace(2) noundef %file, i32 noundef %line, ptr addrspace(2) noundef %func, i1 noundef zeroext false) #20
  br label %if.end

if.end:                                           ; preds = %if.then.i, %entry, %if.then, %_Z26__asan_address_is_poisonedmim.exit
  ret void
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define internal spir_func noundef i64 @_ZN12_GLOBAL__N_111MemToShadowEmi(i64 noundef %addr, i32 noundef %as) unnamed_addr #15 !srcloc !27 !sycl_kernel_omit_args !28 {
entry:
  %0 = load i64, ptr addrspace(1) @__DeviceType, align 8, !tbaa !29
  switch i64 %0, label %if.else6 [
    i64 1, label %if.then
    i64 2, label %if.then4
  ]

if.then:                                          ; preds = %entry
  %1 = load i64, ptr addrspace(1) @__AsanShadowMemoryGlobalStart, align 8, !tbaa !14
  %shr.i = lshr i64 %addr, 3
  %add.i = add i64 %1, %shr.i
  br label %cleanup

if.then4:                                         ; preds = %entry
  switch i32 %as, label %cleanup [
    i32 4, label %if.then.i
    i32 3, label %if.then40.i
    i32 1, label %if.then17.i
  ]

if.then.i:                                        ; preds = %if.then4
  %2 = inttoptr i64 %addr to ptr addrspace(4)
  %call.i.i = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef %2, i32 noundef 5) #22
  %tobool.not.i = icmp eq ptr addrspace(1) %call.i.i, null
  br i1 %tobool.not.i, label %if.else.i, label %if.then17.i

if.else.i:                                        ; preds = %if.then.i
  %call.i82.i = tail call spir_func noundef ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4) noundef %2, i32 noundef 7) #22
  %tobool3.not.i = icmp eq ptr %call.i82.i, null
  br i1 %tobool3.not.i, label %if.else5.i, label %if.end12.thread.i

if.end12.thread.i:                                ; preds = %if.else.i
  %3 = ptrtoint ptr %call.i82.i to i64
  br label %cleanup

if.else5.i:                                       ; preds = %if.else.i
  %call.i83.i = tail call spir_func noundef ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4) noundef %2, i32 noundef 4) #22
  %tobool7.not.i = icmp eq ptr addrspace(3) %call.i83.i, null
  br i1 %tobool7.not.i, label %cleanup, label %if.then40.i

if.then17.i:                                      ; preds = %if.then.i, %if.then4
  %tobool18.not.i = icmp ult i64 %addr, 72057594037927936
  br i1 %tobool18.not.i, label %if.else23.i, label %if.then19.i

if.then19.i:                                      ; preds = %if.then17.i
  %4 = load i64, ptr addrspace(1) @__AsanShadowMemoryGlobalStart, align 8, !tbaa !14
  %add.i13 = add i64 %4, 35184372088832
  %and21.i = lshr i64 %addr, 3
  %shr.i14 = and i64 %and21.i, 35184372088831
  %add22.i = add i64 %add.i13, %shr.i14
  br label %if.end28.i

if.else23.i:                                      ; preds = %if.then17.i
  %5 = load i64, ptr addrspace(1) @__AsanShadowMemoryGlobalStart, align 8, !tbaa !14
  %and25.i = lshr i64 %addr, 3
  %shr26.i = and i64 %and25.i, 17592186044415
  %add27.i = add i64 %5, %shr26.i
  br label %if.end28.i

if.end28.i:                                       ; preds = %if.else23.i, %if.then19.i
  %6 = phi i64 [ %4, %if.then19.i ], [ %5, %if.else23.i ]
  %shadow_ptr.1.i = phi i64 [ %add22.i, %if.then19.i ], [ %add27.i, %if.else23.i ]
  %7 = load i64, ptr addrspace(1) @__AsanShadowMemoryGlobalEnd, align 8, !tbaa !14
  %cmp30.i = icmp ugt i64 %shadow_ptr.1.i, %7
  br i1 %cmp30.i, label %if.then31.i, label %cleanup

if.then31.i:                                      ; preds = %if.end28.i
  %call33.i = tail call spir_func noundef i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) noundef @_ZL28__global_shadow_out_of_bound, i64 noundef %addr, i64 noundef %shadow_ptr.1.i, i64 noundef %6) #20
  br label %cleanup

if.then40.i:                                      ; preds = %if.else5.i, %if.then4
  %8 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32
  %9 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInNumWorkgroups, i64 8), align 8
  %mul.i = mul i64 %8, %9
  %10 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInNumWorkgroups, i64 16), align 16
  %11 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, i64 8), align 8
  %mul4181.i = add i64 %mul.i, %11
  %add43.i = mul i64 %mul4181.i, %10
  %12 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, i64 16), align 16
  %add44.i = add i64 %add43.i, %12
  %13 = load i64, ptr addrspace(1) @__AsanShadowMemoryLocalStart, align 8, !tbaa !14
  %14 = shl i64 %add44.i, 14
  %shr47.i = and i64 %14, 2305843009213677568
  %add48.i = add i64 %13, %shr47.i
  %and49.i = lshr i64 %addr, 3
  %shr50.i = and i64 %and49.i, 16383
  %add51.i = add i64 %add48.i, %shr50.i
  %15 = load i64, ptr addrspace(1) @__AsanShadowMemoryLocalEnd, align 8, !tbaa !14
  %cmp53.i = icmp ugt i64 %add51.i, %15
  br i1 %cmp53.i, label %if.then54.i, label %cleanup

if.then54.i:                                      ; preds = %if.then40.i
  %call56.i = tail call spir_func noundef i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) noundef @_ZL27__local_shadow_out_of_bound, i64 noundef %addr, i64 noundef %add51.i, i64 noundef %add44.i, i64 noundef %13) #20
  br label %cleanup

if.else6:                                         ; preds = %entry
  %conv = trunc i64 %0 to i32
  %call8 = tail call spir_func noundef i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) noundef @_ZL23__unsupport_device_type, i32 noundef %conv) #20
  br label %cleanup

cleanup:                                          ; preds = %if.then54.i, %if.then40.i, %if.then31.i, %if.end28.i, %if.else5.i, %if.end12.thread.i, %if.then4, %if.then, %if.else6
  %retval.0 = phi i64 [ 0, %if.else6 ], [ %add.i, %if.then ], [ 0, %if.else5.i ], [ 0, %if.then31.i ], [ %shadow_ptr.1.i, %if.end28.i ], [ 0, %if.then54.i ], [ %add51.i, %if.then40.i ], [ %3, %if.end12.thread.i ], [ 0, %if.then4 ]
  ret i64 %retval.0
}

; Function Attrs: convergent mustprogress norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z26__asan_report_access_errormimbmPU3AS2KciS0_b(i64 noundef %addr, i32 noundef %as, i64 noundef %size, i1 noundef zeroext %is_write, i64 noundef %poisoned_addr, ptr addrspace(2) noundef %file, i32 noundef %line, ptr addrspace(2) noundef %func, i1 noundef zeroext %is_recover) local_unnamed_addr #16 !srcloc !31 {
entry:
  %call = tail call spir_func noundef i64 @_ZN12_GLOBAL__N_111MemToShadowEmi(i64 noundef %poisoned_addr, i32 noundef %as) #20
  %0 = inttoptr i64 %call to ptr addrspace(4)
  %1 = load i8, ptr addrspace(4) %0, align 1, !tbaa !26
  %cmp = icmp sgt i8 %1, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %add.ptr = getelementptr inbounds i8, ptr addrspace(4) %0, i64 1
  %2 = load i8, ptr addrspace(4) %add.ptr, align 1, !tbaa !26
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %shadow_value.0.in = phi i8 [ %2, %if.then ], [ %1, %entry ]
  switch i8 %shadow_value.0.in, label %sw.default [
    i8 -127, label %sw.epilog
    i8 -126, label %sw.bb3
    i8 -125, label %sw.bb4
    i8 -111, label %sw.bb5
    i8 -110, label %sw.bb6
    i8 -109, label %sw.bb7
    i8 -15, label %sw.bb8
    i8 -14, label %sw.bb8
    i8 -13, label %sw.bb8
    i8 -124, label %sw.bb9
    i8 -95, label %sw.bb10
    i8 -123, label %sw.bb11
  ]

sw.bb3:                                           ; preds = %if.end
  br label %sw.epilog

sw.bb4:                                           ; preds = %if.end
  br label %sw.epilog

sw.bb5:                                           ; preds = %if.end
  br label %sw.epilog

sw.bb6:                                           ; preds = %if.end
  br label %sw.epilog

sw.bb7:                                           ; preds = %if.end
  br label %sw.epilog

sw.bb8:                                           ; preds = %if.end, %if.end, %if.end
  br label %sw.epilog

sw.bb9:                                           ; preds = %if.end
  br label %sw.epilog

sw.bb10:                                          ; preds = %if.end
  br label %sw.epilog

sw.bb11:                                          ; preds = %if.end
  br label %sw.epilog

sw.default:                                       ; preds = %if.end
  br label %sw.epilog

sw.epilog:                                        ; preds = %if.end, %sw.default, %sw.bb11, %sw.bb10, %sw.bb9, %sw.bb8, %sw.bb7, %sw.bb6, %sw.bb5, %sw.bb4, %sw.bb3
  %memory_type.0 = phi i32 [ 0, %sw.default ], [ 7, %sw.bb11 ], [ 4, %sw.bb10 ], [ 6, %sw.bb9 ], [ 5, %sw.bb8 ], [ 3, %sw.bb7 ], [ 2, %sw.bb6 ], [ 1, %sw.bb5 ], [ 3, %sw.bb4 ], [ 2, %sw.bb3 ], [ 1, %if.end ]
  %error_type.0 = phi i32 [ 0, %sw.default ], [ 1, %sw.bb11 ], [ 1, %sw.bb10 ], [ 1, %sw.bb9 ], [ 1, %sw.bb8 ], [ 3, %sw.bb7 ], [ 3, %sw.bb6 ], [ 3, %sw.bb5 ], [ 1, %sw.bb4 ], [ 1, %sw.bb3 ], [ 1, %if.end ]
  %conv12 = trunc i64 %size to i32
  %frombool.i = zext i1 %is_write to i8
  %frombool1.i = zext i1 %is_recover to i8
  %call.i.i = tail call spir_func noundef i32 @_Z29__spirv_AtomicCompareExchangePiN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagES4_ii(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @__DeviceSanitizerReportMem to ptr addrspace(4)), i32 noundef 1, i32 noundef 16, i32 noundef 16, i32 noundef 1, i32 noundef 0) #20
  %cmp.i = icmp eq i32 %call.i.i, 0
  br i1 %cmp.i, label %if.then.i, label %_ZL27__asan_internal_report_savemiPU3AS2KciS0_bj25DeviceSanitizerMemoryType24DeviceSanitizerErrorTypeb.exit

if.then.i:                                        ; preds = %sw.epilog
  %tobool.not.i = icmp eq ptr addrspace(2) %file, null
  br i1 %tobool.not.i, label %if.end.i, label %for.cond.i

for.cond.i:                                       ; preds = %if.then.i, %for.inc.i
  %FileLength.0.i = phi i32 [ %inc.i, %for.inc.i ], [ 0, %if.then.i ]
  %C.0.i = phi ptr addrspace(2) [ %incdec.ptr.i, %for.inc.i ], [ %file, %if.then.i ]
  %3 = load i8, ptr addrspace(2) %C.0.i, align 1, !tbaa !26
  %cmp4.not.i = icmp eq i8 %3, 0
  br i1 %cmp4.not.i, label %if.end.i, label %for.inc.i

for.inc.i:                                        ; preds = %for.cond.i
  %incdec.ptr.i = getelementptr inbounds i8, ptr addrspace(2) %C.0.i, i64 1
  %inc.i = add nuw nsw i32 %FileLength.0.i, 1
  br label %for.cond.i, !llvm.loop !32

if.end.i:                                         ; preds = %for.cond.i, %if.then.i
  %FileLength.1.i = phi i32 [ 0, %if.then.i ], [ %FileLength.0.i, %for.cond.i ]
  %tobool5.not.i = icmp eq ptr addrspace(2) %func, null
  br i1 %tobool5.not.i, label %if.end18.thread.i, label %for.cond8.i

if.end18.thread.i:                                ; preds = %if.end.i
  %spec.select105.i = tail call i32 @llvm.umin.i32(i32 %FileLength.1.i, i32 256)
  br label %5

for.cond8.i:                                      ; preds = %if.end.i, %for.inc14.i
  %FuncLength.0.i = phi i32 [ %inc16.i, %for.inc14.i ], [ 0, %if.end.i ]
  %C7.0.i = phi ptr addrspace(2) [ %incdec.ptr15.i, %for.inc14.i ], [ %func, %if.end.i ]
  %4 = load i8, ptr addrspace(2) %C7.0.i, align 1, !tbaa !26
  %cmp10.not.i = icmp eq i8 %4, 0
  br i1 %cmp10.not.i, label %if.end18.i, label %for.inc14.i

for.inc14.i:                                      ; preds = %for.cond8.i
  %incdec.ptr15.i = getelementptr inbounds i8, ptr addrspace(2) %C7.0.i, i64 1
  %inc16.i = add i32 %FuncLength.0.i, 1
  br label %for.cond8.i, !llvm.loop !34

if.end18.i:                                       ; preds = %for.cond8.i
  %spec.select.i = tail call i32 @llvm.umin.i32(i32 %FileLength.1.i, i32 256)
  %spec.select111.i = tail call i32 @llvm.umin.i32(i32 %FuncLength.0.i, i32 256)
  br label %5

5:                                                ; preds = %if.end18.i, %if.end18.thread.i
  %spec.select109.i = phi i32 [ %spec.select105.i, %if.end18.thread.i ], [ %spec.select.i, %if.end18.i ]
  %6 = phi i32 [ 0, %if.end18.thread.i ], [ %spec.select111.i, %if.end18.i ]
  br label %for.cond25.i

for.cond25.i:                                     ; preds = %for.body29.i, %5
  %Idx.0.i = phi i32 [ 0, %5 ], [ %inc34.i, %for.body29.i ]
  %cmp26.i = icmp ult i32 %Idx.0.i, %spec.select109.i
  br i1 %cmp26.i, label %for.body29.i, label %for.cond.cleanup27.i

for.cond.cleanup27.i:                             ; preds = %for.cond25.i
  %idxprom38.i = zext nneg i32 %spec.select109.i to i64
  %arrayidx39.i = getelementptr inbounds [257 x i8], ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 4), i64 0, i64 %idxprom38.i
  store i8 0, ptr addrspace(1) %arrayidx39.i, align 1, !tbaa !26
  br label %for.cond41.i

for.body29.i:                                     ; preds = %for.cond25.i
  %idxprom.i = zext nneg i32 %Idx.0.i to i64
  %arrayidx.i = getelementptr inbounds i8, ptr addrspace(2) %file, i64 %idxprom.i
  %7 = load i8, ptr addrspace(2) %arrayidx.i, align 1, !tbaa !26
  %arrayidx32.i = getelementptr inbounds [257 x i8], ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 4), i64 0, i64 %idxprom.i
  store i8 %7, ptr addrspace(1) %arrayidx32.i, align 1, !tbaa !26
  %inc34.i = add nuw nsw i32 %Idx.0.i, 1
  br label %for.cond25.i, !llvm.loop !35

for.cond41.i:                                     ; preds = %for.body45.i, %for.cond.cleanup27.i
  %Idx40.0.i = phi i32 [ 0, %for.cond.cleanup27.i ], [ %inc52.i, %for.body45.i ]
  %cmp42.i = icmp ult i32 %Idx40.0.i, %6
  br i1 %cmp42.i, label %for.body45.i, label %for.cond.cleanup43.i

for.cond.cleanup43.i:                             ; preds = %for.cond41.i
  %idxprom56.i = zext nneg i32 %6 to i64
  %arrayidx57.i = getelementptr inbounds [257 x i8], ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 261), i64 0, i64 %idxprom56.i
  store i8 0, ptr addrspace(1) %arrayidx57.i, align 1, !tbaa !26
  store i32 %line, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 520), align 8, !tbaa !36
  %call59.i = tail call spir_func noundef i64 @_Z28__spirv_GlobalInvocationId_xv() #21
  store i64 %call59.i, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 528), align 8, !tbaa !41
  %call61.i = tail call spir_func noundef i64 @_Z28__spirv_GlobalInvocationId_yv() #21
  store i64 %call61.i, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 536), align 8, !tbaa !42
  %call63.i = tail call spir_func noundef i64 @_Z28__spirv_GlobalInvocationId_zv() #21
  store i64 %call63.i, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 544), align 8, !tbaa !43
  %call65.i = tail call spir_func noundef i64 @_Z27__spirv_LocalInvocationId_xv() #21
  store i64 %call65.i, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 552), align 8, !tbaa !44
  %call67.i = tail call spir_func noundef i64 @_Z27__spirv_LocalInvocationId_yv() #21
  store i64 %call67.i, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 560), align 8, !tbaa !45
  %call69.i = tail call spir_func noundef i64 @_Z27__spirv_LocalInvocationId_zv() #21
  store i64 %call69.i, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 568), align 8, !tbaa !46
  store i8 %frombool.i, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 576), align 8, !tbaa !47
  store i32 %conv12, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 580), align 4, !tbaa !48
  store i32 %error_type.0, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 588), align 4, !tbaa !49
  store i32 %memory_type.0, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 584), align 8, !tbaa !50
  store i8 %frombool1.i, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 592), align 8, !tbaa !51
  tail call spir_func void @_Z19__spirv_AtomicStorePiN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEi(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @__DeviceSanitizerReportMem to ptr addrspace(4)), i32 noundef 1, i32 noundef 16, i32 noundef 2) #20
  br label %_ZL27__asan_internal_report_savemiPU3AS2KciS0_bj25DeviceSanitizerMemoryType24DeviceSanitizerErrorTypeb.exit

for.body45.i:                                     ; preds = %for.cond41.i
  %idxprom46.i = zext nneg i32 %Idx40.0.i to i64
  %arrayidx47.i = getelementptr inbounds i8, ptr addrspace(2) %func, i64 %idxprom46.i
  %8 = load i8, ptr addrspace(2) %arrayidx47.i, align 1, !tbaa !26
  %arrayidx50.i = getelementptr inbounds [257 x i8], ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__DeviceSanitizerReportMem, i64 261), i64 0, i64 %idxprom46.i
  store i8 %8, ptr addrspace(1) %arrayidx50.i, align 1, !tbaa !26
  %inc52.i = add nuw nsw i32 %Idx40.0.i, 1
  br label %for.cond41.i, !llvm.loop !52

_ZL27__asan_internal_report_savemiPU3AS2KciS0_bj25DeviceSanitizerMemoryType24DeviceSanitizerErrorTypeb.exit: ; preds = %sw.epilog, %for.cond.cleanup43.i
  ret void
}

; Function Attrs: convergent nounwind
declare extern_weak dso_local spir_func noundef i32 @_Z29__spirv_AtomicCompareExchangePiN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagES4_ii(ptr addrspace(4) noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #11

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #17

; Function Attrs: inlinehint mustprogress norecurse nounwind
define weak dso_local spir_func noundef i64 @_Z28__spirv_GlobalInvocationId_xv() local_unnamed_addr #18 comdat !srcloc !53 {
entry:
  %0 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  ret i64 %0
}

; Function Attrs: inlinehint mustprogress norecurse nounwind
define weak dso_local spir_func noundef i64 @_Z28__spirv_GlobalInvocationId_yv() local_unnamed_addr #18 comdat !srcloc !54 {
entry:
  %0 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, i64 8), align 8
  ret i64 %0
}

; Function Attrs: inlinehint mustprogress norecurse nounwind
define weak dso_local spir_func noundef i64 @_Z28__spirv_GlobalInvocationId_zv() local_unnamed_addr #18 comdat !srcloc !55 {
entry:
  %0 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, i64 16), align 16
  ret i64 %0
}

; Function Attrs: inlinehint mustprogress norecurse nounwind
define weak dso_local spir_func noundef i64 @_Z27__spirv_LocalInvocationId_xv() local_unnamed_addr #18 comdat !srcloc !56 {
entry:
  %0 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32
  ret i64 %0
}

; Function Attrs: inlinehint mustprogress norecurse nounwind
define weak dso_local spir_func noundef i64 @_Z27__spirv_LocalInvocationId_yv() local_unnamed_addr #18 comdat !srcloc !57 {
entry:
  %0 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, i64 8), align 8
  ret i64 %0
}

; Function Attrs: inlinehint mustprogress norecurse nounwind
define weak dso_local spir_func noundef i64 @_Z27__spirv_LocalInvocationId_zv() local_unnamed_addr #18 comdat !srcloc !58 {
entry:
  %0 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, i64 16), align 16
  ret i64 %0
}

; Function Attrs: convergent nounwind
declare extern_weak dso_local spir_func void @_Z19__spirv_AtomicStorePiN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEi(ptr addrspace(4) noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #11

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef, i32 noundef) local_unnamed_addr #19

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local spir_func noundef ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4) noundef, i32 noundef) local_unnamed_addr #19

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local spir_func noundef ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4) noundef, i32 noundef) local_unnamed_addr #19

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) noundef, ...) local_unnamed_addr #11

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define weak dso_local spir_func void @__asan_load8(i64 noundef %addr, i32 noundef %as, ptr addrspace(2) noundef %file, i32 noundef %line, ptr addrspace(2) noundef %func) #14 !srcloc !59 {
entry:
  %call = tail call spir_func noundef i64 @_ZN12_GLOBAL__N_111MemToShadowEmi(i64 noundef %addr, i32 noundef %as) #20
  %tobool.not = icmp eq i64 %call, 0
  br i1 %tobool.not, label %if.end, label %land.lhs.true

land.lhs.true:                                    ; preds = %entry
  %0 = inttoptr i64 %call to ptr addrspace(4)
  %1 = load i8, ptr addrspace(4) %0, align 1, !tbaa !26
  %tobool1.not = icmp eq i8 %1, 0
  br i1 %tobool1.not, label %if.end, label %if.then

if.then:                                          ; preds = %land.lhs.true
  tail call spir_func void @_Z26__asan_report_access_errormimbmPU3AS2KciS0_b(i64 noundef %addr, i32 noundef %as, i64 noundef 8, i1 noundef zeroext false, i64 noundef %addr, ptr addrspace(2) noundef %file, i32 noundef %line, ptr addrspace(2) noundef %func, i1 noundef zeroext false) #20
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %entry
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define weak dso_local spir_func void @__asan_store8(i64 noundef %addr, i32 noundef %as, ptr addrspace(2) noundef %file, i32 noundef %line, ptr addrspace(2) noundef %func) #14 !srcloc !60 {
entry:
  %call = tail call spir_func noundef i64 @_ZN12_GLOBAL__N_111MemToShadowEmi(i64 noundef %addr, i32 noundef %as) #20
  %tobool.not = icmp eq i64 %call, 0
  br i1 %tobool.not, label %if.end, label %land.lhs.true

land.lhs.true:                                    ; preds = %entry
  %0 = inttoptr i64 %call to ptr addrspace(4)
  %1 = load i8, ptr addrspace(4) %0, align 1, !tbaa !26
  %tobool1.not = icmp eq i8 %1, 0
  br i1 %tobool1.not, label %if.end, label %if.then

if.then:                                          ; preds = %land.lhs.true
  tail call spir_func void @_Z26__asan_report_access_errormimbmPU3AS2KciS0_b(i64 noundef %addr, i32 noundef %as, i64 noundef 8, i1 noundef zeroext true, i64 noundef %addr, ptr addrspace(2) noundef %file, i32 noundef %line, ptr addrspace(2) noundef %func, i1 noundef zeroext false) #20
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %entry
  ret void
}

attributes #0 = { "sycl-device-global-size"="16" "sycl-unique-id"="_Z10dev_global" }
attributes #1 = { "sycl-unique-id"="_Z12__DeviceType" }
attributes #2 = { "sycl-unique-id"="_Z29__AsanShadowMemoryGlobalStart" }
attributes #3 = { "sycl-unique-id"="_Z27__AsanShadowMemoryGlobalEnd" }
attributes #4 = { "sycl-unique-id"="_Z28__AsanShadowMemoryLocalStart" }
attributes #5 = { "sycl-unique-id"="_Z26__AsanShadowMemoryLocalEnd" }
attributes #6 = { "sycl-unique-id"="_Z26__DeviceSanitizerReportMem" }
attributes #7 = { nounwind uwtable "frame-pointer"="all" "sycl-optlevel"="0" }
attributes #8 = { convergent mustprogress noinline norecurse nounwind optnone sanitize_address uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="device_global_image_scope.cpp" "sycl-optlevel"="0" "sycl-single-task" "uniform-work-group-size"="true" }
attributes #9 = { convergent mustprogress noinline norecurse nounwind optnone sanitize_address uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-optlevel"="0" }
attributes #10 = { alwaysinline convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/export/users/maosuzha/ics_workspace/syclos/libdevice/itt_compiler_wrappers.cpp" "sycl-optlevel"="2" }
attributes #11 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #12 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #13 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/export/users/maosuzha/ics_workspace/syclos/libdevice/itt_stubs.cpp" "sycl-optlevel"="2" }
attributes #14 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/export/users/maosuzha/ics_workspace/syclos/libdevice/sanitizer_utils.cpp" "sycl-optlevel"="2" }
attributes #15 = { convergent inlinehint mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-optlevel"="2" }
attributes #16 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-optlevel"="2" }
attributes #17 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #18 = { inlinehint mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/export/users/maosuzha/ics_workspace/syclos/libdevice/sanitizer_utils.cpp" "sycl-optlevel"="2" }
attributes #19 = { convergent mustprogress nofree nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #20 = { convergent nounwind }
attributes #21 = { nounwind }
attributes #22 = { convergent nounwind willreturn memory(none) }

!opencl.spir.version = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!spirv.Source = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
!llvm.ident = !{!4, !5, !5, !5, !5, !5, !5, !4, !5, !5, !5, !5, !5, !5, !5, !5, !4, !5, !5, !5, !5, !4}
!llvm.module.flags = !{!6, !7, !8}

!0 = !{!1}
!1 = !{i32 6147, i32 2, !"_Z10dev_global"}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 19.0.0git (https://github.com/intel/llvm.git 5ec9cce34290bb937846e7ee81dad0100e2dba17)"}
!5 = !{!"clang version 19.0.0git (https://github.com/intel/llvm.git 55c6d2b3751e3b59e9aaf3972f375c33dc0b9d8b)"}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{i32 5139065}
!10 = !{}
!11 = !{i32 3766433}
!12 = !{i32 3762560}
!13 = !{i32 442}
!14 = !{!15, !15, i64 0}
!15 = !{!"long", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C++ TBAA"}
!18 = !{i32 1030}
!19 = !{i32 462}
!20 = !{!21, !21, i64 0}
!21 = !{!"any pointer", !16, i64 0}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !16, i64 0}
!24 = !{i32 592}
!25 = !{i32 -2147117330}
!26 = !{!16, !16, i64 0}
!27 = !{i32 5823}
!28 = !{i1 false, i1 false}
!29 = !{!30, !30, i64 0}
!30 = !{!"_ZTS10DeviceType", !16, i64 0}
!31 = !{i32 10108}
!32 = distinct !{!32, !33}
!33 = !{!"llvm.loop.mustprogress"}
!34 = distinct !{!34, !33}
!35 = distinct !{!35, !33}
!36 = !{!37, !23, i64 520}
!37 = !{!"_ZTS21DeviceSanitizerReport", !23, i64 0, !16, i64 4, !16, i64 261, !23, i64 520, !15, i64 528, !15, i64 536, !15, i64 544, !15, i64 552, !15, i64 560, !15, i64 568, !38, i64 576, !23, i64 580, !39, i64 584, !40, i64 588, !38, i64 592}
!38 = !{!"bool", !16, i64 0}
!39 = !{!"_ZTS25DeviceSanitizerMemoryType", !16, i64 0}
!40 = !{!"_ZTS24DeviceSanitizerErrorType", !16, i64 0}
!41 = !{!37, !15, i64 528}
!42 = !{!37, !15, i64 536}
!43 = !{!37, !15, i64 544}
!44 = !{!37, !15, i64 552}
!45 = !{!37, !15, i64 560}
!46 = !{!37, !15, i64 568}
!47 = !{!37, !38, i64 576}
!48 = !{!37, !23, i64 580}
!49 = !{!37, !40, i64 588}
!50 = !{!37, !39, i64 584}
!51 = !{!37, !38, i64 592}
!52 = distinct !{!52, !33}
!53 = !{i32 322507}
!54 = !{i32 322618}
!55 = !{i32 322729}
!56 = !{i32 322841}
!57 = !{i32 322950}
!58 = !{i32 323059}
!59 = !{i32 -2147115491}
!60 = !{i32 -2147111779}
