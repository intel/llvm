; ModuleID = 'dword_atomic_accessor_smoke-sycl-spir64-unknown-unknown-0d7fb7.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.Config = type { i64, i64, i64, i64, i64, i64 }
%class.anon = type { %struct.Config, %"class.sycl::_V1::accessor" }
%"class.sycl::_V1::accessor" = type { %"class.sycl::_V1::detail::AccessorImplDevice", %union.anon }
%"class.sycl::_V1::detail::AccessorImplDevice" = type { %"class.sycl::_V1::id", %"class.sycl::_V1::id", %"class.sycl::_V1::id" }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%union.anon = type { i32 addrspace(1)* }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.4" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.4" = type { <1 x i16> }
%"class.sycl::_V1::ext::intel::esimd::simd" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl" = type { <8 x i32> }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.0" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.0" = type { <8 x i16> }
%class.anon.5 = type { %struct.Config, %"class.sycl::_V1::accessor.6" }
%"class.sycl::_V1::accessor.6" = type { %"class.sycl::_V1::detail::AccessorImplDevice", %union.anon.9 }
%union.anon.9 = type { i64 addrspace(1)* }
%"class.sycl::_V1::ext::intel::esimd::simd.10" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.11" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.11" = type { <8 x i64> }

$_ZTS6TestIDIiLi8E7ImplIncE = comdat any

$_ZZZ4testIiLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIlLi8E7ImplIncE = comdat any

$_ZZZ4testIlLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIjLi8E7ImplIncE = comdat any

$_ZZZ4testIjLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDImLi8E7ImplIncE = comdat any

$_ZZZ4testImLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIiLi8E7ImplDecE = comdat any

$_ZZZ4testIiLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIlLi8E7ImplDecE = comdat any

$_ZZZ4testIlLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIjLi8E7ImplDecE = comdat any

$_ZZZ4testIjLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDImLi8E7ImplDecE = comdat any

$_ZZZ4testImLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIiLi8E10ImplIntAddE = comdat any

$_ZZZ4testIiLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIlLi8E10ImplIntAddE = comdat any

$_ZZZ4testIlLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIjLi8E10ImplIntAddE = comdat any

$_ZZZ4testIjLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDImLi8E10ImplIntAddE = comdat any

$_ZZZ4testImLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIiLi8E10ImplIntSubE = comdat any

$_ZZZ4testIiLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIlLi8E10ImplIntSubE = comdat any

$_ZZZ4testIlLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIjLi8E10ImplIntSubE = comdat any

$_ZZZ4testIjLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDImLi8E10ImplIntSubE = comdat any

$_ZZZ4testImLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIiLi8E8ImplSMaxE = comdat any

$_ZZZ4testIiLi8E8ImplSMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIlLi8E8ImplSMaxE = comdat any

$_ZZZ4testIlLi8E8ImplSMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIiLi8E8ImplSMinE = comdat any

$_ZZZ4testIiLi8E8ImplSMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIlLi8E8ImplSMinE = comdat any

$_ZZZ4testIlLi8E8ImplSMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIjLi8E8ImplUMaxE = comdat any

$_ZZZ4testIjLi8E8ImplUMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDImLi8E8ImplUMaxE = comdat any

$_ZZZ4testImLi8E8ImplUMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIjLi8E8ImplUMinE = comdat any

$_ZZZ4testIjLi8E8ImplUMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDImLi8E8ImplUMinE = comdat any

$_ZZZ4testImLi8E8ImplUMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIiLi8E8ImplLoadE = comdat any

$_ZZZ4testIiLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIlLi8E8ImplLoadE = comdat any

$_ZZZ4testIlLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIjLi8E8ImplLoadE = comdat any

$_ZZZ4testIjLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDImLi8E8ImplLoadE = comdat any

$_ZZZ4testImLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIiLi8E9ImplStoreE = comdat any

$_ZZZ4testIiLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIlLi8E9ImplStoreE = comdat any

$_ZZZ4testIlLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDIjLi8E9ImplStoreE = comdat any

$_ZZZ4testIjLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

$_ZTS6TestIDImLi8E9ImplStoreE = comdat any

$_ZZZ4testImLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_ = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIiLi8E7ImplIncE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !53 !kernel_arg_base_type !53 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !64
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIiLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #4

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIiLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %agg.tmp14 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp14.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp14 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !73
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !81
  %.tr19 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv243 = add i32 %.tr, %.tr19
  %add2.i.i.i.i = shl i32 %conv243, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr19, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr19, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr19, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr19, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr19, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr19, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i21 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i21 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i21, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i22 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i22, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !82
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %agg.tmp.sroa.2.0..sroa_idx41 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp14.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i28 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %14 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !83
  %cmp13 = icmp sgt i64 %14, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx41, align 8
  %call.i.i.i25 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i25) #8
  %call.i.i.i27 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28, <8 x i16> noundef %call.i.i.i27) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1iEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !84
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28) #8, !noalias !84
  %call.i7.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !84
  %call3.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE2EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i7.i) #8, !noalias !84
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i32> noundef %call3.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !87
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef, <1 x i16> noundef, i16 noundef zeroext, <1 x i16> noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef, <8 x i16> noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z15__esimd_barrierv() local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1iEjT_(i32 addrspace(1)* noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE2EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef, i32 noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIlLi8E7ImplIncE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !89 !kernel_arg_base_type !89 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !90
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIlLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIlLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %agg.tmp14 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp14.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp14 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !99
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !102
  %.tr19 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv243 = add i32 %.tr, %.tr19
  %add2.i.i.i.i = shl i32 %conv243, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr19, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr19, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr19, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr19, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr19, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr19, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i21 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i21 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i21, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i22 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i22, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !103
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %agg.tmp.sroa.2.0..sroa_idx41 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp14.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i28 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %14 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !104
  %cmp13 = icmp sgt i64 %14, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx41, align 8
  %call.i.i.i25 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i25) #8
  %call.i.i.i27 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28, <8 x i16> noundef %call.i.i.i27) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1lEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !105
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28) #8, !noalias !105
  %call.i7.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !105
  %call3.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE2ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i7.i) #8, !noalias !105
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i64> noundef %call3.i) #8
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !108
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1lEjT_(i64 addrspace(1)* noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE2ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef, i32 noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef, <8 x i64> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIjLi8E7ImplIncE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !109 !kernel_arg_base_type !109 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !110
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIjLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIjLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %agg.tmp14 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp14.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp14 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !119
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !122
  %.tr19 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv243 = add i32 %.tr, %.tr19
  %add2.i.i.i.i = shl i32 %conv243, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr19, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr19, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr19, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr19, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr19, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr19, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i21 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i21 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i21, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i22 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i22, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !123
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %agg.tmp.sroa.2.0..sroa_idx41 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp14.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i28 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %14 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !124
  %cmp13 = icmp sgt i64 %14, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx41, align 8
  %call.i.i.i25 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i25) #8
  %call.i.i.i27 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28, <8 x i16> noundef %call.i.i.i27) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1jEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !125
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28) #8, !noalias !125
  %call.i7.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !125
  %call3.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE2EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i7.i) #8, !noalias !125
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i32> noundef %call3.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !128
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1jEjT_(i32 addrspace(1)* noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE2EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef, i32 noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDImLi8E7ImplIncE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !129 !kernel_arg_base_type !129 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !130
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testImLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testImLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %agg.tmp14 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp14.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp14 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !139
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !142
  %.tr19 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv243 = add i32 %.tr, %.tr19
  %add2.i.i.i.i = shl i32 %conv243, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr19, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr19, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr19, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr19, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr19, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr19, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i21 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i21 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i21, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i22 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i22, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !143
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %agg.tmp.sroa.2.0..sroa_idx41 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp14.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i28 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %14 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !144
  %cmp13 = icmp sgt i64 %14, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx41, align 8
  %call.i.i.i25 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i25) #8
  %call.i.i.i27 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28, <8 x i16> noundef %call.i.i.i27) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1mEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !145
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28) #8, !noalias !145
  %call.i7.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !145
  %call3.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE2EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i7.i) #8, !noalias !145
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i64> noundef %call3.i) #8
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !148
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1mEjT_(i64 addrspace(1)* noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE2EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef, i32 noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef, <8 x i64> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIiLi8E7ImplDecE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !53 !kernel_arg_base_type !53 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !149
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIiLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIiLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %agg.tmp14 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp14.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp14 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !158
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !160
  %.tr19 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv243 = add i32 %.tr, %.tr19
  %add2.i.i.i.i = shl i32 %conv243, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr19, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr19, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr19, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr19, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr19, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr19, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i21 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i21 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i21, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i22 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i22, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !161
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %agg.tmp.sroa.2.0..sroa_idx41 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp14.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i28 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %14 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !162
  %cmp13 = icmp sgt i64 %14, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx41, align 8
  %call.i.i.i25 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i25) #8
  %call.i.i.i27 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28, <8 x i16> noundef %call.i.i.i27) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1iEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !163
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28) #8, !noalias !163
  %call.i7.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !163
  %call3.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE3EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i7.i) #8, !noalias !163
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i32> noundef %call3.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !166
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE3EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef, i32 noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIlLi8E7ImplDecE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !89 !kernel_arg_base_type !89 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !167
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIlLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIlLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %agg.tmp14 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp14.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp14 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !176
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !178
  %.tr19 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv243 = add i32 %.tr, %.tr19
  %add2.i.i.i.i = shl i32 %conv243, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr19, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr19, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr19, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr19, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr19, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr19, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i21 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i21 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i21, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i22 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i22, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !179
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %agg.tmp.sroa.2.0..sroa_idx41 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp14.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i28 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %14 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !180
  %cmp13 = icmp sgt i64 %14, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx41, align 8
  %call.i.i.i25 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i25) #8
  %call.i.i.i27 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28, <8 x i16> noundef %call.i.i.i27) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1lEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !181
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28) #8, !noalias !181
  %call.i7.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !181
  %call3.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE3ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i7.i) #8, !noalias !181
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i64> noundef %call3.i) #8
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !184
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE3ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef, i32 noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIjLi8E7ImplDecE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !109 !kernel_arg_base_type !109 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !185
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIjLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIjLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %agg.tmp14 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp14.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp14 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !194
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !196
  %.tr19 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv243 = add i32 %.tr, %.tr19
  %add2.i.i.i.i = shl i32 %conv243, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr19, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr19, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr19, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr19, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr19, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr19, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i21 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i21 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i21, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i22 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i22, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !197
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %agg.tmp.sroa.2.0..sroa_idx41 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp14.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i28 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %14 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !198
  %cmp13 = icmp sgt i64 %14, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx41, align 8
  %call.i.i.i25 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i25) #8
  %call.i.i.i27 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28, <8 x i16> noundef %call.i.i.i27) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1jEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !199
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28) #8, !noalias !199
  %call.i7.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !199
  %call3.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE3EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i7.i) #8, !noalias !199
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i32> noundef %call3.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !202
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE3EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef, i32 noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDImLi8E7ImplDecE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !129 !kernel_arg_base_type !129 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !203
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testImLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testImLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %agg.tmp14 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp14.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp14 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !212
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !214
  %.tr19 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv243 = add i32 %.tr, %.tr19
  %add2.i.i.i.i = shl i32 %conv243, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr19, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr19, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr19, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr19, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr19, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr19, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i21 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i21 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i21, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i22 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i22 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i22, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !215
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %agg.tmp.sroa.2.0..sroa_idx41 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp14.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i28 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %14 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !216
  %cmp13 = icmp sgt i64 %14, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx41, align 8
  %call.i.i.i25 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i25) #8
  %call.i.i.i27 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28, <8 x i16> noundef %call.i.i.i27) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1mEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !217
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i28) #8, !noalias !217
  %call.i7.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !217
  %call3.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE3EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i7.i) #8, !noalias !217
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i64> noundef %call3.i) #8
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !220
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic0ILN4sycl3_V13ext5intel5esimd9atomic_opE3EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeE(<8 x i16> noundef, i32 noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIiLi8E10ImplIntAddE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !53 !kernel_arg_base_type !53 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !221
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIiLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIiLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !230
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !232
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !233
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %v0 to i8*
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i32>* %M_data.i.i26 to <8 x i32> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !234
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  store <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <8 x i32>* %M_data.i.i26, align 32, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i32> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1iEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !235
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !235
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !235
  %call.i10.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !235
  %call4.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE0EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i32> noundef %call.i10.i) #8, !noalias !235
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i32> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !238
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE0EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIlLi8E10ImplIntAddE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !89 !kernel_arg_base_type !89 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !239
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIlLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIlLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !248
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !250
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !251
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0 to i8*
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i64>* %M_data.i.i26 to <8 x i64> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !252
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  store <8 x i64> <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>, <8 x i64>* %M_data.i.i26, align 64, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i64> @_Z13__esimd_vloadIlLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i64> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1lEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !253
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !253
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !253
  %call.i10.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadIlLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !253
  %call4.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE0ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i64> noundef %call.i10.i) #8, !noalias !253
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i64> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !256
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z13__esimd_vloadIlLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE0ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i64> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIjLi8E10ImplIntAddE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !109 !kernel_arg_base_type !109 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !257
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIjLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIjLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !266
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !268
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !269
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %v0 to i8*
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i32>* %M_data.i.i26 to <8 x i32> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !270
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  store <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <8 x i32>* %M_data.i.i26, align 32, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i32> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1jEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !271
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !271
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !271
  %call.i10.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !271
  %call4.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE0EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i32> noundef %call.i10.i) #8, !noalias !271
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i32> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !274
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE0EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDImLi8E10ImplIntAddE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !129 !kernel_arg_base_type !129 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !275
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testImLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testImLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !284
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !286
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !287
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0 to i8*
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i64>* %M_data.i.i26 to <8 x i64> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !288
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  store <8 x i64> <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>, <8 x i64>* %M_data.i.i26, align 64, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i64> @_Z13__esimd_vloadImLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i64> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1mEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !289
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !289
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !289
  %call.i10.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadImLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !289
  %call4.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE0EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i64> noundef %call.i10.i) #8, !noalias !289
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i64> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !292
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z13__esimd_vloadImLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE0EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i64> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIiLi8E10ImplIntSubE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !53 !kernel_arg_base_type !53 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !293
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIiLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIiLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !302
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !304
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !305
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %v0 to i8*
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i32>* %M_data.i.i26 to <8 x i32> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !306
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  store <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <8 x i32>* %M_data.i.i26, align 32, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i32> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1iEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !307
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !307
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !307
  %call.i10.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !307
  %call4.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE1EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i32> noundef %call.i10.i) #8, !noalias !307
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i32> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !310
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE1EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIlLi8E10ImplIntSubE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !89 !kernel_arg_base_type !89 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !311
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIlLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIlLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !320
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !322
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !323
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0 to i8*
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i64>* %M_data.i.i26 to <8 x i64> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !324
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  store <8 x i64> <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>, <8 x i64>* %M_data.i.i26, align 64, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i64> @_Z13__esimd_vloadIlLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i64> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1lEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !325
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !325
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !325
  %call.i10.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadIlLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !325
  %call4.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE1ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i64> noundef %call.i10.i) #8, !noalias !325
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i64> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !328
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE1ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i64> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIjLi8E10ImplIntSubE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !109 !kernel_arg_base_type !109 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !329
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIjLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIjLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !338
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !340
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !341
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %v0 to i8*
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i32>* %M_data.i.i26 to <8 x i32> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !342
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  store <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <8 x i32>* %M_data.i.i26, align 32, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i32> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1jEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !343
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !343
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !343
  %call.i10.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !343
  %call4.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE1EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i32> noundef %call.i10.i) #8, !noalias !343
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i32> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !346
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE1EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDImLi8E10ImplIntSubE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !129 !kernel_arg_base_type !129 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !347
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testImLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testImLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !356
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !358
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !359
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0 to i8*
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i64>* %M_data.i.i26 to <8 x i64> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !360
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  store <8 x i64> <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>, <8 x i64>* %M_data.i.i26, align 64, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i64> @_Z13__esimd_vloadImLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i64> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1mEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !361
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !361
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !361
  %call.i10.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadImLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !361
  %call4.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE1EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i64> noundef %call.i10.i) #8, !noalias !361
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i64> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !364
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE1EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i64> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIiLi8E8ImplSMaxE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !53 !kernel_arg_base_type !53 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !365
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIiLi8E8ImplSMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIiLi8E8ImplSMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !374
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !376
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !377
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %v0 to i8*
  %mul.i = sub nsw i64 0, %0
  %conv1.i = sitofp i64 %mul.i to float
  %add.i = fadd float %conv1.i, 5.000000e-01
  %conv2.i = fptosi float %add.i to i32
  %splat.splatinsert.i.i = insertelement <8 x i32> poison, i32 %conv2.i, i64 0
  %splat.splat.i.i = shufflevector <8 x i32> %splat.splatinsert.i.i, <8 x i32> poison, <8 x i32> zeroinitializer
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i32>* %M_data.i.i26 to <8 x i32> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !378
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  store <8 x i32> %splat.splat.i.i, <8 x i32>* %M_data.i.i26, align 32, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i32> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1iEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !379
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !379
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !379
  %call.i10.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !379
  %call4.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE12EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i32> noundef %call.i10.i) #8, !noalias !379
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i32> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !382
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE12EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIlLi8E8ImplSMaxE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !89 !kernel_arg_base_type !89 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !383
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIlLi8E8ImplSMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIlLi8E8ImplSMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !392
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !394
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !395
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0 to i8*
  %mul.i = sub nsw i64 0, %0
  %conv1.i = sitofp i64 %mul.i to float
  %add.i = fadd float %conv1.i, 5.000000e-01
  %conv2.i = fptosi float %add.i to i64
  %splat.splatinsert.i.i = insertelement <8 x i64> poison, i64 %conv2.i, i64 0
  %splat.splat.i.i = shufflevector <8 x i64> %splat.splatinsert.i.i, <8 x i64> poison, <8 x i32> zeroinitializer
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i64>* %M_data.i.i26 to <8 x i64> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !396
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  store <8 x i64> %splat.splat.i.i, <8 x i64>* %M_data.i.i26, align 64, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i64> @_Z13__esimd_vloadIlLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i64> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1lEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !397
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !397
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !397
  %call.i10.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadIlLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !397
  %call4.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE12ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i64> noundef %call.i10.i) #8, !noalias !397
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i64> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !400
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE12ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i64> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIiLi8E8ImplSMinE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !53 !kernel_arg_base_type !53 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !401
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIiLi8E8ImplSMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIiLi8E8ImplSMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !410
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !412
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !413
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %v0 to i8*
  %mul.i = sub nsw i64 0, %0
  %conv1.i = sitofp i64 %mul.i to float
  %add.i = fadd float %conv1.i, 5.000000e-01
  %conv2.i = fptosi float %add.i to i32
  %splat.splatinsert.i.i = insertelement <8 x i32> poison, i32 %conv2.i, i64 0
  %splat.splat.i.i = shufflevector <8 x i32> %splat.splatinsert.i.i, <8 x i32> poison, <8 x i32> zeroinitializer
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i32>* %M_data.i.i26 to <8 x i32> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !414
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  store <8 x i32> %splat.splat.i.i, <8 x i32>* %M_data.i.i26, align 32, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i32> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1iEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !415
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !415
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !415
  %call.i10.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !415
  %call4.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE11EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i32> noundef %call.i10.i) #8, !noalias !415
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i32> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !418
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE11EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIlLi8E8ImplSMinE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !89 !kernel_arg_base_type !89 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !419
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIlLi8E8ImplSMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIlLi8E8ImplSMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !428
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !430
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !431
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0 to i8*
  %mul.i = sub nsw i64 0, %0
  %conv1.i = sitofp i64 %mul.i to float
  %add.i = fadd float %conv1.i, 5.000000e-01
  %conv2.i = fptosi float %add.i to i64
  %splat.splatinsert.i.i = insertelement <8 x i64> poison, i64 %conv2.i, i64 0
  %splat.splat.i.i = shufflevector <8 x i64> %splat.splatinsert.i.i, <8 x i64> poison, <8 x i32> zeroinitializer
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i64>* %M_data.i.i26 to <8 x i64> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !432
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  store <8 x i64> %splat.splat.i.i, <8 x i64>* %M_data.i.i26, align 64, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i64> @_Z13__esimd_vloadIlLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i64> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1lEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !433
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !433
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !433
  %call.i10.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadIlLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !433
  %call4.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE11ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i64> noundef %call.i10.i) #8, !noalias !433
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i64> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !436
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE11ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i64> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIjLi8E8ImplUMaxE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !109 !kernel_arg_base_type !109 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !437
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIjLi8E8ImplUMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIjLi8E8ImplUMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %conv = trunc i64 %0 to i32
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !446
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !448
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !449
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %v0 to i8*
  %conv1.i = sitofp i32 %conv to float
  %add.i = fadd float %conv1.i, 5.000000e-01
  %conv2.i = fptoui float %add.i to i32
  %splat.splatinsert.i.i = insertelement <8 x i32> poison, i32 %conv2.i, i64 0
  %splat.splat.i.i = shufflevector <8 x i32> %splat.splatinsert.i.i, <8 x i32> poison, <8 x i32> zeroinitializer
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i32>* %M_data.i.i26 to <8 x i32> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !450
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  store <8 x i32> %splat.splat.i.i, <8 x i32>* %M_data.i.i26, align 32, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i32> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1jEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !451
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !451
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !451
  %call.i10.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !451
  %call4.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE5EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i32> noundef %call.i10.i) #8, !noalias !451
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i32> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !454
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE5EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDImLi8E8ImplUMaxE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !129 !kernel_arg_base_type !129 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !455
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testImLi8E8ImplUMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testImLi8E8ImplUMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %conv = trunc i64 %0 to i32
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !464
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !466
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !467
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0 to i8*
  %conv1.i = sitofp i32 %conv to float
  %add.i = fadd float %conv1.i, 5.000000e-01
  %conv2.i = fptoui float %add.i to i64
  %splat.splatinsert.i.i = insertelement <8 x i64> poison, i64 %conv2.i, i64 0
  %splat.splat.i.i = shufflevector <8 x i64> %splat.splatinsert.i.i, <8 x i64> poison, <8 x i32> zeroinitializer
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i64>* %M_data.i.i26 to <8 x i64> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !468
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  store <8 x i64> %splat.splat.i.i, <8 x i64>* %M_data.i.i26, align 64, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i64> @_Z13__esimd_vloadImLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i64> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1mEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !469
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !469
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !469
  %call.i10.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadImLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !469
  %call4.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE5EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i64> noundef %call.i10.i) #8, !noalias !469
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i64> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !472
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE5EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i64> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIjLi8E8ImplUMinE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !109 !kernel_arg_base_type !109 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !473
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIjLi8E8ImplUMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIjLi8E8ImplUMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %conv = trunc i64 %0 to i32
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !482
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !484
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !485
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %v0 to i8*
  %conv1.i = sitofp i32 %conv to float
  %add.i = fadd float %conv1.i, 5.000000e-01
  %conv2.i = fptoui float %add.i to i32
  %splat.splatinsert.i.i = insertelement <8 x i32> poison, i32 %conv2.i, i64 0
  %splat.splat.i.i = shufflevector <8 x i32> %splat.splatinsert.i.i, <8 x i32> poison, <8 x i32> zeroinitializer
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i32>* %M_data.i.i26 to <8 x i32> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !486
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  store <8 x i32> %splat.splat.i.i, <8 x i32>* %M_data.i.i26, align 32, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i32> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1jEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !487
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !487
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !487
  %call.i10.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !487
  %call4.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE4EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i32> noundef %call.i10.i) #8, !noalias !487
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i32> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !490
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE4EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDImLi8E8ImplUMinE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !129 !kernel_arg_base_type !129 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !491
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testImLi8E8ImplUMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testImLi8E8ImplUMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %conv = trunc i64 %0 to i32
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !500
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !502
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv249 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv249, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i23 to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i23, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i24 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i24 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i24, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !503
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i, <1 x i16> noundef %call.i.i.i, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0 to i8*
  %conv1.i = sitofp i32 %conv to float
  %add.i = fadd float %conv1.i, 5.000000e-01
  %conv2.i = fptoui float %add.i to i64
  %splat.splatinsert.i.i = insertelement <8 x i64> poison, i64 %conv2.i, i64 0
  %splat.splat.i.i = shufflevector <8 x i64> %splat.splatinsert.i.i, <8 x i64> poison, <8 x i32> zeroinitializer
  %M_data.i.i26 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i64>* %M_data.i.i26 to <8 x i64> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx47 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i31 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i34 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %16 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !504
  %cmp13 = icmp sgt i64 %16, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  store <8 x i64> %splat.splat.i.i, <8 x i64>* %M_data.i.i26, align 64, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx47, align 8
  %call.i.i.i28 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i28) #8
  %call.i.i.i30 = call spir_func noundef <8 x i64> @_Z13__esimd_vloadImLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31, <8 x i64> noundef %call.i.i.i30) #8
  %call.i.i.i33 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34, <8 x i16> noundef %call.i.i.i33) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #7
  %call1.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1mEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !505
  %call.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i34) #8, !noalias !505
  %call.i8.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !505
  %call.i10.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadImLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i31) #8, !noalias !505
  %call4.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE4EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i, i32 noundef %call1.i.i, <8 x i32> noundef %call.i8.i, <8 x i64> noundef %call.i10.i) #8, !noalias !505
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i, <8 x i64> noundef %call4.i) #8
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !508
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE4EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i64> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIiLi8E8ImplLoadE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !53 !kernel_arg_base_type !53 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !509
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIiLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIiLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %agg.tmp1.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp2.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp3.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %agg.tmp14 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp14.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp14 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !518
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !520
  %.tr19 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv244 = add i32 %.tr, %.tr19
  %add2.i.i.i.i = shl i32 %conv244, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr19, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr19, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr19, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr19, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr19, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr19, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i21 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i21 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i21 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i21 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i21, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !521
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i22 = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i24 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i24, <1 x i16> noundef %call.i.i.i22, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %agg.tmp.sroa.2.0..sroa_idx42 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp14.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i29 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %14 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to i8*
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp2.i to i8*
  %16 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to i8*
  %agg.tmp1.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp3.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %M_data.i2.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp1.ascast.i, i64 0, i32 0, i32 0
  %M_data.i.i5.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp2.i, i64 0, i32 0, i32 0
  %17 = addrspacecast <8 x i32>* %M_data.i.i5.i to <8 x i32> addrspace(4)*
  %M_data.i2.i.i8.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp3.ascast.i, i64 0, i32 0, i32 0
  %M_data.i.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %18 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !522
  %cmp13 = icmp sgt i64 %18, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx42, align 8
  %call.i.i.i26 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i26) #8
  %call.i.i.i28 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i29, <8 x i16> noundef %call.i.i.i28) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14)
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15)
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16)
  %call.i.i.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !523
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i, <8 x i32> noundef %call.i.i.i.i) #8, !noalias !523
  store <8 x i32> zeroinitializer, <8 x i32>* %M_data.i.i5.i, align 32, !tbaa !63, !noalias !523
  %call.i.i.i7.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i29) #8, !noalias !523
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i8.i, <8 x i16> noundef %call.i.i.i7.i) #8, !noalias !523
  %call1.i.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1iEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !526
  %call.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i8.i) #8, !noalias !526
  %call.i8.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i) #8, !noalias !526
  %call.i10.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %17) #8, !noalias !526
  %call4.i.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE9EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i.i, i32 noundef %call1.i.i.i, <8 x i32> noundef %call.i8.i.i, <8 x i32> noundef %call.i10.i.i) #8, !noalias !526
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i.i, <8 x i32> noundef %call4.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !529
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE9EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIlLi8E8ImplLoadE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !89 !kernel_arg_base_type !89 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !530
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIlLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIlLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %agg.tmp1.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp2.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp3.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %agg.tmp14 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp14.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp14 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !539
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !541
  %.tr19 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv244 = add i32 %.tr, %.tr19
  %add2.i.i.i.i = shl i32 %conv244, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr19, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr19, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr19, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr19, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr19, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr19, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i21 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i21 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i21 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i21 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i21, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !542
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i22 = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i24 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i24, <1 x i16> noundef %call.i.i.i22, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %agg.tmp.sroa.2.0..sroa_idx42 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp14.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i29 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %14 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to i8*
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp2.i to i8*
  %16 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to i8*
  %agg.tmp1.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp3.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %M_data.i2.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp1.ascast.i, i64 0, i32 0, i32 0
  %M_data.i.i5.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp2.i, i64 0, i32 0, i32 0
  %17 = addrspacecast <8 x i64>* %M_data.i.i5.i to <8 x i64> addrspace(4)*
  %M_data.i2.i.i8.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp3.ascast.i, i64 0, i32 0, i32 0
  %M_data.i.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %18 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !543
  %cmp13 = icmp sgt i64 %18, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx42, align 8
  %call.i.i.i26 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i26) #8
  %call.i.i.i28 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i29, <8 x i16> noundef %call.i.i.i28) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14)
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15)
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16)
  %call.i.i.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !544
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i, <8 x i32> noundef %call.i.i.i.i) #8, !noalias !544
  store <8 x i64> zeroinitializer, <8 x i64>* %M_data.i.i5.i, align 64, !tbaa !63, !noalias !544
  %call.i.i.i7.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i29) #8, !noalias !544
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i8.i, <8 x i16> noundef %call.i.i.i7.i) #8, !noalias !544
  %call1.i.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1lEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !547
  %call.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i8.i) #8, !noalias !547
  %call.i8.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i) #8, !noalias !547
  %call.i10.i.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadIlLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %17) #8, !noalias !547
  %call4.i.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE9ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i.i, i32 noundef %call1.i.i.i, <8 x i32> noundef %call.i8.i.i, <8 x i64> noundef %call.i10.i.i) #8, !noalias !547
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i.i, <8 x i64> noundef %call4.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !550
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE9ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i64> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIjLi8E8ImplLoadE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !109 !kernel_arg_base_type !109 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !551
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIjLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIjLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %agg.tmp1.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp2.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp3.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %agg.tmp14 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp14.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp14 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !560
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !562
  %.tr19 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv244 = add i32 %.tr, %.tr19
  %add2.i.i.i.i = shl i32 %conv244, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr19, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr19, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr19, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr19, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr19, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr19, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i21 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i21 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i21 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i21 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i21, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !563
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i22 = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i24 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i24, <1 x i16> noundef %call.i.i.i22, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %agg.tmp.sroa.2.0..sroa_idx42 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp14.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i29 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %14 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to i8*
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp2.i to i8*
  %16 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to i8*
  %agg.tmp1.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp3.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %M_data.i2.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp1.ascast.i, i64 0, i32 0, i32 0
  %M_data.i.i5.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp2.i, i64 0, i32 0, i32 0
  %17 = addrspacecast <8 x i32>* %M_data.i.i5.i to <8 x i32> addrspace(4)*
  %M_data.i2.i.i8.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp3.ascast.i, i64 0, i32 0, i32 0
  %M_data.i.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %18 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !564
  %cmp13 = icmp sgt i64 %18, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx42, align 8
  %call.i.i.i26 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i26) #8
  %call.i.i.i28 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i29, <8 x i16> noundef %call.i.i.i28) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14)
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15)
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16)
  %call.i.i.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !565
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i, <8 x i32> noundef %call.i.i.i.i) #8, !noalias !565
  store <8 x i32> zeroinitializer, <8 x i32>* %M_data.i.i5.i, align 32, !tbaa !63, !noalias !565
  %call.i.i.i7.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i29) #8, !noalias !565
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i8.i, <8 x i16> noundef %call.i.i.i7.i) #8, !noalias !565
  %call1.i.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1jEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !568
  %call.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i8.i) #8, !noalias !568
  %call.i8.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i) #8, !noalias !568
  %call.i10.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %17) #8, !noalias !568
  %call4.i.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE9EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i.i, i32 noundef %call1.i.i.i, <8 x i32> noundef %call.i8.i.i, <8 x i32> noundef %call.i10.i.i) #8, !noalias !568
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i.i, <8 x i32> noundef %call4.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !571
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE9EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDImLi8E8ImplLoadE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !129 !kernel_arg_base_type !129 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !572
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testImLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testImLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %agg.tmp1.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp2.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp3.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %agg.tmp14 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp14.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp14 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !581
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !583
  %.tr19 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv244 = add i32 %.tr, %.tr19
  %add2.i.i.i.i = shl i32 %conv244, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr19, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr19, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr19, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr19, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr19, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr19, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i21 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i21 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i21 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i21 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i21, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !584
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i22 = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i24 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i24, <1 x i16> noundef %call.i.i.i22, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %agg.tmp.sroa.2.0..sroa_idx42 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp14.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i29 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %14 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to i8*
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp2.i to i8*
  %16 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to i8*
  %agg.tmp1.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp3.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %M_data.i2.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp1.ascast.i, i64 0, i32 0, i32 0
  %M_data.i.i5.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp2.i, i64 0, i32 0, i32 0
  %17 = addrspacecast <8 x i64>* %M_data.i.i5.i to <8 x i64> addrspace(4)*
  %M_data.i2.i.i8.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp3.ascast.i, i64 0, i32 0, i32 0
  %M_data.i.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %18 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !585
  %cmp13 = icmp sgt i64 %18, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx42, align 8
  %call.i.i.i26 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i26) #8
  %call.i.i.i28 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i29, <8 x i16> noundef %call.i.i.i28) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14)
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15)
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16)
  %call.i.i.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !586
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i, <8 x i32> noundef %call.i.i.i.i) #8, !noalias !586
  store <8 x i64> zeroinitializer, <8 x i64>* %M_data.i.i5.i, align 64, !tbaa !63, !noalias !586
  %call.i.i.i7.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i29) #8, !noalias !586
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i8.i, <8 x i16> noundef %call.i.i.i7.i) #8, !noalias !586
  %call1.i.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1mEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !589
  %call.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i8.i) #8, !noalias !589
  %call.i8.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i) #8, !noalias !589
  %call.i10.i.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadImLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %17) #8, !noalias !589
  %call4.i.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE9EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i.i, i32 noundef %call1.i.i.i, <8 x i32> noundef %call.i8.i.i, <8 x i64> noundef %call.i10.i.i) #8, !noalias !589
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i.i, <8 x i64> noundef %call4.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !592
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE9EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i64> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIiLi8E9ImplStoreE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !53 !kernel_arg_base_type !53 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !593
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIiLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIiLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %agg.tmp1.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp2.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp3.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !602
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !604
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv250 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv250, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i23 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i23 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i23 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i23, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !605
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i24 = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i26 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i26, <1 x i16> noundef %call.i.i.i24, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %v0 to i8*
  %M_data.i.i27 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i32>* %M_data.i.i27 to <8 x i32> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx48 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i32 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i35 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %16 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to i8*
  %17 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp2.i to i8*
  %18 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to i8*
  %agg.tmp1.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp2.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp2.i to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp3.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %M_data.i2.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp1.ascast.i, i64 0, i32 0, i32 0
  %M_data.i2.i.i7.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp2.ascast.i, i64 0, i32 0, i32 0
  %M_data.i2.i.i10.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp3.ascast.i, i64 0, i32 0, i32 0
  %M_data.i.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %19 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !606
  %cmp13 = icmp sgt i64 %19, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  store <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>, <8 x i32>* %M_data.i.i27, align 32, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx48, align 8
  %call.i.i.i29 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i29) #8
  %call.i.i.i31 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i32, <8 x i32> noundef %call.i.i.i31) #8
  %call.i.i.i34 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i35, <8 x i16> noundef %call.i.i.i34) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #7
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %16)
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %17)
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %18)
  %call.i.i.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !607
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i, <8 x i32> noundef %call.i.i.i.i) #8, !noalias !607
  %call.i.i.i6.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i32) #8, !noalias !607
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i7.i, <8 x i32> noundef %call.i.i.i6.i) #8, !noalias !607
  %call.i.i.i9.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i35) #8, !noalias !607
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i10.i, <8 x i16> noundef %call.i.i.i9.i) #8, !noalias !607
  %call1.i.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1iEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !610
  %call.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i10.i) #8, !noalias !610
  %call.i8.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i) #8, !noalias !610
  %call.i10.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i7.i) #8, !noalias !610
  %call4.i.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE6EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i.i, i32 noundef %call1.i.i.i, <8 x i32> noundef %call.i8.i.i, <8 x i32> noundef %call.i10.i.i) #8, !noalias !610
  call spir_func void @_Z14__esimd_vstoreIiLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i.i, <8 x i32> noundef %call4.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %16)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %17)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %18)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !613
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE6EiLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIlLi8E9ImplStoreE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !89 !kernel_arg_base_type !89 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !614
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIlLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIlLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %agg.tmp1.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp2.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp3.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !623
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !625
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv250 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv250, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i23 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i23 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i23 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i23, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !626
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i24 = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i26 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i26, <1 x i16> noundef %call.i.i.i24, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0 to i8*
  %M_data.i.i27 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i64>* %M_data.i.i27 to <8 x i64> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx48 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i32 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i35 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %16 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to i8*
  %17 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp2.i to i8*
  %18 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to i8*
  %agg.tmp1.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp2.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp2.i to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %agg.tmp3.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %M_data.i2.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp1.ascast.i, i64 0, i32 0, i32 0
  %M_data.i2.i.i7.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %agg.tmp2.ascast.i, i64 0, i32 0, i32 0
  %M_data.i2.i.i10.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp3.ascast.i, i64 0, i32 0, i32 0
  %M_data.i.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %19 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !627
  %cmp13 = icmp sgt i64 %19, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  store <8 x i64> <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>, <8 x i64>* %M_data.i.i27, align 64, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx48, align 8
  %call.i.i.i29 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i29) #8
  %call.i.i.i31 = call spir_func noundef <8 x i64> @_Z13__esimd_vloadIlLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i32, <8 x i64> noundef %call.i.i.i31) #8
  %call.i.i.i34 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i35, <8 x i16> noundef %call.i.i.i34) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #7
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %16)
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %17)
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %18)
  %call.i.i.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !628
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i, <8 x i32> noundef %call.i.i.i.i) #8, !noalias !628
  %call.i.i.i6.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadIlLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i32) #8, !noalias !628
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i7.i, <8 x i64> noundef %call.i.i.i6.i) #8, !noalias !628
  %call.i.i.i9.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i35) #8, !noalias !628
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i10.i, <8 x i16> noundef %call.i.i.i9.i) #8, !noalias !628
  %call1.i.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1lEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !631
  %call.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i10.i) #8, !noalias !631
  %call.i8.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i) #8, !noalias !631
  %call.i10.i.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadIlLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i7.i) #8, !noalias !631
  %call4.i.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE6ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i.i, i32 noundef %call1.i.i.i, <8 x i32> noundef %call.i8.i.i, <8 x i64> noundef %call.i10.i.i) #8, !noalias !631
  call spir_func void @_Z14__esimd_vstoreIlLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i.i, <8 x i64> noundef %call4.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %16)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %17)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %18)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !634
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE6ElLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i64> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDIjLi8E9ImplStoreE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i32 addrspace(1)* noundef align 4 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !109 !kernel_arg_base_type !109 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon, %class.anon* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i32 addrspace(1)* %_arg_accessor, i32 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !635
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testIjLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testIjLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %agg.tmp1.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp2.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp3.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !644
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 2
  %stride = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !646
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv250 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv250, 2
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 3
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 12
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 4
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 20
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 24
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 28
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i23 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i23 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i23 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i23, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !647
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i24 = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i26 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i26, <1 x i16> noundef %call.i.i.i24, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %v0 to i8*
  %M_data.i.i27 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i32>* %M_data.i.i27 to <8 x i32> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx48 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i32 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i35 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %tmp to i8*
  %16 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to i8*
  %17 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp2.i to i8*
  %18 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to i8*
  %agg.tmp1.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp2.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp2.i to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp3.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %M_data.i2.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp1.ascast.i, i64 0, i32 0, i32 0
  %M_data.i2.i.i7.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp2.ascast.i, i64 0, i32 0, i32 0
  %M_data.i2.i.i10.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp3.ascast.i, i64 0, i32 0, i32 0
  %M_data.i.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %19 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !648
  %cmp13 = icmp sgt i64 %19, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #7
  store <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>, <8 x i32>* %M_data.i.i27, align 32, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx48, align 8
  %call.i.i.i29 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i29) #8
  %call.i.i.i31 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i32, <8 x i32> noundef %call.i.i.i31) #8
  %call.i.i.i34 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i35, <8 x i16> noundef %call.i.i.i34) #8
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #7
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %16)
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %17)
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %18)
  %call.i.i.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !649
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i, <8 x i32> noundef %call.i.i.i.i) #8, !noalias !649
  %call.i.i.i6.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i32) #8, !noalias !649
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i7.i, <8 x i32> noundef %call.i.i.i6.i) #8, !noalias !649
  %call.i.i.i9.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i35) #8, !noalias !649
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i10.i, <8 x i16> noundef %call.i.i.i9.i) #8, !noalias !649
  %call1.i.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1jEjT_(i32 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !652
  %call.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i10.i) #8, !noalias !652
  %call.i8.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i) #8, !noalias !652
  %call.i10.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i7.i) #8, !noalias !652
  %call4.i.i = call spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE6EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i.i, i32 noundef %call1.i.i.i, <8 x i32> noundef %call.i8.i.i, <8 x i32> noundef %call.i10.i.i) #8, !noalias !652
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i.i.i.i.i, <8 x i32> noundef %call4.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %16)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %17)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %18)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !655
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i32> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE6EjLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i32> noundef) local_unnamed_addr #6

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS6TestIDImLi8E9ImplStoreE(%struct.Config* noundef byval(%struct.Config) align 8 %_arg_cfg, i64 addrspace(1)* noundef align 8 %_arg_accessor) local_unnamed_addr #0 comdat !srcloc !50 !kernel_arg_addr_space !51 !kernel_arg_access_qual !52 !kernel_arg_type !129 !kernel_arg_base_type !129 !kernel_arg_type_qual !54 !kernel_arg_accessor_ptr !55 !sycl_explicit_simd !56 !intel_reqd_sub_group_size !57 !sycl_fixed_targets !56 {
entry:
  %__SYCLKernel = alloca %class.anon.5, align 8
  %agg.tmp = alloca %"class.sycl::_V1::id", align 8
  %__SYCLKernel.ascast = addrspacecast %class.anon.5* %__SYCLKernel to %class.anon.5 addrspace(4)*
  %0 = bitcast %class.anon.5* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #7
  %1 = bitcast %struct.Config* %_arg_cfg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 48, i1 false), !tbaa.struct !58
  %accessor = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1
  %2 = bitcast %"class.sycl::_V1::accessor.6"* %accessor to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 24, i1 false)
  %MData.i = getelementptr inbounds %class.anon.5, %class.anon.5* %__SYCLKernel, i64 0, i32 1, i32 1, i32 0
  store i64 addrspace(1)* %_arg_accessor, i64 addrspace(1)** %MData.i, align 8, !tbaa !63
  %3 = load i64, i64 addrspace(1)* getelementptr inbounds (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32, !noalias !656
  %cmp.i.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %arrayinit.begin.i.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %agg.tmp, i64 0, i32 0, i32 0, i64 0
  store i64 %3, i64* %arrayinit.begin.i.i, align 8, !tbaa !59
  call spir_func void @_ZZZ4testImLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %__SYCLKernel.ascast, %"class.sycl::_V1::id"* noundef nonnull byval(%"class.sycl::_V1::id") align 8 %agg.tmp) #8
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #7
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZZZ4testImLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_ENKUlNS2_2idILi1EEEE_clESB_(%class.anon.5 addrspace(4)* noundef align 8 dereferenceable_or_null(80) %this, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %ii) local_unnamed_addr #5 comdat align 2 !srcloc !50 !sycl_explicit_simd !56 {
entry:
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", align 2
  %agg.tmp1.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp2.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp3.i = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %offsets = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %m = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %v0 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15 = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %agg.tmp16 = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp17 = alloca %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", align 16
  %tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.10", align 64
  %agg.tmp15.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp15 to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp16.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp16 to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %agg.tmp17.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp17 to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %tmp.ascast = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %arrayidx.i = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %ii, i64 0, i32 0, i32 0, i64 0
  %0 = load i64, i64* %arrayidx.i, align 8, !tbaa !59
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #7
  %start_ind = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 2
  %2 = load i64, i64 addrspace(4)* %start_ind, align 8, !tbaa !665
  %.tr = trunc i64 %2 to i32
  %conv2 = shl i32 %.tr, 3
  %stride = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 5
  %3 = load i64, i64 addrspace(4)* %stride, align 8, !tbaa !667
  %.tr21 = trunc i64 %3 to i32
  %vecinit.i.i.i.i = insertelement <8 x i32> undef, i32 %conv2, i64 0
  %conv250 = add i32 %.tr, %.tr21
  %add2.i.i.i.i = shl i32 %conv250, 3
  %vecinit3.i.i.i.i = insertelement <8 x i32> %vecinit.i.i.i.i, i32 %add2.i.i.i.i, i64 1
  %mul4.i.i.i.i = shl i32 %.tr21, 4
  %add5.i.i.i.i = add i32 %mul4.i.i.i.i, %conv2
  %vecinit6.i.i.i.i = insertelement <8 x i32> %vecinit3.i.i.i.i, i32 %add5.i.i.i.i, i64 2
  %mul7.i.i.i.i = mul i32 %.tr21, 24
  %add8.i.i.i.i = add i32 %mul7.i.i.i.i, %conv2
  %vecinit9.i.i.i.i = insertelement <8 x i32> %vecinit6.i.i.i.i, i32 %add8.i.i.i.i, i64 3
  %mul10.i.i.i.i = shl i32 %.tr21, 5
  %add11.i.i.i.i = add i32 %mul10.i.i.i.i, %conv2
  %vecinit12.i.i.i.i = insertelement <8 x i32> %vecinit9.i.i.i.i, i32 %add11.i.i.i.i, i64 4
  %mul13.i.i.i.i = mul i32 %.tr21, 40
  %add14.i.i.i.i = add i32 %mul13.i.i.i.i, %conv2
  %vecinit15.i.i.i.i = insertelement <8 x i32> %vecinit12.i.i.i.i, i32 %add14.i.i.i.i, i64 5
  %mul16.i.i.i.i = mul i32 %.tr21, 48
  %add17.i.i.i.i = add i32 %mul16.i.i.i.i, %conv2
  %vecinit18.i.i.i.i = insertelement <8 x i32> %vecinit15.i.i.i.i, i32 %add17.i.i.i.i, i64 6
  %mul19.i.i.i.i = mul i32 %.tr21, 56
  %add20.i.i.i.i = add i32 %mul19.i.i.i.i, %conv2
  %vecinit21.i.i.i.i = insertelement <8 x i32> %vecinit18.i.i.i.i, i32 %add20.i.i.i.i, i64 7
  %M_data.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %offsets, i64 0, i32 0, i32 0
  %4 = addrspacecast <8 x i32>* %M_data.i.i to <8 x i32> addrspace(4)*
  store <8 x i32> %vecinit21.i.i.i.i, <8 x i32>* %M_data.i.i, align 32, !tbaa !63
  %5 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  %M_data.i.i23 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %m, i64 0, i32 0, i32 0
  %6 = addrspacecast <8 x i16>* %M_data.i.i23 to <8 x i16> addrspace(4)*
  %7 = addrspacecast <8 x i16>* %M_data.i.i23 to <8 x i16> addrspace(4)*
  %8 = addrspacecast <8 x i16>* %M_data.i.i23 to <8 x i16> addrspace(4)*
  store <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>* %M_data.i.i23, align 16, !tbaa !63
  %masked_lane = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 3
  %9 = load i64, i64 addrspace(4)* %masked_lane, align 8, !tbaa !668
  %cmp = icmp slt i64 %9, 8
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %conv9 = trunc i64 %9 to i16
  %10 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %10) #7
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl.3"* %ref.tmp.i, i64 0, i32 0, i32 0
  %11 = addrspacecast <1 x i16>* %M_data.i.i.i to <1 x i16> addrspace(4)*
  store <1 x i16> zeroinitializer, <1 x i16>* %M_data.i.i.i, align 2, !tbaa !63
  %call.i.i.i24 = call spir_func noundef <1 x i16> @_Z13__esimd_vloadItLi1EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<1 x i16> addrspace(4)* noundef %11) #8
  %call.i.i.i.i26 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %7) #8
  %12 = shl i16 %conv9, 1
  %call4.i.i.i = call spir_func noundef <8 x i16> @_Z16__esimd_wrregionItLi8ELi1ELi0ELi1ELi1ELi0EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_NS6_IS7_XT1_EE4typeEtNS6_ItXT1_EE4typeE(<8 x i16> noundef %call.i.i.i.i26, <1 x i16> noundef %call.i.i.i24, i16 noundef zeroext %12, <1 x i16> noundef <i16 1>) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %8, <8 x i16> noundef %call4.i.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %10) #7
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  call spir_func void @_Z13__esimd_fenceh(i8 noundef zeroext 33) #8
  call spir_func void @_Z15__esimd_barrierv() #8
  %repeat = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 0, i32 4
  %13 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0 to i8*
  %M_data.i.i27 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10"* %v0, i64 0, i32 0, i32 0
  %14 = addrspacecast <8 x i64>* %M_data.i.i27 to <8 x i64> addrspace(4)*
  %agg.tmp.sroa.2.0..sroa_idx48 = getelementptr inbounds %class.anon.5, %class.anon.5 addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  %M_data.i2.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp15.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i32 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %agg.tmp16.ascast, i64 0, i32 0, i32 0
  %M_data.i2.i.i35 = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp17.ascast, i64 0, i32 0, i32 0
  %15 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %tmp to i8*
  %16 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to i8*
  %17 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp2.i to i8*
  %18 = bitcast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to i8*
  %agg.tmp1.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd"* %agg.tmp1.i to %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)*
  %agg.tmp2.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::simd.10"* %agg.tmp2.i to %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)*
  %agg.tmp3.ascast.i = addrspacecast %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl"* %agg.tmp3.i to %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)*
  %M_data.i2.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd" addrspace(4)* %agg.tmp1.ascast.i, i64 0, i32 0, i32 0
  %M_data.i2.i.i7.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %agg.tmp2.ascast.i, i64 0, i32 0, i32 0
  %M_data.i2.i.i10.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl", %"class.sycl::_V1::ext::intel::esimd::detail::simd_mask_impl" addrspace(4)* %agg.tmp3.ascast.i, i64 0, i32 0, i32 0
  %M_data.i.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.10", %"class.sycl::_V1::ext::intel::esimd::simd.10" addrspace(4)* %tmp.ascast, i64 0, i32 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end
  %cnt.0 = phi i32 [ 0, %if.end ], [ %inc, %for.body ]
  %conv11 = zext i32 %cnt.0 to i64
  %19 = load i64, i64 addrspace(4)* %repeat, align 8, !tbaa !669
  %cmp13 = icmp sgt i64 %19, %conv11
  br i1 %cmp13, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #7
  ret void

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %13) #7
  store <8 x i64> <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>, <8 x i64>* %M_data.i.i27, align 64, !tbaa !63
  %agg.tmp.sroa.2.0.copyload = load i64 addrspace(1)*, i64 addrspace(1)* addrspace(4)* %agg.tmp.sroa.2.0..sroa_idx48, align 8
  %call.i.i.i29 = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %4) #8
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i, <8 x i32> noundef %call.i.i.i29) #8
  %call.i.i.i31 = call spir_func noundef <8 x i64> @_Z13__esimd_vloadImLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %14) #8
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i32, <8 x i64> noundef %call.i.i.i31) #8
  %call.i.i.i34 = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %6) #8
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i35, <8 x i16> noundef %call.i.i.i34) #8
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #7
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %16)
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %17)
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %18)
  %call.i.i.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i) #8, !noalias !670
  call spir_func void @_Z14__esimd_vstoreIjLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i, <8 x i32> noundef %call.i.i.i.i) #8, !noalias !670
  %call.i.i.i6.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadImLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i32) #8, !noalias !670
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i7.i, <8 x i64> noundef %call.i.i.i6.i) #8, !noalias !670
  %call.i.i.i9.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i35) #8, !noalias !670
  call spir_func void @_Z14__esimd_vstoreItLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i10.i, <8 x i16> noundef %call.i.i.i9.i) #8, !noalias !670
  %call1.i.i.i = call spir_func noundef i32 @_Z25__esimd_get_surface_indexIPU3AS1mEjT_(i64 addrspace(1)* noundef %agg.tmp.sroa.2.0.copyload) #8, !noalias !673
  %call.i.i.i = call spir_func noundef <8 x i16> @_Z13__esimd_vloadItLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i16> addrspace(4)* noundef %M_data.i2.i.i10.i) #8, !noalias !673
  %call.i8.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIjLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %M_data.i2.i.i.i) #8, !noalias !673
  %call.i10.i.i = call spir_func noundef <8 x i64> @_Z13__esimd_vloadImLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i64> addrspace(4)* noundef %M_data.i2.i.i7.i) #8, !noalias !673
  %call4.i.i = call spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE6EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef %call.i.i.i, i32 noundef %call1.i.i.i, <8 x i32> noundef %call.i8.i.i, <8 x i64> noundef %call.i10.i.i) #8, !noalias !673
  call spir_func void @_Z14__esimd_vstoreImLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x i64> addrspace(4)* noundef %M_data.i.i.i.i.i, <8 x i64> noundef %call4.i.i) #8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %16)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %17)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %18)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #7
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %13) #7
  %inc = add nuw nsw i32 %cnt.0, 1
  br label %for.cond, !llvm.loop !676
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef <8 x i64> @_Z21__esimd_dword_atomic1ILN4sycl3_V13ext5intel5esimd9atomic_opE6EmLi8EjENS4_6detail15raw_vector_typeIT0_XT1_EE4typeENS7_ItXT1_EE4typeET2_NS7_IjXT1_EE4typeESA_(<8 x i16> noundef, i32 noundef, <8 x i32> noundef, <8 x i64> noundef) local_unnamed_addr #6

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="dword_atomic_accessor_smoke.cpp" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #5 = { convergent inlinehint norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #6 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #7 = { nounwind }
attributes #8 = { convergent nounwind }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!sycl_aspects = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44}
!llvm.ident = !{!45, !46, !46, !46, !46, !46, !47, !47, !47, !46, !46, !46, !46, !46, !46, !46, !47, !47, !46, !46, !46}
!llvm.module.flags = !{!48, !49}

!0 = !{i32 1, i32 2}
!1 = !{i32 0, i32 100000}
!2 = !{i32 4, i32 100000}
!3 = !{!"cpu", i32 1}
!4 = !{!"gpu", i32 2}
!5 = !{!"accelerator", i32 3}
!6 = !{!"custom", i32 4}
!7 = !{!"fp16", i32 5}
!8 = !{!"fp64", i32 6}
!9 = !{!"image", i32 9}
!10 = !{!"online_compiler", i32 10}
!11 = !{!"online_linker", i32 11}
!12 = !{!"queue_profiling", i32 12}
!13 = !{!"usm_device_allocations", i32 13}
!14 = !{!"usm_host_allocations", i32 14}
!15 = !{!"usm_shared_allocations", i32 15}
!16 = !{!"usm_system_allocations", i32 17}
!17 = !{!"ext_intel_pci_address", i32 18}
!18 = !{!"ext_intel_gpu_eu_count", i32 19}
!19 = !{!"ext_intel_gpu_eu_simd_width", i32 20}
!20 = !{!"ext_intel_gpu_slices", i32 21}
!21 = !{!"ext_intel_gpu_subslices_per_slice", i32 22}
!22 = !{!"ext_intel_gpu_eu_count_per_subslice", i32 23}
!23 = !{!"ext_intel_max_mem_bandwidth", i32 24}
!24 = !{!"ext_intel_mem_channel", i32 25}
!25 = !{!"usm_atomic_host_allocations", i32 26}
!26 = !{!"usm_atomic_shared_allocations", i32 27}
!27 = !{!"atomic64", i32 28}
!28 = !{!"ext_intel_device_info_uuid", i32 29}
!29 = !{!"ext_oneapi_srgb", i32 30}
!30 = !{!"ext_oneapi_native_assert", i32 31}
!31 = !{!"host_debuggable", i32 32}
!32 = !{!"ext_intel_gpu_hw_threads_per_eu", i32 33}
!33 = !{!"ext_oneapi_cuda_async_barrier", i32 34}
!34 = !{!"ext_oneapi_bfloat16_math_functions", i32 35}
!35 = !{!"ext_intel_free_memory", i32 36}
!36 = !{!"ext_intel_device_id", i32 37}
!37 = !{!"ext_intel_memory_clock_rate", i32 38}
!38 = !{!"ext_intel_memory_bus_width", i32 39}
!39 = !{!"emulated", i32 40}
!40 = !{!"int64_base_atomics", i32 7}
!41 = !{!"int64_extended_atomics", i32 8}
!42 = !{!"usm_system_allocator", i32 17}
!43 = !{!"usm_restricted_shared_allocations", i32 16}
!44 = !{!"host", i32 0}
!45 = !{!"clang version 17.0.0 (https://github.com/fineg74/llvm 5cf6439228d04be9c5789d8dbc7e30371f659dc9)"}
!46 = !{!"clang version 17.0.0 (https://github.com/fineg74/llvm 274eb89f4318826478ebfaf168bd18142b5b8be0)"}
!47 = !{!"clang version 17.0.0 (https://github.com/fineg74/llvm 686344c913870a76c1e1627e3173a1337233b5fb)"}
!48 = !{i32 1, !"wchar_size", i32 4}
!49 = !{i32 7, !"frame-pointer", i32 2}
!50 = !{i32 5479131}
!51 = !{i32 0, i32 1}
!52 = !{!"none", !"none"}
!53 = !{!"Config", !"int*"}
!54 = !{!"", !""}
!55 = !{i1 false, i1 true}
!56 = !{}
!57 = !{i32 1}
!58 = !{i64 0, i64 8, !59, i64 8, i64 8, !59, i64 16, i64 8, !59, i64 24, i64 8, !59, i64 32, i64 8, !59, i64 40, i64 8, !59}
!59 = !{!60, !60, i64 0}
!60 = !{!"long", !61, i64 0}
!61 = !{!"omnipotent char", !62, i64 0}
!62 = !{!"Simple C++ TBAA"}
!63 = !{!61, !61, i64 0}
!64 = !{!65, !67, !69, !71}
!65 = distinct !{!65, !66, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!66 = distinct !{!66, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!67 = distinct !{!67, !68, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!68 = distinct !{!68, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!69 = distinct !{!69, !70, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!70 = distinct !{!70, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!71 = distinct !{!71, !72, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!72 = distinct !{!72, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!73 = !{!74, !60, i64 16}
!74 = !{!"_ZTSZZ4testIiLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !76, i64 48}
!75 = !{!"_ZTS6Config", !60, i64 0, !60, i64 8, !60, i64 16, !60, i64 24, !60, i64 32, !60, i64 40}
!76 = !{!"_ZTSN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE", !77, i64 0, !61, i64 24}
!77 = !{!"_ZTSN4sycl3_V16detail18AccessorImplDeviceILi1EEE", !78, i64 0, !80, i64 8, !80, i64 16}
!78 = !{!"_ZTSN4sycl3_V12idILi1EEE", !79, i64 0}
!79 = !{!"_ZTSN4sycl3_V16detail5arrayILi1EEE", !61, i64 0}
!80 = !{!"_ZTSN4sycl3_V15rangeILi1EEE", !79, i64 0}
!81 = !{!74, !60, i64 40}
!82 = !{!74, !60, i64 24}
!83 = !{!74, !60, i64 32}
!84 = !{!85}
!85 = distinct !{!85, !86, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE2EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!86 = distinct !{!86, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE2EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE"}
!87 = distinct !{!87, !88}
!88 = !{!"llvm.loop.mustprogress"}
!89 = !{!"Config", !"long*"}
!90 = !{!91, !93, !95, !97}
!91 = distinct !{!91, !92, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!92 = distinct !{!92, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!93 = distinct !{!93, !94, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!94 = distinct !{!94, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!95 = distinct !{!95, !96, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!96 = distinct !{!96, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!97 = distinct !{!97, !98, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!98 = distinct !{!98, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!99 = !{!100, !60, i64 16}
!100 = !{!"_ZTSZZ4testIlLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !101, i64 48}
!101 = !{!"_ZTSN4sycl3_V18accessorIlLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE", !77, i64 0, !61, i64 24}
!102 = !{!100, !60, i64 40}
!103 = !{!100, !60, i64 24}
!104 = !{!100, !60, i64 32}
!105 = !{!106}
!106 = distinct !{!106, !107, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE2ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!107 = distinct !{!107, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE2ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE"}
!108 = distinct !{!108, !88}
!109 = !{!"Config", !"uint*"}
!110 = !{!111, !113, !115, !117}
!111 = distinct !{!111, !112, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!112 = distinct !{!112, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!113 = distinct !{!113, !114, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!114 = distinct !{!114, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!115 = distinct !{!115, !116, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!116 = distinct !{!116, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!117 = distinct !{!117, !118, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!118 = distinct !{!118, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!119 = !{!120, !60, i64 16}
!120 = !{!"_ZTSZZ4testIjLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !121, i64 48}
!121 = !{!"_ZTSN4sycl3_V18accessorIjLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE", !77, i64 0, !61, i64 24}
!122 = !{!120, !60, i64 40}
!123 = !{!120, !60, i64 24}
!124 = !{!120, !60, i64 32}
!125 = !{!126}
!126 = distinct !{!126, !127, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE2EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!127 = distinct !{!127, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE2EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE"}
!128 = distinct !{!128, !88}
!129 = !{!"Config", !"ulong*"}
!130 = !{!131, !133, !135, !137}
!131 = distinct !{!131, !132, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!132 = distinct !{!132, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!133 = distinct !{!133, !134, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!134 = distinct !{!134, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!135 = distinct !{!135, !136, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!136 = distinct !{!136, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!137 = distinct !{!137, !138, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!138 = distinct !{!138, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!139 = !{!140, !60, i64 16}
!140 = !{!"_ZTSZZ4testImLi8E7ImplIncEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !141, i64 48}
!141 = !{!"_ZTSN4sycl3_V18accessorImLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE", !77, i64 0, !61, i64 24}
!142 = !{!140, !60, i64 40}
!143 = !{!140, !60, i64 24}
!144 = !{!140, !60, i64 32}
!145 = !{!146}
!146 = distinct !{!146, !147, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE2EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!147 = distinct !{!147, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE2EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE"}
!148 = distinct !{!148, !88}
!149 = !{!150, !152, !154, !156}
!150 = distinct !{!150, !151, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!151 = distinct !{!151, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!152 = distinct !{!152, !153, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!153 = distinct !{!153, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!154 = distinct !{!154, !155, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!155 = distinct !{!155, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!156 = distinct !{!156, !157, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!157 = distinct !{!157, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!158 = !{!159, !60, i64 16}
!159 = !{!"_ZTSZZ4testIiLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !76, i64 48}
!160 = !{!159, !60, i64 40}
!161 = !{!159, !60, i64 24}
!162 = !{!159, !60, i64 32}
!163 = !{!164}
!164 = distinct !{!164, !165, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE3EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!165 = distinct !{!165, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE3EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE"}
!166 = distinct !{!166, !88}
!167 = !{!168, !170, !172, !174}
!168 = distinct !{!168, !169, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!169 = distinct !{!169, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!170 = distinct !{!170, !171, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!171 = distinct !{!171, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!172 = distinct !{!172, !173, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!173 = distinct !{!173, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!174 = distinct !{!174, !175, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!175 = distinct !{!175, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!176 = !{!177, !60, i64 16}
!177 = !{!"_ZTSZZ4testIlLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !101, i64 48}
!178 = !{!177, !60, i64 40}
!179 = !{!177, !60, i64 24}
!180 = !{!177, !60, i64 32}
!181 = !{!182}
!182 = distinct !{!182, !183, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE3ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!183 = distinct !{!183, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE3ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE"}
!184 = distinct !{!184, !88}
!185 = !{!186, !188, !190, !192}
!186 = distinct !{!186, !187, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!187 = distinct !{!187, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!188 = distinct !{!188, !189, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!189 = distinct !{!189, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!190 = distinct !{!190, !191, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!191 = distinct !{!191, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!192 = distinct !{!192, !193, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!193 = distinct !{!193, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!194 = !{!195, !60, i64 16}
!195 = !{!"_ZTSZZ4testIjLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !121, i64 48}
!196 = !{!195, !60, i64 40}
!197 = !{!195, !60, i64 24}
!198 = !{!195, !60, i64 32}
!199 = !{!200}
!200 = distinct !{!200, !201, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE3EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!201 = distinct !{!201, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE3EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE"}
!202 = distinct !{!202, !88}
!203 = !{!204, !206, !208, !210}
!204 = distinct !{!204, !205, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!205 = distinct !{!205, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!206 = distinct !{!206, !207, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!207 = distinct !{!207, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!208 = distinct !{!208, !209, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!209 = distinct !{!209, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!210 = distinct !{!210, !211, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!211 = distinct !{!211, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!212 = !{!213, !60, i64 16}
!213 = !{!"_ZTSZZ4testImLi8E7ImplDecEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !141, i64 48}
!214 = !{!213, !60, i64 40}
!215 = !{!213, !60, i64 24}
!216 = !{!213, !60, i64 32}
!217 = !{!218}
!218 = distinct !{!218, !219, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE3EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!219 = distinct !{!219, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE3EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE"}
!220 = distinct !{!220, !88}
!221 = !{!222, !224, !226, !228}
!222 = distinct !{!222, !223, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!223 = distinct !{!223, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!224 = distinct !{!224, !225, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!225 = distinct !{!225, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!226 = distinct !{!226, !227, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!227 = distinct !{!227, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!228 = distinct !{!228, !229, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!229 = distinct !{!229, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!230 = !{!231, !60, i64 16}
!231 = !{!"_ZTSZZ4testIiLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !76, i64 48}
!232 = !{!231, !60, i64 40}
!233 = !{!231, !60, i64 24}
!234 = !{!231, !60, i64 32}
!235 = !{!236}
!236 = distinct !{!236, !237, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE0EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!237 = distinct !{!237, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE0EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!238 = distinct !{!238, !88}
!239 = !{!240, !242, !244, !246}
!240 = distinct !{!240, !241, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!241 = distinct !{!241, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!242 = distinct !{!242, !243, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!243 = distinct !{!243, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!244 = distinct !{!244, !245, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!245 = distinct !{!245, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!246 = distinct !{!246, !247, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!247 = distinct !{!247, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!248 = !{!249, !60, i64 16}
!249 = !{!"_ZTSZZ4testIlLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !101, i64 48}
!250 = !{!249, !60, i64 40}
!251 = !{!249, !60, i64 24}
!252 = !{!249, !60, i64 32}
!253 = !{!254}
!254 = distinct !{!254, !255, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE0ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!255 = distinct !{!255, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE0ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!256 = distinct !{!256, !88}
!257 = !{!258, !260, !262, !264}
!258 = distinct !{!258, !259, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!259 = distinct !{!259, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!260 = distinct !{!260, !261, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!261 = distinct !{!261, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!262 = distinct !{!262, !263, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!263 = distinct !{!263, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!264 = distinct !{!264, !265, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!265 = distinct !{!265, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!266 = !{!267, !60, i64 16}
!267 = !{!"_ZTSZZ4testIjLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !121, i64 48}
!268 = !{!267, !60, i64 40}
!269 = !{!267, !60, i64 24}
!270 = !{!267, !60, i64 32}
!271 = !{!272}
!272 = distinct !{!272, !273, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE0EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!273 = distinct !{!273, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE0EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!274 = distinct !{!274, !88}
!275 = !{!276, !278, !280, !282}
!276 = distinct !{!276, !277, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!277 = distinct !{!277, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!278 = distinct !{!278, !279, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!279 = distinct !{!279, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!280 = distinct !{!280, !281, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!281 = distinct !{!281, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!282 = distinct !{!282, !283, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!283 = distinct !{!283, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!284 = !{!285, !60, i64 16}
!285 = !{!"_ZTSZZ4testImLi8E10ImplIntAddEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !141, i64 48}
!286 = !{!285, !60, i64 40}
!287 = !{!285, !60, i64 24}
!288 = !{!285, !60, i64 32}
!289 = !{!290}
!290 = distinct !{!290, !291, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE0EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!291 = distinct !{!291, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE0EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!292 = distinct !{!292, !88}
!293 = !{!294, !296, !298, !300}
!294 = distinct !{!294, !295, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!295 = distinct !{!295, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!296 = distinct !{!296, !297, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!297 = distinct !{!297, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!298 = distinct !{!298, !299, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!299 = distinct !{!299, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!300 = distinct !{!300, !301, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!301 = distinct !{!301, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!302 = !{!303, !60, i64 16}
!303 = !{!"_ZTSZZ4testIiLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !76, i64 48}
!304 = !{!303, !60, i64 40}
!305 = !{!303, !60, i64 24}
!306 = !{!303, !60, i64 32}
!307 = !{!308}
!308 = distinct !{!308, !309, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE1EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!309 = distinct !{!309, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE1EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!310 = distinct !{!310, !88}
!311 = !{!312, !314, !316, !318}
!312 = distinct !{!312, !313, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!313 = distinct !{!313, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!314 = distinct !{!314, !315, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!315 = distinct !{!315, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!316 = distinct !{!316, !317, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!317 = distinct !{!317, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!318 = distinct !{!318, !319, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!319 = distinct !{!319, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!320 = !{!321, !60, i64 16}
!321 = !{!"_ZTSZZ4testIlLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !101, i64 48}
!322 = !{!321, !60, i64 40}
!323 = !{!321, !60, i64 24}
!324 = !{!321, !60, i64 32}
!325 = !{!326}
!326 = distinct !{!326, !327, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE1ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!327 = distinct !{!327, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE1ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!328 = distinct !{!328, !88}
!329 = !{!330, !332, !334, !336}
!330 = distinct !{!330, !331, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!331 = distinct !{!331, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!332 = distinct !{!332, !333, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!333 = distinct !{!333, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!334 = distinct !{!334, !335, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!335 = distinct !{!335, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!336 = distinct !{!336, !337, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!337 = distinct !{!337, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!338 = !{!339, !60, i64 16}
!339 = !{!"_ZTSZZ4testIjLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !121, i64 48}
!340 = !{!339, !60, i64 40}
!341 = !{!339, !60, i64 24}
!342 = !{!339, !60, i64 32}
!343 = !{!344}
!344 = distinct !{!344, !345, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE1EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!345 = distinct !{!345, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE1EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!346 = distinct !{!346, !88}
!347 = !{!348, !350, !352, !354}
!348 = distinct !{!348, !349, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!349 = distinct !{!349, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!350 = distinct !{!350, !351, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!351 = distinct !{!351, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!352 = distinct !{!352, !353, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!353 = distinct !{!353, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!354 = distinct !{!354, !355, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!355 = distinct !{!355, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!356 = !{!357, !60, i64 16}
!357 = !{!"_ZTSZZ4testImLi8E10ImplIntSubEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !141, i64 48}
!358 = !{!357, !60, i64 40}
!359 = !{!357, !60, i64 24}
!360 = !{!357, !60, i64 32}
!361 = !{!362}
!362 = distinct !{!362, !363, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE1EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!363 = distinct !{!363, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE1EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!364 = distinct !{!364, !88}
!365 = !{!366, !368, !370, !372}
!366 = distinct !{!366, !367, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!367 = distinct !{!367, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!368 = distinct !{!368, !369, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!369 = distinct !{!369, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!370 = distinct !{!370, !371, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!371 = distinct !{!371, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!372 = distinct !{!372, !373, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!373 = distinct !{!373, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!374 = !{!375, !60, i64 16}
!375 = !{!"_ZTSZZ4testIiLi8E8ImplSMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !76, i64 48}
!376 = !{!375, !60, i64 40}
!377 = !{!375, !60, i64 24}
!378 = !{!375, !60, i64 32}
!379 = !{!380}
!380 = distinct !{!380, !381, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE12EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!381 = distinct !{!381, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE12EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!382 = distinct !{!382, !88}
!383 = !{!384, !386, !388, !390}
!384 = distinct !{!384, !385, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!385 = distinct !{!385, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!386 = distinct !{!386, !387, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!387 = distinct !{!387, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!388 = distinct !{!388, !389, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!389 = distinct !{!389, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!390 = distinct !{!390, !391, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!391 = distinct !{!391, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!392 = !{!393, !60, i64 16}
!393 = !{!"_ZTSZZ4testIlLi8E8ImplSMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !101, i64 48}
!394 = !{!393, !60, i64 40}
!395 = !{!393, !60, i64 24}
!396 = !{!393, !60, i64 32}
!397 = !{!398}
!398 = distinct !{!398, !399, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE12ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!399 = distinct !{!399, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE12ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!400 = distinct !{!400, !88}
!401 = !{!402, !404, !406, !408}
!402 = distinct !{!402, !403, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!403 = distinct !{!403, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!404 = distinct !{!404, !405, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!405 = distinct !{!405, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!406 = distinct !{!406, !407, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!407 = distinct !{!407, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!408 = distinct !{!408, !409, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!409 = distinct !{!409, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!410 = !{!411, !60, i64 16}
!411 = !{!"_ZTSZZ4testIiLi8E8ImplSMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !76, i64 48}
!412 = !{!411, !60, i64 40}
!413 = !{!411, !60, i64 24}
!414 = !{!411, !60, i64 32}
!415 = !{!416}
!416 = distinct !{!416, !417, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE11EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!417 = distinct !{!417, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE11EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!418 = distinct !{!418, !88}
!419 = !{!420, !422, !424, !426}
!420 = distinct !{!420, !421, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!421 = distinct !{!421, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!422 = distinct !{!422, !423, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!423 = distinct !{!423, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!424 = distinct !{!424, !425, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!425 = distinct !{!425, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!426 = distinct !{!426, !427, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!427 = distinct !{!427, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!428 = !{!429, !60, i64 16}
!429 = !{!"_ZTSZZ4testIlLi8E8ImplSMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !101, i64 48}
!430 = !{!429, !60, i64 40}
!431 = !{!429, !60, i64 24}
!432 = !{!429, !60, i64 32}
!433 = !{!434}
!434 = distinct !{!434, !435, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE11ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!435 = distinct !{!435, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE11ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!436 = distinct !{!436, !88}
!437 = !{!438, !440, !442, !444}
!438 = distinct !{!438, !439, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!439 = distinct !{!439, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!440 = distinct !{!440, !441, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!441 = distinct !{!441, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!442 = distinct !{!442, !443, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!443 = distinct !{!443, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!444 = distinct !{!444, !445, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!445 = distinct !{!445, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!446 = !{!447, !60, i64 16}
!447 = !{!"_ZTSZZ4testIjLi8E8ImplUMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !121, i64 48}
!448 = !{!447, !60, i64 40}
!449 = !{!447, !60, i64 24}
!450 = !{!447, !60, i64 32}
!451 = !{!452}
!452 = distinct !{!452, !453, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE5EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!453 = distinct !{!453, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE5EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!454 = distinct !{!454, !88}
!455 = !{!456, !458, !460, !462}
!456 = distinct !{!456, !457, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!457 = distinct !{!457, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!458 = distinct !{!458, !459, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!459 = distinct !{!459, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!460 = distinct !{!460, !461, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!461 = distinct !{!461, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!462 = distinct !{!462, !463, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!463 = distinct !{!463, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!464 = !{!465, !60, i64 16}
!465 = !{!"_ZTSZZ4testImLi8E8ImplUMaxEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !141, i64 48}
!466 = !{!465, !60, i64 40}
!467 = !{!465, !60, i64 24}
!468 = !{!465, !60, i64 32}
!469 = !{!470}
!470 = distinct !{!470, !471, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE5EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!471 = distinct !{!471, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE5EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!472 = distinct !{!472, !88}
!473 = !{!474, !476, !478, !480}
!474 = distinct !{!474, !475, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!475 = distinct !{!475, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!476 = distinct !{!476, !477, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!477 = distinct !{!477, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!478 = distinct !{!478, !479, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!479 = distinct !{!479, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!480 = distinct !{!480, !481, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!481 = distinct !{!481, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!482 = !{!483, !60, i64 16}
!483 = !{!"_ZTSZZ4testIjLi8E8ImplUMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !121, i64 48}
!484 = !{!483, !60, i64 40}
!485 = !{!483, !60, i64 24}
!486 = !{!483, !60, i64 32}
!487 = !{!488}
!488 = distinct !{!488, !489, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE4EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!489 = distinct !{!489, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE4EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!490 = distinct !{!490, !88}
!491 = !{!492, !494, !496, !498}
!492 = distinct !{!492, !493, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!493 = distinct !{!493, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!494 = distinct !{!494, !495, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!495 = distinct !{!495, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!496 = distinct !{!496, !497, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!497 = distinct !{!497, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!498 = distinct !{!498, !499, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!499 = distinct !{!499, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!500 = !{!501, !60, i64 16}
!501 = !{!"_ZTSZZ4testImLi8E8ImplUMinEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !141, i64 48}
!502 = !{!501, !60, i64 40}
!503 = !{!501, !60, i64 24}
!504 = !{!501, !60, i64 32}
!505 = !{!506}
!506 = distinct !{!506, !507, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE4EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!507 = distinct !{!507, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE4EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!508 = distinct !{!508, !88}
!509 = !{!510, !512, !514, !516}
!510 = distinct !{!510, !511, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!511 = distinct !{!511, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!512 = distinct !{!512, !513, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!513 = distinct !{!513, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!514 = distinct !{!514, !515, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!515 = distinct !{!515, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!516 = distinct !{!516, !517, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!517 = distinct !{!517, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!518 = !{!519, !60, i64 16}
!519 = !{!"_ZTSZZ4testIiLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !76, i64 48}
!520 = !{!519, !60, i64 40}
!521 = !{!519, !60, i64 24}
!522 = !{!519, !60, i64 32}
!523 = !{!524}
!524 = distinct !{!524, !525, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE21EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!525 = distinct !{!525, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE21EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE"}
!526 = !{!527, !524}
!527 = distinct !{!527, !528, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE9EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!528 = distinct !{!528, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE9EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!529 = distinct !{!529, !88}
!530 = !{!531, !533, !535, !537}
!531 = distinct !{!531, !532, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!532 = distinct !{!532, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!533 = distinct !{!533, !534, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!534 = distinct !{!534, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!535 = distinct !{!535, !536, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!536 = distinct !{!536, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!537 = distinct !{!537, !538, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!538 = distinct !{!538, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!539 = !{!540, !60, i64 16}
!540 = !{!"_ZTSZZ4testIlLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !101, i64 48}
!541 = !{!540, !60, i64 40}
!542 = !{!540, !60, i64 24}
!543 = !{!540, !60, i64 32}
!544 = !{!545}
!545 = distinct !{!545, !546, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE21ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!546 = distinct !{!546, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE21ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE"}
!547 = !{!548, !545}
!548 = distinct !{!548, !549, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE9ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!549 = distinct !{!549, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE9ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!550 = distinct !{!550, !88}
!551 = !{!552, !554, !556, !558}
!552 = distinct !{!552, !553, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!553 = distinct !{!553, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!554 = distinct !{!554, !555, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!555 = distinct !{!555, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!556 = distinct !{!556, !557, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!557 = distinct !{!557, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!558 = distinct !{!558, !559, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!559 = distinct !{!559, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!560 = !{!561, !60, i64 16}
!561 = !{!"_ZTSZZ4testIjLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !121, i64 48}
!562 = !{!561, !60, i64 40}
!563 = !{!561, !60, i64 24}
!564 = !{!561, !60, i64 32}
!565 = !{!566}
!566 = distinct !{!566, !567, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE21EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!567 = distinct !{!567, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE21EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE"}
!568 = !{!569, !566}
!569 = distinct !{!569, !570, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE9EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!570 = distinct !{!570, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE9EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!571 = distinct !{!571, !88}
!572 = !{!573, !575, !577, !579}
!573 = distinct !{!573, !574, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!574 = distinct !{!574, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!575 = distinct !{!575, !576, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!576 = distinct !{!576, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!577 = distinct !{!577, !578, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!578 = distinct !{!578, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!579 = distinct !{!579, !580, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!580 = distinct !{!580, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!581 = !{!582, !60, i64 16}
!582 = !{!"_ZTSZZ4testImLi8E8ImplLoadEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !141, i64 48}
!583 = !{!582, !60, i64 40}
!584 = !{!582, !60, i64 24}
!585 = !{!582, !60, i64 32}
!586 = !{!587}
!587 = distinct !{!587, !588, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE21EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!588 = distinct !{!588, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE21EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEENS3_6detail14simd_mask_implItXT1_EEE"}
!589 = !{!590, !587}
!590 = distinct !{!590, !591, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE9EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!591 = distinct !{!591, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE9EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!592 = distinct !{!592, !88}
!593 = !{!594, !596, !598, !600}
!594 = distinct !{!594, !595, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!595 = distinct !{!595, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!596 = distinct !{!596, !597, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!597 = distinct !{!597, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!598 = distinct !{!598, !599, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!599 = distinct !{!599, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!600 = distinct !{!600, !601, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!601 = distinct !{!601, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!602 = !{!603, !60, i64 16}
!603 = !{!"_ZTSZZ4testIiLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !76, i64 48}
!604 = !{!603, !60, i64 40}
!605 = !{!603, !60, i64 24}
!606 = !{!603, !60, i64 32}
!607 = !{!608}
!608 = distinct !{!608, !609, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE22EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!609 = distinct !{!609, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE22EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!610 = !{!611, !608}
!611 = distinct !{!611, !612, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE6EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!612 = distinct !{!612, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE6EiLi8EjNS0_8accessorIiLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!613 = distinct !{!613, !88}
!614 = !{!615, !617, !619, !621}
!615 = distinct !{!615, !616, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!616 = distinct !{!616, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!617 = distinct !{!617, !618, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!618 = distinct !{!618, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!619 = distinct !{!619, !620, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!620 = distinct !{!620, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!621 = distinct !{!621, !622, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!622 = distinct !{!622, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!623 = !{!624, !60, i64 16}
!624 = !{!"_ZTSZZ4testIlLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !101, i64 48}
!625 = !{!624, !60, i64 40}
!626 = !{!624, !60, i64 24}
!627 = !{!624, !60, i64 32}
!628 = !{!629}
!629 = distinct !{!629, !630, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE22ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!630 = distinct !{!630, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE22ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!631 = !{!632, !629}
!632 = distinct !{!632, !633, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE6ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!633 = distinct !{!633, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE6ElLi8EjNS0_8accessorIlLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!634 = distinct !{!634, !88}
!635 = !{!636, !638, !640, !642}
!636 = distinct !{!636, !637, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!637 = distinct !{!637, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!638 = distinct !{!638, !639, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!639 = distinct !{!639, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!640 = distinct !{!640, !641, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!641 = distinct !{!641, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!642 = distinct !{!642, !643, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!643 = distinct !{!643, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!644 = !{!645, !60, i64 16}
!645 = !{!"_ZTSZZ4testIjLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !121, i64 48}
!646 = !{!645, !60, i64 40}
!647 = !{!645, !60, i64 24}
!648 = !{!645, !60, i64 32}
!649 = !{!650}
!650 = distinct !{!650, !651, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE22EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!651 = distinct !{!651, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE22EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!652 = !{!653, !650}
!653 = distinct !{!653, !654, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE6EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!654 = distinct !{!654, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE6EjLi8EjNS0_8accessorIjLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!655 = distinct !{!655, !88}
!656 = !{!657, !659, !661, !663}
!657 = distinct !{!657, !658, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!658 = distinct !{!658, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!659 = distinct !{!659, !660, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!660 = distinct !{!660, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!661 = distinct !{!661, !662, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!662 = distinct !{!662, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!663 = distinct !{!663, !664, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!664 = distinct !{!664, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!665 = !{!666, !60, i64 16}
!666 = !{!"_ZTSZZ4testImLi8E9ImplStoreEbN4sycl3_V15queueERK6ConfigENKUlRNS2_7handlerEE_clES8_EUlNS2_2idILi1EEEE_", !75, i64 0, !141, i64 48}
!667 = !{!666, !60, i64 40}
!668 = !{!666, !60, i64 24}
!669 = !{!666, !60, i64 32}
!670 = !{!671}
!671 = distinct !{!671, !672, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE22EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!672 = distinct !{!672, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE22EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!673 = !{!674, !671}
!674 = distinct !{!674, !675, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE6EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE: %agg.result"}
!675 = distinct !{!675, !"_ZN4sycl3_V13ext5intel5esimd13atomic_updateILNS3_9atomic_opE6EmLi8EjNS0_8accessorImLi1ELNS0_6access4modeE1026ELNS7_6targetE2014ELNS7_11placeholderE0ENS1_6oneapi22accessor_property_listIJEEEEEEENSt9enable_ifIXaasr3stdE13is_integral_vIT2_Entsr3std10is_pointerIT3_EE5valueENS3_4simdIT0_XT1_EEEE4typeESH_NSI_ISG_XT1_EEESK_NS3_6detail14simd_mask_implItXT1_EEE"}
!676 = distinct !{!676, !88}
