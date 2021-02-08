; This test checks whether subroutine arguments are converted
; correctly to llvm's native vector type when callee is an extern function.
;
; RUN: opt < %s -ESIMDLowerVecArg -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "genx64-unknown-unknown-sycldevice"

%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd" = type { <16 x float> }

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE4Test" = comdat any

@_ZL2VL = internal unnamed_addr addrspace(1) constant i32 16, align 4

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE4Test"(float addrspace(1)* %_arg_, float addrspace(1)* %_arg_1, float addrspace(1)* %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_addr_space !9 !kernel_arg_access_qual !10 !kernel_arg_type !11 !kernel_arg_base_type !11 !kernel_arg_type_qual !12 !kernel_arg_accessor_ptr !13 !sycl_explicit_simd !14 !intel_reqd_sub_group_size !15 {
entry:
  %vc.i = alloca %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd", align 64
  %agg.tmp5.i = alloca %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd", align 64
  %agg.tmp6.i = alloca %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd", align 64
  %call.esimd.i.i.i.i = tail call <3 x i32> @llvm.genx.local.id.v3i32() #4
  %local_id.x.i.i.i.i = extractelement <3 x i32> %call.esimd.i.i.i.i, i32 0
  %call.esimd1.i.i.i.i = tail call <3 x i32> @llvm.genx.local.size.v3i32() #4
  %wgsize.x.i.i.i.i = extractelement <3 x i32> %call.esimd1.i.i.i.i, i32 0
  %group.id.x.i.i.i.i = tail call i32 @llvm.genx.group.id.x() #4
  %mul.i.i.i.i = mul i32 %wgsize.x.i.i.i.i, %group.id.x.i.i.i.i
  %add.i.i.i.i = add i32 %mul.i.i.i.i, %local_id.x.i.i.i.i
  %0 = bitcast %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* %agg.tmp5.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %0)
  %1 = bitcast %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* %agg.tmp6.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %1)
  %2 = load i32, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @_ZL2VL to i32 addrspace(4)*), align 4, !tbaa !16, !noalias !20
  %mul.i.tr.i = mul i32 %2, %add.i.i.i.i
  %conv.i = shl i32 %mul.i.tr.i, 2
  %3 = ptrtoint float addrspace(1)* %_arg_ to i64
  %4 = trunc i64 %3 to i32
  %call1.esimd.i18.i = tail call <16 x float> @llvm.genx.oword.ld.unaligned.v16f32(i32 0, i32 %4, i32 %conv.i), !noalias !23
  %5 = ptrtoint float addrspace(1)* %_arg_1 to i64
  %6 = trunc i64 %5 to i32
  %call1.esimd.i.i = tail call <16 x float> @llvm.genx.oword.ld.unaligned.v16f32(i32 0, i32 %6, i32 %conv.i), !noalias !26
  %7 = bitcast %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* %vc.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %7) #4
  %va.sroa.0.0..sroa_idx.i = getelementptr inbounds %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd", %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* %agg.tmp5.i, i64 0, i32 0
  store <16 x float> %call1.esimd.i18.i, <16 x float>* %va.sroa.0.0..sroa_idx.i, align 64, !tbaa.struct !29
  %vb.sroa.0.0..sroa_idx.i = getelementptr inbounds %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd", %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* %agg.tmp6.i, i64 0, i32 0
  store <16 x float> %call1.esimd.i.i, <16 x float>* %vb.sroa.0.0..sroa_idx.i, align 64, !tbaa.struct !29
  %8 = addrspacecast %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* %vc.i to %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)*

; CHECK:  [[BITCASTRESULT1:%[a-zA-Z0-9_]*]] = bitcast {{.+}} addrspace(4)* %8 to <16 x float> addrspace(4)*
; CHECK:  [[BITCASTRESULT2:%[a-zA-Z0-9_]*]] = bitcast {{.+}} %agg.tmp5.i to <16 x float>*
; CHECK:  [[BITCASTRESULT3:%[a-zA-Z0-9_]*]] = bitcast {{.+}} %agg.tmp6.i to <16 x float>*
; CHECK-NEXT:  call spir_func void @_Z4vaddN2cl4sycl5INTEL3gpu4simdIfLi16EEES4_(<16 x float> addrspace(4)* [[BITCASTRESULT1]], <16 x float>* [[BITCASTRESULT2]], <16 x float>* [[BITCASTRESULT3]])

  call spir_func void @_Z4vaddN2cl4sycl5INTEL3gpu4simdIfLi16EEES4_(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)* sret(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd") align 64 %8, %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* nonnull byval(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd") align 64 %agg.tmp5.i, %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* nonnull byval(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd") align 64 %agg.tmp6.i) #6
  %agg.tmp8.sroa.0.0..sroa_idx.i = getelementptr inbounds %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd", %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* %vc.i, i64 0, i32 0
  %agg.tmp8.sroa.0.0.copyload.i = load <16 x float>, <16 x float>* %agg.tmp8.sroa.0.0..sroa_idx.i, align 64, !tbaa.struct !29
  %9 = lshr i32 %mul.i.tr.i, 2
  %shr.i.i = and i32 %9, 268435455
  %10 = ptrtoint float addrspace(1)* %_arg_3 to i64
  %11 = trunc i64 %10 to i32
  call void @llvm.genx.oword.st.v16f32(i32 %11, i32 %shr.i.i, <16 x float> %agg.tmp8.sroa.0.0.copyload.i)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %7) #4
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %1)
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg %0, i8* nocapture %1) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg %0, i8* nocapture %1) #1

; CHECK: declare dso_local spir_func void @_Z4vaddN2cl4sycl5INTEL3gpu4simdIfLi16EEES4_(<16 x float> addrspace(4)* {{.+}}, <16 x float>* {{.+}}, <16 x float>* {{.+}}) local_unnamed_addr #2
; Function Attrs: convergent
declare dso_local spir_func void @_Z4vaddN2cl4sycl5INTEL3gpu4simdIfLi16EEES4_(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)* sret(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd") align 64 %0, %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* byval(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd") align 64 %1, %"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd"* byval(%"class._ZTSN2cl4sycl5INTEL3gpu4simdIfLi16EEE.cl::sycl::INTEL::gpu::simd") align 64 %2) local_unnamed_addr #2

; Function Attrs: nounwind readonly
declare !genx_intrinsic_id !31 <16 x float> @llvm.genx.oword.ld.unaligned.v16f32(i32 %0, i32 %1, i32 %2) #3

; Function Attrs: nounwind
declare !genx_intrinsic_id !32 void @llvm.genx.oword.st.v16f32(i32 %0, i32 %1, <16 x float> %2) #4

; Function Attrs: nounwind readnone
declare !genx_intrinsic_id !33 <3 x i32> @llvm.genx.local.id.v3i32() #5

; Function Attrs: nounwind readnone
declare !genx_intrinsic_id !34 <3 x i32> @llvm.genx.local.size.v3i32() #5

; Function Attrs: nounwind readnone
declare !genx_intrinsic_id !35 i32 @llvm.genx.group.id.x() #5

attributes #0 = { convergent norecurse "CMGenxMain" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="512" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "oclrt"="1" "stack-protector-buffer-size"="8" "sycl-module-id"="..\\extern.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { convergent "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readonly }
attributes #4 = { nounwind }
attributes #5 = { nounwind readnone }
attributes #6 = { convergent }

!llvm.dependent-libraries = !{!0}
!llvm.module.flags = !{!1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}
!genx.kernels = !{!5}

!0 = !{!"libcpmt"}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 0, i32 100000}
!4 = !{!"clang version 12.0.0 (https://github.com/pratikashar/llvm.git ced4e34be27b1f1d7503800e2f6f9901945e1e25)"}
!5 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*)* @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE4Test", !"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE4Test", !6, i32 0, i32 0, !7, !8}
!6 = !{i32 2, i32 2, i32 2}
!7 = !{i32 0, i32 0, i32 0}
!8 = !{!"buffer_t", !"buffer_t", !"buffer_t"}
!9 = !{i32 1, i32 1, i32 1}
!10 = !{!"none", !"none", !"none"}
!11 = !{!"float*", !"float*", !"float*"}
!12 = !{!"", !"", !""}
!13 = !{i1 true, i1 true, i1 true}
!14 = !{}
!15 = !{i32 1}
!16 = !{!17, !17, i64 0}
!17 = !{!"int", !18, i64 0}
!18 = !{!"omnipotent char", !19, i64 0}
!19 = !{!"Simple C++ TBAA"}
!20 = !{!21}
!21 = distinct !{!21, !22, !"_ZNK2cl4sycl2idILi1EEmlIjEENSt9enable_ifIXsr3std11is_integralIT_EE5valueES2_E4typeERKS5_: %agg.result"}
!22 = distinct !{!22, !"_ZNK2cl4sycl2idILi1EEmlIjEENSt9enable_ifIXsr3std11is_integralIT_EE5valueES2_E4typeERKS5_"}
!23 = !{!24}
!24 = distinct !{!24, !25, !"_ZN2cl4sycl5INTEL3gpu10block_loadIfLi16ENS0_8accessorIfLi1ELNS0_6access4modeE1024ELNS5_6targetE2014ELNS5_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEEEENS2_4simdIT_XT0_EEET1_j: %agg.result"}
!25 = distinct !{!25, !"_ZN2cl4sycl5INTEL3gpu10block_loadIfLi16ENS0_8accessorIfLi1ELNS0_6access4modeE1024ELNS5_6targetE2014ELNS5_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEEEENS2_4simdIT_XT0_EEET1_j"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"_ZN2cl4sycl5INTEL3gpu10block_loadIfLi16ENS0_8accessorIfLi1ELNS0_6access4modeE1024ELNS5_6targetE2014ELNS5_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEEEENS2_4simdIT_XT0_EEET1_j: %agg.result"}
!28 = distinct !{!28, !"_ZN2cl4sycl5INTEL3gpu10block_loadIfLi16ENS0_8accessorIfLi1ELNS0_6access4modeE1024ELNS5_6targetE2014ELNS5_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEEEENS2_4simdIT_XT0_EEET1_j"}
!29 = !{i64 0, i64 64, !30}
!30 = !{!18, !18, i64 0}
!31 = !{i32 9967}
!32 = !{i32 9968}
!33 = !{i32 9953}
!34 = !{i32 9958}
!35 = !{i32 9942}
