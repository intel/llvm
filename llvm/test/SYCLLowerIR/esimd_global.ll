; This test checks whether globals are converted
; correctly to llvm's native vector type.
;
; RUN: opt < %s -ESIMDLowerVecArg -S | FileCheck %s                            

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd" = type { <16 x i32> }

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE4Test" = comdat any

; CHECK: [[NEWGLOBAL:[@a-zA-Z0-9_]*]] = dso_local global <16 x i32> zeroinitializer, align 64 #0
@0 = dso_local global %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd" zeroinitializer, align 64 #0

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE4Test"(i32 addrspace(1)* %_arg_) local_unnamed_addr #1 comdat !kernel_arg_addr_space !8 !kernel_arg_access_qual !9 !kernel_arg_type !10 !kernel_arg_base_type !10 !kernel_arg_type_qual !11 !sycl_explicit_simd !12 !intel_reqd_sub_group_size !8 {
entry:
  %vc.i = alloca %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd", align 64
  %agg.tmp.i = alloca %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd", align 64
  %call.esimd.i.i.i.i.i = call <3 x i32> @llvm.genx.local.id.v3i32() #5
  %local_id.y.i.i.i.i.i = extractelement <3 x i32> %call.esimd.i.i.i.i.i, i32 1
  %local_id.y.cast.ty.i.i.i.i.i = zext i32 %local_id.y.i.i.i.i.i to i64
  %call.esimd1.i.i.i.i.i = call <3 x i32> @llvm.genx.local.size.v3i32() #5
  %wgsize.y.i.i.i.i.i = extractelement <3 x i32> %call.esimd1.i.i.i.i.i, i32 1
  %wgsize.y.cast.ty.i.i.i.i.i = zext i32 %wgsize.y.i.i.i.i.i to i64
  %group.id.y.i.i.i.i.i = call i32 @llvm.genx.group.id.y() #5
  %group.id.y.cast.ty.i.i.i.i.i = zext i32 %group.id.y.i.i.i.i.i to i64
  %mul.i.i.i.i.i = mul nuw i64 %wgsize.y.cast.ty.i.i.i.i.i, %group.id.y.cast.ty.i.i.i.i.i
  %add.i.i.i.i.i = add i64 %mul.i.i.i.i.i, %local_id.y.cast.ty.i.i.i.i.i
  %local_id.x.i.i.i.i.i = extractelement <3 x i32> %call.esimd.i.i.i.i.i, i32 0
  %local_id.x.cast.ty.i.i.i.i.i = zext i32 %local_id.x.i.i.i.i.i to i64
  %wgsize.x.i.i.i.i.i = extractelement <3 x i32> %call.esimd1.i.i.i.i.i, i32 0
  %wgsize.x.cast.ty.i.i.i.i.i = zext i32 %wgsize.x.i.i.i.i.i to i64
  %group.id.x.i.i.i.i.i = call i32 @llvm.genx.group.id.x() #5
  %group.id.x.cast.ty.i.i.i.i.i = zext i32 %group.id.x.i.i.i.i.i to i64
  %mul.i4.i.i.i.i = mul nuw i64 %group.id.x.cast.ty.i.i.i.i.i, %wgsize.x.cast.ty.i.i.i.i.i
  %add.i5.i.i.i.i = add i64 %mul.i4.i.i.i.i, %local_id.x.cast.ty.i.i.i.i.i
  %0 = bitcast %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd"* %agg.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %0)
  %1 = bitcast %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd"* %vc.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %1) #5
  %conv.i = trunc i64 %add.i5.i.i.i.i to i32
  %2 = addrspacecast %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd"* %vc.i to %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)*
  %splat.splatinsert.i.i = insertelement <16 x i32> undef, i32 %conv.i, i32 0
  %splat.splat.i.i = shufflevector <16 x i32> %splat.splatinsert.i.i, <16 x i32> undef, <16 x i32> zeroinitializer
  %M_data.i13.i = getelementptr inbounds %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd", %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)* %2, i64 0, i32 0
  store <16 x i32> %splat.splat.i.i, <16 x i32> addrspace(4)* %M_data.i13.i, align 64, !tbaa !13
  %conv3.i = trunc i64 %add.i.i.i.i.i to i32
  %splat.splatinsert.i20.i = insertelement <8 x i32> undef, i32 %conv3.i, i32 0
  %splat.splat.i21.i = shufflevector <8 x i32> %splat.splatinsert.i20.i, <8 x i32> undef, <8 x i32> zeroinitializer
  %call.esimd.i.i.i.i.i2 = call <16 x i32> @llvm.genx.vload.v16i32.p4v16i32(<16 x i32> addrspace(4)* %M_data.i13.i) #5
  %call4.esimd.i.i.i.i = call <16 x i32> @llvm.genx.wrregioni.v16i32.v8i32.i16.v8i1(<16 x i32> %call.esimd.i.i.i.i.i2, <8 x i32> %splat.splat.i21.i, i32 0, i32 8, i32 1, i16 0, i32 0, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>) #5
  call void @llvm.genx.vstore.v16i32.p4v16i32(<16 x i32> %call4.esimd.i.i.i.i, <16 x i32> addrspace(4)* %M_data.i13.i) #5
  %cmp.i = icmp eq i64 %add.i.i.i.i.i, 0
  %..i = select i1 %cmp.i, i64 %add.i5.i.i.i.i, i64 %add.i.i.i.i.i
  %conv9.i = trunc i64 %..i to i32
; CHECK: store <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <16 x i32> addrspace(4)* addrspacecast (<16 x i32>* getelementptr inbounds ({{.+}}, {{.+}}* bitcast (<16 x i32>* [[NEWGLOBAL]] to {{.+}}*), i64 0, i32 0) to <16 x i32> addrspace(4)*), align 64, !tbaa.struct !16
  store <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <16 x i32> addrspace(4)* addrspacecast (<16 x i32>* getelementptr inbounds (%"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd", %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd"* @0, i64 0, i32 0) to <16 x i32> addrspace(4)*), align 64, !tbaa.struct !16
  %mul.i = shl nsw i32 %conv9.i, 4
  %idx.ext.i = sext i32 %mul.i to i64
  %add.ptr.i16 = getelementptr inbounds i32, i32 addrspace(1)* %_arg_, i64 %idx.ext.i
  %add.ptr.i = addrspacecast i32 addrspace(1)* %add.ptr.i16 to i32 addrspace(4)*
  %3 = addrspacecast %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd"* %agg.tmp.i to %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)*
  %call.esimd.i.i.i = call <16 x i32> @llvm.genx.vload.v16i32.p4v16i32(<16 x i32> addrspace(4)* %M_data.i13.i) #5
  %M_data.i2.i.i = getelementptr inbounds %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd", %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)* %3, i64 0, i32 0
  call void @llvm.genx.vstore.v16i32.p4v16i32(<16 x i32> %call.esimd.i.i.i, <16 x i32> addrspace(4)* %M_data.i2.i.i) #5
  call spir_func void @_Z3fooPiN2cl4sycl5INTEL3gpu4simdIiLi16EEE(i32 addrspace(4)* %add.ptr.i, %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd"* nonnull %agg.tmp.i) #5
  store <16 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>, <16 x i32> addrspace(4)* addrspacecast (<16 x i32>* getelementptr inbounds (%"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd", %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd"* @0, i64 0, i32 0) to <16 x i32> addrspace(4)*), align 64, !tbaa.struct !16
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %1) #5
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %0)
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg %0, i8* nocapture %1) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg %0, i8* nocapture %1) #2

; Function Attrs: noinline norecurse nounwind
define dso_local spir_func void @_Z3fooPiN2cl4sycl5INTEL3gpu4simdIiLi16EEE(i32 addrspace(4)* %C, %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd"* %v) local_unnamed_addr #3 {
entry:
  %agg.tmp = alloca %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd", align 64
  %0 = addrspacecast %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd"* %v to %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)*
  %1 = addrspacecast %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd"* %agg.tmp to %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)*
  %M_data.i.i = getelementptr inbounds %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd", %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)* %0, i64 0, i32 0
  %call.esimd.i.i = call <16 x i32> @llvm.genx.vload.v16i32.p4v16i32(<16 x i32> addrspace(4)* %M_data.i.i), !noalias !17
; CHECK: {{.+}} = call <16 x i32> @llvm.genx.vload.v16i32.p4v16i32(<16 x i32> addrspace(4)* getelementptr ({{.+}}, {{.+}} addrspace(4)* addrspacecast ({{.+}}* bitcast (<16 x i32>* [[NEWGLOBAL]] to {{.+}}*) to {{.+}} addrspace(4)*), i64 0, i32 0)), !noalias !17
  %call.esimd.i8.i = call <16 x i32> @llvm.genx.vload.v16i32.p4v16i32(<16 x i32> addrspace(4)* getelementptr (%"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd", %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)* addrspacecast (%"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd"* @0 to %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)*), i64 0, i32 0)), !noalias !17
  %add.i = add <16 x i32> %call.esimd.i8.i, %call.esimd.i.i
  %M_data.i.i.i = getelementptr inbounds %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd", %"class._ZTSN2cl4sycl5INTEL3gpu4simdIiLi16EEE.cl::sycl::INTEL::gpu::simd" addrspace(4)* %1, i64 0, i32 0
  call void @llvm.genx.vstore.v16i32.p4v16i32(<16 x i32> %add.i, <16 x i32> addrspace(4)* %M_data.i.i.i)
  %2 = ptrtoint i32 addrspace(4)* %C to i64
  %call.esimd.i.i2 = call <16 x i32> @llvm.genx.vload.v16i32.p4v16i32(<16 x i32> addrspace(4)* %M_data.i.i.i)
  call void @llvm.genx.svm.block.st.v16i32(i64 %2, <16 x i32> %call.esimd.i.i2)
  ret void
}

; Function Attrs: nounwind readnone
declare !genx_intrinsic_id !20 <16 x i32> @llvm.genx.wrregioni.v16i32.v8i32.i16.v8i1(<16 x i32> %0, <8 x i32> %1, i32 %2, i32 %3, i32 %4, i16 %5, i32 %6, <8 x i1> %7) #4

; Function Attrs: nounwind
declare !genx_intrinsic_id !21 <16 x i32> @llvm.genx.vload.v16i32.p4v16i32(<16 x i32> addrspace(4)* %0) #5

; Function Attrs: nounwind
declare !genx_intrinsic_id !22 void @llvm.genx.vstore.v16i32.p4v16i32(<16 x i32> %0, <16 x i32> addrspace(4)* %1) #5

; Function Attrs: nounwind
declare !genx_intrinsic_id !23 void @llvm.genx.svm.block.st.v16i32(i64 %0, <16 x i32> %1) #5

; Function Attrs: nounwind readnone
declare !genx_intrinsic_id !24 <3 x i32> @llvm.genx.local.id.v3i32() #4

; Function Attrs: nounwind readnone
declare !genx_intrinsic_id !25 <3 x i32> @llvm.genx.local.size.v3i32() #4

; Function Attrs: nounwind readnone
declare !genx_intrinsic_id !26 i32 @llvm.genx.group.id.y() #4

; Function Attrs: nounwind readnone
declare !genx_intrinsic_id !27 i32 @llvm.genx.group.id.x() #4

attributes #0 = { "genx_byte_offset"="192" "genx_volatile" }
attributes #1 = { norecurse "CMGenxMain" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="512" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "oclrt"="1" "stack-protector-buffer-size"="8" "sycl-module-id"="subroutine.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind willreturn }
attributes #3 = { noinline norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="512" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone }
attributes #5 = { nounwind }

!llvm.dependent-libraries = !{!0}
!llvm.module.flags = !{!1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}
!genx.kernels = !{!5}

!0 = !{!"libcpmt"}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 6, i32 100000}
!4 = !{!"clang version 11.0.0"}
!5 = !{void (i32 addrspace(1)*)* @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE4Test", !"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE4Test", !6, i32 0, i32 0, !6, !7, i32 0, i32 0}
!6 = !{i32 0}
!7 = !{!"svmptr_t"}
!8 = !{i32 1}
!9 = !{!"none"}
!10 = !{!"int*"}
!11 = !{!""}
!12 = !{}
!13 = !{!14, !14, i64 0}
!14 = !{!"omnipotent char", !15, i64 0}
!15 = !{!"Simple C++ TBAA"}
!16 = !{i64 0, i64 64, !13}
!17 = !{!18}
!18 = distinct !{!18, !19, !"_ZNK2cl4sycl5INTEL3gpu4simdIiLi16EEplERKS4_: %agg.result"}
!19 = distinct !{!19, !"_ZNK2cl4sycl5INTEL3gpu4simdIiLi16EEplERKS4_"}
!20 = !{i32 8275}
!21 = !{i32 8268}
!22 = !{i32 8269}
!23 = !{i32 8166}
!24 = !{i32 8029}
!25 = !{i32 8034}
!26 = !{i32 8020}
!27 = !{i32 8019}

