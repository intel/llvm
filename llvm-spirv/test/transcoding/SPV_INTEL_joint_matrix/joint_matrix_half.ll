; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -s -o %t.pre.bc
; RUN: llvm-dis %t.pre.bc -o - | FileCheck %s --check-prefix=CHECK-PRE
; RUN: llvm-spirv %t.bc -spirv-ext=+SPV_INTEL_joint_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-PRE: %spirv.JointMatrixINTEL._float_2_2_0_3
; CHECK-PRE: %spirv.JointMatrixINTEL._half_2_16_0_3
; CHECK-PRE: %spirv.JointMatrixINTEL._half_16_2_3_3

; CHECK-SPIRV-DAG: TypeFloat [[#FloatTy:]] 32
; CHECK-SPIRV-DAG: TypeFloat [[#HalfTy:]] 16
; CHECK-SPIRV-DAG: TypeInt [[#IntTy:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#IntTy]] [[#Zero:]] 0
; CHECK-SPIRV-DAG: Constant [[#IntTy]] [[#Two:]] 2
; CHECK-SPIRV-DAG: Constant [[#IntTy]] [[#Three:]] 3
; CHECK-SPIRV-DAG: Constant [[#IntTy]] [[#Sixteen:]] 16
; CHECK-SPIRV: TypeJointMatrixINTEL [[#CTy:]] [[#FloatTy]] [[#Two]] [[#Two]] [[#Zero]] [[#Three]]
; CHECK-SPIRV: TypeJointMatrixINTEL [[#ATy:]] [[#HalfTy]] [[#Two]] [[#Sixteen]] [[#Zero]] [[#Three]]
; CHECK-SPIRV: TypeJointMatrixINTEL [[#BTy:]] [[#HalfTy]] [[#Sixteen]] [[#Two]] [[#Three]] [[#Three]]

; CHECK-LLVM: %spirv.JointMatrixINTEL._float_2_2_0_3
; CHECK-LLVM: %spirv.JointMatrixINTEL._half_2_16_0_3
; CHECK-LLVM: %spirv.JointMatrixINTEL._half_16_2_3_3

; ModuleID = 'joint_matrix_test-sycl-spir64-unknown-unknown.bc'
source_filename = "joint_matrix_test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"struct._ZTSN2cl4sycl6detail14AssertHappenedE.cl::sycl::detail::AssertHappened" = type { i32, [257 x i8], [257 x i8], [129 x i8], i32, i64, i64, i64, i64, i64, i64 }
%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class.cl::sycl::detail::half_impl::half" = type { half }
%"struct.__spv::__spirv_JointMatrixINTEL" = type { [2 x [2 x [1 x [4 x float]]]]* }
%"struct.__spv::__spirv_JointMatrixINTEL.0" = type { [2 x [16 x [1 x [4 x %"class.cl::sycl::detail::half_impl::half"]]]]* }
%"struct.__spv::__spirv_JointMatrixINTEL.1" = type { [16 x [2 x [4 x [4 x %"class.cl::sycl::detail::half_impl::half"]]]]* }

$_ZTSN2cl4sycl6detail16AssertInfoCopierE = comdat any

$_ZTSZ4mainE11matrix_test = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSN2cl4sycl6detail16AssertInfoCopierE(%"struct._ZTSN2cl4sycl6detail14AssertHappenedE.cl::sycl::detail::AssertHappened" addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !6 {
entry:
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds %"struct._ZTSN2cl4sycl6detail14AssertHappenedE.cl::sycl::detail::AssertHappened", %"struct._ZTSN2cl4sycl6detail14AssertHappenedE.cl::sycl::detail::AssertHappened" addrspace(1)* %_arg_, i64 %2
  %3 = bitcast %"struct._ZTSN2cl4sycl6detail14AssertHappenedE.cl::sycl::detail::AssertHappened" addrspace(1)* %add.ptr.i to i8 addrspace(1)*
  %4 = addrspacecast i8 addrspace(1)* %3 to i8 addrspace(4)*
  tail call spir_func void @__devicelib_assert_read(i8 addrspace(4)* %4) #3
  ret void
}

; Function Attrs: convergent
declare extern_weak dso_local spir_func void @__devicelib_assert_read(i8 addrspace(4)*) local_unnamed_addr #1

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSZ4mainE11matrix_test(float addrspace(1)* %_arg_, i64 %_arg_1, %"class.cl::sycl::detail::half_impl::half" addrspace(1)* %_arg_3, %"class.cl::sycl::detail::half_impl::half" addrspace(1)* %_arg_5) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !6 !intel_reqd_sub_group_size !7 {
entry:
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32, !noalias !8
  %1 = extractelement <3 x i64> %0, i64 1
  %2 = extractelement <3 x i64> %0, i64 0
  %3 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInLocalInvocationId to <3 x i64> addrspace(4)*), align 32, !noalias !15
  %4 = extractelement <3 x i64> %3, i64 1
  %5 = extractelement <3 x i64> %3, i64 0
  %cmp.i.i = icmp ult i64 %1, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %cmp.i45.i = icmp ult i64 %2, 2147483648
  tail call void @llvm.assume(i1 %cmp.i45.i)
  %cmp.i43.i = icmp ult i64 %4, 2147483648
  tail call void @llvm.assume(i1 %cmp.i43.i)
  %sub.i = sub nsw i64 %1, %4
  %cmp.i41.i = icmp ult i64 %5, 2147483648
  tail call void @llvm.assume(i1 %cmp.i41.i)
  %sub5.i = sub nsw i64 %2, %5
  %mul6.i = shl nsw i64 %sub.i, 6
  %add.ptr.i51 = getelementptr inbounds float, float addrspace(1)* %_arg_, i64 %mul6.i
  %add.ptr7.i52 = getelementptr inbounds float, float addrspace(1)* %add.ptr.i51, i64 %sub5.i
  %add.ptr7.i = addrspacecast float addrspace(1)* %add.ptr7.i52 to float addrspace(4)*
  %call8.i = tail call spir_func %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIfLm2ELm2ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(float addrspace(4)* %add.ptr7.i, i64 %_arg_1, i32 0, i32 3, i32 0) #3
  %add.ptr11.i53 = getelementptr inbounds %"class.cl::sycl::detail::half_impl::half", %"class.cl::sycl::detail::half_impl::half" addrspace(1)* %_arg_3, i64 %mul6.i
  %add.ptr16.i55 = getelementptr inbounds %"class.cl::sycl::detail::half_impl::half", %"class.cl::sycl::detail::half_impl::half" addrspace(1)* %_arg_5, i64 %sub5.i
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.body.i, %entry
  %k.0.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  %C.0.i = phi %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* [ %call8.i, %entry ], [ %call19.i, %for.body.i ]
  %cmp.i = icmp ult i32 %k.0.i, 32
  br i1 %cmp.i, label %for.body.i, label %_ZZ4mainENKUlN2cl4sycl7nd_itemILi2EEEE_clES2_.exit

for.body.i:                                       ; preds = %for.cond.i
  %idx.ext46.i = zext i32 %k.0.i to i64
  %add.ptr12.i54 = getelementptr inbounds %"class.cl::sycl::detail::half_impl::half", %"class.cl::sycl::detail::half_impl::half" addrspace(1)* %add.ptr11.i53, i64 %idx.ext46.i
  %add.ptr12.i = addrspacecast %"class.cl::sycl::detail::half_impl::half" addrspace(1)* %add.ptr12.i54 to %"class.cl::sycl::detail::half_impl::half" addrspace(4)*
  %call13.i = tail call spir_func %"struct.__spv::__spirv_JointMatrixINTEL.0" addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIN2cl4sycl6detail9half_impl4halfELm2ELm16ELN5__spv12MatrixLayoutE0ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPSA_mS6_S8_i(%"class.cl::sycl::detail::half_impl::half" addrspace(4)* %add.ptr12.i, i64 %_arg_1, i32 0, i32 3, i32 0) #3
  %mul14.i = shl nuw nsw i32 %k.0.i, 5
  %idx.ext1547.i = zext i32 %mul14.i to i64
  %add.ptr17.i56 = getelementptr inbounds %"class.cl::sycl::detail::half_impl::half", %"class.cl::sycl::detail::half_impl::half" addrspace(1)* %add.ptr16.i55, i64 %idx.ext1547.i
  %add.ptr17.i = addrspacecast %"class.cl::sycl::detail::half_impl::half" addrspace(1)* %add.ptr17.i56 to %"class.cl::sycl::detail::half_impl::half" addrspace(4)*
  %call18.i = tail call spir_func %"struct.__spv::__spirv_JointMatrixINTEL.1" addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIN2cl4sycl6detail9half_impl4halfELm16ELm2ELN5__spv12MatrixLayoutE3ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPSA_mS6_S8_i(%"class.cl::sycl::detail::half_impl::half" addrspace(4)* %add.ptr17.i, i64 %_arg_1, i32 0, i32 3, i32 0) #3
  %call19.i = tail call spir_func %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* @_Z27__spirv_JointMatrixMadINTELIN2cl4sycl6detail9half_impl4halfEfLm2ELm16ELm2ELN5__spv12MatrixLayoutE0ELS6_3ELS6_0ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT6_EXT7_EEEPNS9_IT_XT1_EXT2_EXT4_EXT7_EEEPNS9_ISD_XT2_EXT3_EXT5_EXT7_EEESC_S8_(%"struct.__spv::__spirv_JointMatrixINTEL.0" addrspace(4)* %call13.i, %"struct.__spv::__spirv_JointMatrixINTEL.1" addrspace(4)* %call18.i, %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* %C.0.i, i32 3) #3
  %add.i = add nuw nsw i32 %k.0.i, 16
  br label %for.cond.i, !llvm.loop !20

_ZZ4mainENKUlN2cl4sycl7nd_itemILi2EEEE_clES2_.exit: ; preds = %for.cond.i
  tail call spir_func void @_Z29__spirv_JointMatrixStoreINTELIfLm2ELm2ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEvPT_PNS0_24__spirv_JointMatrixINTELIS4_XT0_EXT1_EXT2_EXT3_EEEmS1_S3_i(float addrspace(4)* %add.ptr7.i, %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* %C.0.i, i64 %_arg_1, i32 0, i32 3, i32 0) #3
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIfLm2ELm2ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(float addrspace(4)*, i64, i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %"struct.__spv::__spirv_JointMatrixINTEL.0" addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIN2cl4sycl6detail9half_impl4halfELm2ELm16ELN5__spv12MatrixLayoutE0ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPSA_mS6_S8_i(%"class.cl::sycl::detail::half_impl::half" addrspace(4)*, i64, i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %"struct.__spv::__spirv_JointMatrixINTEL.1" addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIN2cl4sycl6detail9half_impl4halfELm16ELm2ELN5__spv12MatrixLayoutE3ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPSA_mS6_S8_i(%"class.cl::sycl::detail::half_impl::half" addrspace(4)*, i64, i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* @_Z27__spirv_JointMatrixMadINTELIN2cl4sycl6detail9half_impl4halfEfLm2ELm16ELm2ELN5__spv12MatrixLayoutE0ELS6_3ELS6_0ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT6_EXT7_EEEPNS9_IT_XT1_EXT2_EXT4_EXT7_EEEPNS9_ISD_XT2_EXT3_EXT5_EXT7_EEESC_S8_(%"struct.__spv::__spirv_JointMatrixINTEL.0" addrspace(4)*, %"struct.__spv::__spirv_JointMatrixINTEL.1" addrspace(4)*, %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)*, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z29__spirv_JointMatrixStoreINTELIfLm2ELm2ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEvPT_PNS0_24__spirv_JointMatrixINTELIS4_XT0_EXT1_EXT2_EXT3_EEEmS1_S3_i(float addrspace(4)*, %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)*, i64, i32, i32, i32) local_unnamed_addr #1

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #2

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/work/intel/build/joint_matrix_test.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #3 = { convergent }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!opencl.used.extensions = !{!4}
!opencl.used.optional.core.features = !{!4}
!opencl.compiler.options = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{}
!5 = !{!"Clang"}
!6 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!7 = !{i32 16}
!8 = !{!9, !11, !13}
!9 = distinct !{!9, !10, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv: %agg.result"}
!10 = distinct !{!10, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv"}
!11 = distinct !{!11, !12, !"_ZN7__spirvL22initGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v: %agg.result"}
!12 = distinct !{!12, !"_ZN7__spirvL22initGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v"}
!13 = distinct !{!13, !14, !"_ZN2cl4sycl6detail7Builder10getElementILi2EEEKNS0_7nd_itemIXT_EEEPS5_: %agg.result"}
!14 = distinct !{!14, !"_ZN2cl4sycl6detail7Builder10getElementILi2EEEKNS0_7nd_itemIXT_EEEPS5_"}
!15 = !{!16, !18, !13}
!16 = distinct !{!16, !17, !"_ZN7__spirv28InitSizesSTLocalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv: %agg.result"}
!17 = distinct !{!17, !"_ZN7__spirv28InitSizesSTLocalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv"}
!18 = distinct !{!18, !19, !"_ZN7__spirvL21initLocalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v: %agg.result"}
!19 = distinct !{!19, !"_ZN7__spirvL21initLocalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v"}
!20 = distinct !{!20, !21, !22}
!21 = !{!"llvm.loop.mustprogress"}
!22 = !{!"llvm.loop.unroll.disable"}
