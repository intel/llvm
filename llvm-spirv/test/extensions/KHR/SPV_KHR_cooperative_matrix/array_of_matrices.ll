;; Compiled from joint_matrix_bf16_fill_k_cache.cpp from https://github.com/intel/llvm
;; command: clang++ -fsycl -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 llvm/sycl/test-e2e/Matrix/joint_matrix_bf16_fill_k_cache.cpp -fsycl-device-only -o test.bc
;; and then JointMatrixINTEL target ext type was replaced with CooperativeMatrixKHR

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability CooperativeMatrixKHR
; CHECK-SPIRV-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-SPIRV: TypeInt [[#Int16Ty:]] 16 0
; CHECK-SPIRV: TypeFloat [[#FloatTy:]] 32
; CHECK-SPIRV: TypeCooperativeMatrixKHR [[#MatTy1:]] [[#FloatTy]]
; CHECK-SPIRV: TypeCooperativeMatrixKHR [[#MatTy2:]] [[#Int16Ty]]
; CHECK-SPIRV: TypeCooperativeMatrixKHR [[#MatTy3:]] [[#Int16Ty]]
; CHECK-SPIRV: TypeStruct [[#StructTy1:]] [[#MatTy1]]
; CHECK-SPIRV: TypeArray [[#ArrayTy1:]] [[#StructTy1]] [[#]]
; CHECK-SPIRV: TypeArray [[#]] [[#ArrayTy1]] [[#]]
; CHECK-SPIRV: TypeStruct [[#StructTy2:]] [[#MatTy2]]
; CHECK-SPIRV: TypeArray [[#ArrayTy2:]] [[#StructTy2]] [[#]]
; CHECK-SPIRV: TypeArray [[#]] [[#ArrayTy2]] [[#]]
; CHECK-SPIRV: TypeStruct [[#StructTy3:]] [[#MatTy3]]
; CHECK-SPIRV: TypeArray [[#ArrayTy3:]] [[#StructTy3]] [[#]]
; CHECK-SPIRV: TypeArray [[#]] [[#ArrayTy3]] [[#]]

; CHECK-LLVM: %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix" = type { target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2) }
; CHECK-LLVM: %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.5" = type { target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) }
; CHECK-LLVM: %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.6" = type { target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) }
; CHECK-LLVM: alloca [4 x [4 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix"]]
; CHECK-LLVM: alloca [4 x [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.5"]]
; CHECK-LLVM: alloca [4 x [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.6"]]

; ModuleID = 'test.bc'
source_filename = "llvm/sycl/test-e2e/Matrix/joint_matrix_bf16_fill_k_cache.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::__generated_multi_ptr" = type { ptr addrspace(1) }
%"class.sycl::_V1::__generated_multi_ptr.0" = type { ptr addrspace(1) }
%"class.sycl::_V1::__generated_multi_ptr.1" = type { ptr addrspace(1) }
%"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix" = type { target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2) }
%"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.5" = type { target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) }
%"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.6" = type { target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) }
%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }

$_ZTSZZ12joint_matmulILj256ELj256ELj256ELj256ELj2EN4sycl3_V13ext6oneapi8bfloat16EfLj16EEdPT4_S6_PT5_RNS1_5queueEiENKUlRNS1_7handlerEE_clESC_EUlNS1_7nd_itemILi2EEEE_ = comdat any

@__spirv_BuiltInWorkgroupId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZZ12joint_matmulILj256ELj256ELj256ELj256ELj2EN4sycl3_V13ext6oneapi8bfloat16EfLj16EEdPT4_S6_PT5_RNS1_5queueEiENKUlRNS1_7handlerEE_clESC_EUlNS1_7nd_itemILi2EEEE_(ptr noundef byval(%"class.sycl::_V1::__generated_multi_ptr") align 8 %_arg_pA, ptr noundef byval(%"class.sycl::_V1::__generated_multi_ptr.0") align 8 %_arg_pB, ptr noundef byval(%"class.sycl::_V1::__generated_multi_ptr.1") align 8 %_arg_pC) local_unnamed_addr #0 comdat !srcloc !59 !kernel_arg_buffer_location !60 !intel_reqd_sub_group_size !61 !sycl_fixed_targets !62 !sycl_kernel_omit_args !63 {
entry:
  call void @__itt_offload_wi_start_wrapper()
  %tC.i = alloca [4 x [4 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix"]], align 8
  %tA.i = alloca [4 x [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.5"]], align 8
  %tB.i = alloca [4 x [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.6"]], align 8
  %0 = load i64, ptr %_arg_pA, align 8, !tbaa !64
  %1 = inttoptr i64 %0 to ptr addrspace(1)
  %2 = load i64, ptr %_arg_pB, align 8, !tbaa !64
  %3 = inttoptr i64 %2 to ptr addrspace(1)
  %4 = load i64, ptr %_arg_pC, align 8, !tbaa !64
  %5 = inttoptr i64 %4 to ptr addrspace(1)
  %6 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, i64 8), align 8, !noalias !68
  %7 = load i64, ptr addrspace(1) @__spirv_BuiltInWorkgroupId, align 32, !noalias !68
  %8 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, i64 8), align 8, !noalias !75
  %9 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32, !noalias !75
  %cmp.i.i = icmp ult i64 %6, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %cmp.i208.i = icmp ult i64 %7, 2147483648
  tail call void @llvm.assume(i1 %cmp.i208.i)
  %cmp.i209.i = icmp ult i64 %8, 2147483648
  tail call void @llvm.assume(i1 %cmp.i209.i)
  %cmp.i212.i = icmp ult i64 %9, 2147483648
  tail call void @llvm.assume(i1 %cmp.i212.i)
  %div205.i = lshr i64 %9, 4
  call void @llvm.lifetime.start.p0(i64 128, ptr nonnull %tC.i) #4
  br label %arrayctor.loop.i

arrayctor.loop.i:                                 ; preds = %arrayctor.loop.i, %entry
  %arrayctor.cur.idx.i = phi i64 [ 0, %entry ], [ %arrayctor.cur.add.i, %arrayctor.loop.i ]
  %arrayctor.cur.add.i = add nuw nsw i64 %arrayctor.cur.idx.i, 1
  %arrayctor.done.i = icmp eq i64 %arrayctor.cur.add.i, 16
  br i1 %arrayctor.done.i, label %for.cond.i, label %arrayctor.loop.i

for.cond.i:                                       ; preds = %arrayctor.loop.i, %for.cond.cleanup7.i
  %m.0.i = phi i32 [ %inc12.i, %for.cond.cleanup7.i ], [ 0, %arrayctor.loop.i ]
  %cmp.i = icmp ult i32 %m.0.i, 4
  br i1 %cmp.i, label %for.cond5.preheader.i, label %for.cond14.preheader.i

for.cond5.preheader.i:                            ; preds = %for.cond.i
  %idxprom.i = zext i32 %m.0.i to i64
  br label %for.cond5.i

for.cond14.preheader.i:                           ; preds = %for.cond.i
  %mul50.i = shl nuw nsw i64 %6, 8
  %mul51.i = shl nuw nsw i64 %8, 5
  %add52.i = add nuw nsw i64 %mul50.i, %mul51.i
  %mul80.i = shl nuw nsw i64 %div205.i, 7
  %10 = shl nuw nsw i64 %7, 9
  %11 = add nuw nsw i64 %10, %mul80.i
  br label %for.cond14.i

for.cond5.i:                                      ; preds = %for.body8.i, %for.cond5.preheader.i
  %n.0.i = phi i32 [ %inc.i, %for.body8.i ], [ 0, %for.cond5.preheader.i ]
  %cmp6.i = icmp ult i32 %n.0.i, 4
  br i1 %cmp6.i, label %for.body8.i, label %for.cond.cleanup7.i

for.cond.cleanup7.i:                              ; preds = %for.cond5.i
  %inc12.i = add nuw nsw i32 %m.0.i, 1
  br label %for.cond.i, !llvm.loop !80

for.body8.i:                                      ; preds = %for.cond5.i
  %conv.i = zext i32 %n.0.i to i64
  %arrayidx10.i = getelementptr inbounds [4 x [4 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix"]], ptr %tC.i, i64 0, i64 %idxprom.i, i64 %conv.i
  %call.i.i = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2) @_Z26__spirv_CompositeConstructIffLm8ELm16ELN5__spv9MatrixUseE2ELNS0_12MatrixLayoutE3ELNS0_5Scope4FlagE3EEPNS0_24__spirv_CooperativeMatrixKHRIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEET_(float noundef 0.000000e+00) #5
  store target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2) %call.i.i, ptr %arrayidx10.i, align 8, !tbaa !82
  %inc.i = add nuw nsw i32 %n.0.i, 1
  br label %for.cond5.i, !llvm.loop !84

for.cond14.i:                                     ; preds = %for.cond.cleanup34.i, %for.cond14.preheader.i
  %k2.0.i = phi i32 [ %inc129.i, %for.cond.cleanup34.i ], [ 0, %for.cond14.preheader.i ]
  %cmp15.i = icmp ult i32 %k2.0.i, 8
  br i1 %cmp15.i, label %for.body17.i, label %for.cond132.preheader.i

for.cond132.preheader.i:                          ; preds = %for.cond14.i
  %mul156.i = shl nuw nsw i64 %7, 8
  %mul157.i = shl nuw nsw i64 %div205.i, 6
  %add158.i = add nuw nsw i64 %mul156.i, %mul157.i
  br label %for.cond132.i

for.body17.i:                                     ; preds = %for.cond14.i
  call void @llvm.lifetime.start.p0(i64 64, ptr nonnull %tA.i) #4
  br label %arrayctor.loop20.i

arrayctor.loop20.i:                               ; preds = %arrayctor.loop20.i, %for.body17.i
  %arrayctor.cur21.idx.i = phi i64 [ 0, %for.body17.i ], [ %arrayctor.cur21.add.i, %arrayctor.loop20.i ]
  %arrayctor.cur21.add.i = add nuw nsw i64 %arrayctor.cur21.idx.i, 1
  %arrayctor.done23.i = icmp eq i64 %arrayctor.cur21.add.i, 8
  br i1 %arrayctor.done23.i, label %arrayctor.cont24.i, label %arrayctor.loop20.i

arrayctor.cont24.i:                               ; preds = %arrayctor.loop20.i
  call void @llvm.lifetime.start.p0(i64 64, ptr nonnull %tB.i) #4
  br label %arrayctor.loop27.i

arrayctor.loop27.i:                               ; preds = %arrayctor.loop27.i, %arrayctor.cont24.i
  %arrayctor.cur28.idx.i = phi i64 [ 0, %arrayctor.cont24.i ], [ %arrayctor.cur28.add.i, %arrayctor.loop27.i ]
  %arrayctor.cur28.add.i = add nuw nsw i64 %arrayctor.cur28.idx.i, 1
  %arrayctor.done30.i = icmp eq i64 %arrayctor.cur28.add.i, 8
  br i1 %arrayctor.done30.i, label %for.cond32.preheader.i, label %arrayctor.loop27.i

for.cond32.preheader.i:                           ; preds = %arrayctor.loop27.i
  %12 = shl nuw i32 %k2.0.i, 1
  br label %for.cond32.i

for.cond32.i:                                     ; preds = %for.cond.cleanup92.i, %for.cond32.preheader.i
  %k1.0.i = phi i32 [ %inc126.i, %for.cond.cleanup92.i ], [ 0, %for.cond32.preheader.i ]
  %cmp33.i = icmp ult i32 %k1.0.i, 2
  br i1 %cmp33.i, label %for.body35.i, label %for.cond.cleanup34.i

for.cond.cleanup34.i:                             ; preds = %for.cond32.i
  call void @llvm.lifetime.end.p0(i64 64, ptr nonnull %tB.i) #4
  call void @llvm.lifetime.end.p0(i64 64, ptr nonnull %tA.i) #4
  %inc129.i = add nuw nsw i32 %k2.0.i, 1
  br label %for.cond14.i, !llvm.loop !85

for.body35.i:                                     ; preds = %for.cond32.i
  %13 = add nuw i32 %12, %k1.0.i
  %div37206.i = and i32 %13, 268435455
  %idxprom46.i = zext i32 %k1.0.i to i64
  %mul57.i = shl nuw nsw i32 %div37206.i, 4
  %conv58.i = zext i32 %mul57.i to i64
  %invariant.gep = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %1, i64 %conv58.i
  br label %for.cond39.i

for.cond39.i:                                     ; preds = %for.body42.i, %for.body35.i
  %m38.0.i = phi i32 [ 0, %for.body35.i ], [ %inc60.i, %for.body42.i ]
  %cmp40.i = icmp ult i32 %m38.0.i, 4
  br i1 %cmp40.i, label %for.body42.i, label %for.cond63.preheader.i

for.cond63.preheader.i:                           ; preds = %for.cond39.i
  %mul77.i = shl nuw nsw i32 %div37206.i, 12
  %conv78.i = zext i32 %mul77.i to i64
  %add.ptr.i225.i = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %3, i64 %conv78.i
  br label %for.cond63.i

for.body42.i:                                     ; preds = %for.cond39.i
  %idxprom44.i = zext i32 %m38.0.i to i64
  %arrayidx47.i = getelementptr inbounds [4 x [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.5"]], ptr %tA.i, i64 0, i64 %idxprom44.i, i64 %idxprom46.i
  %mul53.i = shl nuw nsw i32 %m38.0.i, 3
  %conv54.i = zext i32 %mul53.i to i64
  %add55.i = add nuw nsw i64 %add52.i, %conv54.i
  %mul56.i = shl nuw nsw i64 %add55.i, 8
  %gep = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %invariant.gep, i64 %mul56.i
  %call1.i.i = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) @_Z32__spirv_CooperativeMatrixLoadKHRIU3AS1N4sycl3_V13ext6oneapi8bfloat16ES4_Lm8ELm16ELN5__spv9MatrixUseE0ELNS6_12MatrixLayoutE0ELNS6_5Scope4FlagE3EEPNS6_24__spirv_CooperativeMatrixKHRIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS8_SA_i(ptr addrspace(1) noundef %gep, i64 noundef 256, i32 noundef 0, i32 noundef 3, i32 noundef 0) #5
  store target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) %call1.i.i, ptr %arrayidx47.i, align 8, !tbaa !86
  %inc60.i = add nuw nsw i32 %m38.0.i, 1
  br label %for.cond39.i, !llvm.loop !88

for.cond63.i:                                     ; preds = %for.body67.i, %for.cond63.preheader.i
  %n62.0.i = phi i32 [ %inc87.i, %for.body67.i ], [ 0, %for.cond63.preheader.i ]
  %cmp65.i = icmp ult i32 %n62.0.i, 4
  br i1 %cmp65.i, label %for.body67.i, label %for.cond90.i

for.body67.i:                                     ; preds = %for.cond63.i
  %conv64.i = zext i32 %n62.0.i to i64
  %arrayidx72.i = getelementptr inbounds [4 x [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.6"]], ptr %tB.i, i64 0, i64 %conv64.i, i64 %idxprom46.i
  %14 = shl nuw nsw i64 %conv64.i, 5
  %mul85.i = add nuw nsw i64 %14, %11
  %add.ptr.i226.i = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %add.ptr.i225.i, i64 %mul85.i
  %call1.i219.i = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) @_Z32__spirv_CooperativeMatrixLoadKHRIU3AS1N4sycl3_V13ext6oneapi8bfloat16ES4_Lm16ELm16ELN5__spv9MatrixUseE1ELNS6_12MatrixLayoutE2ELNS6_5Scope4FlagE3EEPNS6_24__spirv_CooperativeMatrixKHRIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS8_SA_i(ptr addrspace(1) noundef %add.ptr.i226.i, i64 noundef 512, i32 noundef 2, i32 noundef 3, i32 noundef 0) #5
  store target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) %call1.i219.i, ptr %arrayidx72.i, align 8, !tbaa !89
  %inc87.i = add nuw nsw i32 %n62.0.i, 1
  br label %for.cond63.i, !llvm.loop !91

for.cond90.i:                                     ; preds = %for.cond63.i, %for.cond.cleanup98.i
  %m89.0.i = phi i32 [ %inc123.i, %for.cond.cleanup98.i ], [ 0, %for.cond63.i ]
  %cmp91.i = icmp ult i32 %m89.0.i, 4
  br i1 %cmp91.i, label %for.cond95.preheader.i, label %for.cond.cleanup92.i

for.cond95.preheader.i:                           ; preds = %for.cond90.i
  %idxprom102.i = zext i32 %m89.0.i to i64
  %arrayidx105.i = getelementptr inbounds [4 x [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.5"]], ptr %tA.i, i64 0, i64 %idxprom102.i, i64 %idxprom46.i
  %15 = load target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0), ptr %arrayidx105.i, align 8, !tbaa !86, !noalias !92
  br label %for.cond95.i

for.cond.cleanup92.i:                             ; preds = %for.cond90.i
  %inc126.i = add nuw nsw i32 %k1.0.i, 1
  br label %for.cond32.i, !llvm.loop !95

for.cond95.i:                                     ; preds = %for.body99.i, %for.cond95.preheader.i
  %n94.0.i = phi i32 [ %inc120.i, %for.body99.i ], [ 0, %for.cond95.preheader.i ]
  %cmp97.i = icmp ult i32 %n94.0.i, 4
  br i1 %cmp97.i, label %for.body99.i, label %for.cond.cleanup98.i

for.cond.cleanup98.i:                             ; preds = %for.cond95.i
  %inc123.i = add nuw nsw i32 %m89.0.i, 1
  br label %for.cond90.i, !llvm.loop !96

for.body99.i:                                     ; preds = %for.cond95.i
  %conv96.i = zext i32 %n94.0.i to i64
  %arrayidx109.i = getelementptr inbounds [4 x [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.6"]], ptr %tB.i, i64 0, i64 %conv96.i, i64 %idxprom46.i
  %arrayidx113.i = getelementptr inbounds [4 x [4 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix"]], ptr %tC.i, i64 0, i64 %idxprom102.i, i64 %conv96.i
  %16 = load target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1), ptr %arrayidx109.i, align 8, !tbaa !89, !noalias !92
  %17 = load target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2), ptr %arrayidx113.i, align 8, !tbaa !82, !noalias !92
  %call.i221.i = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2) @_Z31__spirv_CooperativeMatrixMadKHRIN4sycl3_V13ext6oneapi8bfloat16EfLm8ELm16ELm16ELN5__spv9MatrixUseE0ELS6_1ELS6_2ELNS5_12MatrixLayoutE0ELS7_2ELS7_3ELNS5_5Scope4FlagE3EEPNS5_24__spirv_CooperativeMatrixKHRIT0_XT1_EXT3_EXT9_EXT10_EXT6_EEEPNSA_IT_XT1_EXT2_EXT7_EXT10_EXT4_EEEPNSA_ISE_XT2_EXT3_EXT8_EXT10_EXT5_EEESD_S9_(target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) noundef %15, target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) noundef %16, target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2) noundef %17, i32 noundef 3) #5, !noalias !92
  store target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2) %call.i221.i, ptr %arrayidx113.i, align 8, !tbaa !82
  %inc120.i = add nuw nsw i32 %n94.0.i, 1
  br label %for.cond95.i, !llvm.loop !97

for.cond132.i:                                    ; preds = %for.cond.cleanup140.i, %for.cond132.preheader.i
  %m131.0.i = phi i32 [ %inc166.i, %for.cond.cleanup140.i ], [ 0, %for.cond132.preheader.i ]
  %cmp133.i = icmp ult i32 %m131.0.i, 4
  br i1 %cmp133.i, label %for.cond137.preheader.i, label %_ZZZ12joint_matmulILj256ELj256ELj256ELj256ELj2EN4sycl3_V13ext6oneapi8bfloat16EfLj16EEdPT4_S6_PT5_RNS1_5queueEiENKUlRNS1_7handlerEE_clESC_ENKUlNS1_7nd_itemILi2EEEE_clESF_.exit

for.cond137.preheader.i:                          ; preds = %for.cond132.i
  %idxprom143.i = zext i32 %m131.0.i to i64
  %mul152.i = shl nuw nsw i32 %m131.0.i, 3
  %conv153.i = zext i32 %mul152.i to i64
  %add154.i = add nuw nsw i64 %add52.i, %conv153.i
  %mul155.i = shl nuw nsw i64 %add154.i, 8
  %add.ptr.i227.i = getelementptr inbounds float, ptr addrspace(1) %5, i64 %mul155.i
  br label %for.cond137.i

for.cond137.i:                                    ; preds = %for.body141.i, %for.cond137.preheader.i
  %n136.0.i = phi i32 [ %inc163.i, %for.body141.i ], [ 0, %for.cond137.preheader.i ]
  %cmp139.i = icmp ult i32 %n136.0.i, 4
  br i1 %cmp139.i, label %for.body141.i, label %for.cond.cleanup140.i

for.cond.cleanup140.i:                            ; preds = %for.cond137.i
  %inc166.i = add nuw nsw i32 %m131.0.i, 1
  br label %for.cond132.i, !llvm.loop !98

for.body141.i:                                    ; preds = %for.cond137.i
  %conv138.i = zext i32 %n136.0.i to i64
  %arrayidx146.i = getelementptr inbounds [4 x [4 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix"]], ptr %tC.i, i64 0, i64 %idxprom143.i, i64 %conv138.i
  %mul160.i = shl nuw nsw i64 %conv138.i, 4
  %add161.i = add nuw nsw i64 %add158.i, %mul160.i
  %add.ptr.i228.i = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i227.i, i64 %add161.i
  %18 = load target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2), ptr %arrayidx146.i, align 8, !tbaa !82
  tail call spir_func void @_Z33__spirv_CooperativeMatrixStoreKHRIU3AS1ffLm8ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEvPT_PNS1_24__spirv_CooperativeMatrixKHRIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEmS3_S5_i(ptr addrspace(1) noundef %add.ptr.i228.i, target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2) noundef %18, i64 noundef 256, i32 noundef 0, i32 noundef 3, i32 noundef 0) #5
  %inc163.i = add nuw nsw i32 %n136.0.i, 1
  br label %for.cond137.i, !llvm.loop !99

_ZZZ12joint_matmulILj256ELj256ELj256ELj256ELj2EN4sycl3_V13ext6oneapi8bfloat16EfLj16EEdPT4_S6_PT5_RNS1_5queueEiENKUlRNS1_7handlerEE_clESC_ENKUlNS1_7nd_itemILi2EEEE_clESF_.exit: ; preds = %for.cond132.i
  call void @llvm.lifetime.end.p0(i64 128, ptr nonnull %tC.i) #4
  call void @__itt_offload_wi_finish_wrapper()
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #2

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2) @_Z26__spirv_CompositeConstructIffLm8ELm16ELN5__spv9MatrixUseE2ELNS0_12MatrixLayoutE3ELNS0_5Scope4FlagE3EEPNS0_24__spirv_CooperativeMatrixKHRIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEET_(float noundef) local_unnamed_addr #3

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) @_Z32__spirv_CooperativeMatrixLoadKHRIU3AS1N4sycl3_V13ext6oneapi8bfloat16ES4_Lm8ELm16ELN5__spv9MatrixUseE0ELNS6_12MatrixLayoutE0ELNS6_5Scope4FlagE3EEPNS6_24__spirv_CooperativeMatrixKHRIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS8_SA_i(ptr addrspace(1) noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) @_Z32__spirv_CooperativeMatrixLoadKHRIU3AS1N4sycl3_V13ext6oneapi8bfloat16ES4_Lm16ELm16ELN5__spv9MatrixUseE1ELNS6_12MatrixLayoutE2ELNS6_5Scope4FlagE3EEPNS6_24__spirv_CooperativeMatrixKHRIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS8_SA_i(ptr addrspace(1) noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2) @_Z31__spirv_CooperativeMatrixMadKHRIN4sycl3_V13ext6oneapi8bfloat16EfLm8ELm16ELm16ELN5__spv9MatrixUseE0ELS6_1ELS6_2ELNS5_12MatrixLayoutE0ELS7_2ELS7_3ELNS5_5Scope4FlagE3EEPNS5_24__spirv_CooperativeMatrixKHRIT0_XT1_EXT3_EXT9_EXT10_EXT6_EEEPNSA_IT_XT1_EXT2_EXT7_EXT10_EXT4_EEEPNSA_ISE_XT2_EXT3_EXT8_EXT10_EXT5_EEESD_S9_(target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) noundef, target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) noundef, target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2) noundef, i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z33__spirv_CooperativeMatrixStoreKHRIU3AS1ffLm8ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEvPT_PNS1_24__spirv_CooperativeMatrixKHRIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEmS3_S5_i(ptr addrspace(1) noundef, target("spirv.CooperativeMatrixKHR", float, 3, 8, 16, 2) noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #3

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2), ...)

declare void @__itt_offload_wi_start_wrapper()

declare void @__itt_offload_wi_finish_wrapper()

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="llvm/sycl/test-e2e/Matrix/joint_matrix_bf16_fill_k_cache.cpp" "sycl-optlevel"="2" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind }
attributes #5 = { convergent nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!sycl_aspects = !{!4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57}
!llvm.ident = !{!58}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"cpu", i32 1}
!5 = !{!"gpu", i32 2}
!6 = !{!"accelerator", i32 3}
!7 = !{!"custom", i32 4}
!8 = !{!"fp16", i32 5}
!9 = !{!"fp64", i32 6}
!10 = !{!"image", i32 9}
!11 = !{!"online_compiler", i32 10}
!12 = !{!"online_linker", i32 11}
!13 = !{!"queue_profiling", i32 12}
!14 = !{!"usm_device_allocations", i32 13}
!15 = !{!"usm_host_allocations", i32 14}
!16 = !{!"usm_shared_allocations", i32 15}
!17 = !{!"usm_system_allocations", i32 17}
!18 = !{!"ext_intel_pci_address", i32 18}
!19 = !{!"ext_intel_gpu_eu_count", i32 19}
!20 = !{!"ext_intel_gpu_eu_simd_width", i32 20}
!21 = !{!"ext_intel_gpu_slices", i32 21}
!22 = !{!"ext_intel_gpu_subslices_per_slice", i32 22}
!23 = !{!"ext_intel_gpu_eu_count_per_subslice", i32 23}
!24 = !{!"ext_intel_max_mem_bandwidth", i32 24}
!25 = !{!"ext_intel_mem_channel", i32 25}
!26 = !{!"usm_atomic_host_allocations", i32 26}
!27 = !{!"usm_atomic_shared_allocations", i32 27}
!28 = !{!"atomic64", i32 28}
!29 = !{!"ext_intel_device_info_uuid", i32 29}
!30 = !{!"ext_oneapi_srgb", i32 30}
!31 = !{!"ext_oneapi_native_assert", i32 31}
!32 = !{!"host_debuggable", i32 32}
!33 = !{!"ext_intel_gpu_hw_threads_per_eu", i32 33}
!34 = !{!"ext_oneapi_cuda_async_barrier", i32 34}
!35 = !{!"ext_oneapi_bfloat16_math_functions", i32 35}
!36 = !{!"ext_intel_free_memory", i32 36}
!37 = !{!"ext_intel_device_id", i32 37}
!38 = !{!"ext_intel_memory_clock_rate", i32 38}
!39 = !{!"ext_intel_memory_bus_width", i32 39}
!40 = !{!"emulated", i32 40}
!41 = !{!"ext_intel_legacy_image", i32 41}
!42 = !{!"ext_oneapi_bindless_images", i32 42}
!43 = !{!"ext_oneapi_bindless_images_shared_usm", i32 43}
!44 = !{!"ext_oneapi_bindless_images_1d_usm", i32 44}
!45 = !{!"ext_oneapi_bindless_images_2d_usm", i32 45}
!46 = !{!"ext_oneapi_interop_memory_import", i32 46}
!47 = !{!"ext_oneapi_interop_memory_export", i32 47}
!48 = !{!"ext_oneapi_interop_semaphore_import", i32 48}
!49 = !{!"ext_oneapi_interop_semaphore_export", i32 49}
!50 = !{!"ext_oneapi_mipmap", i32 50}
!51 = !{!"ext_oneapi_mipmap_anisotropy", i32 51}
!52 = !{!"ext_oneapi_mipmap_level_reference", i32 52}
!53 = !{!"int64_base_atomics", i32 7}
!54 = !{!"int64_extended_atomics", i32 8}
!55 = !{!"usm_system_allocator", i32 17}
!56 = !{!"usm_restricted_shared_allocations", i32 16}
!57 = !{!"host", i32 0}
!58 = !{!"clang version 18.0.0 (https://github.com/intel/llvm.git cc440821c30daabef517c7c8ff75546719f8094c)"}
!59 = !{i32 242145}
!60 = !{i32 -1, i32 -1, i32 -1}
!61 = !{i32 16}
!62 = !{}
!63 = !{i1 false, i1 false, i1 false}
!64 = !{!65, !65, i64 0}
!65 = !{!"any pointer", !66, i64 0}
!66 = !{!"omnipotent char", !67, i64 0}
!67 = !{!"Simple C++ TBAA"}
!68 = !{!69, !71, !73}
!69 = distinct !{!69, !70, !"_ZN7__spirv22InitSizesSTWorkgroupIdILi2EN4sycl3_V12idILi2EEEE8initSizeEv: %agg.result"}
!70 = distinct !{!70, !"_ZN7__spirv22InitSizesSTWorkgroupIdILi2EN4sycl3_V12idILi2EEEE8initSizeEv"}
!71 = distinct !{!71, !72, !"_ZN7__spirv15initWorkgroupIdILi2EN4sycl3_V12idILi2EEEEET0_v: %agg.result"}
!72 = distinct !{!72, !"_ZN7__spirv15initWorkgroupIdILi2EN4sycl3_V12idILi2EEEEET0_v"}
!73 = distinct !{!73, !74, !"_ZN4sycl3_V16detail7Builder10getElementILi2EEEKNS0_7nd_itemIXT_EEEPS5_: %agg.result"}
!74 = distinct !{!74, !"_ZN4sycl3_V16detail7Builder10getElementILi2EEEKNS0_7nd_itemIXT_EEEPS5_"}
!75 = !{!76, !78, !73}
!76 = distinct !{!76, !77, !"_ZN7__spirv28InitSizesSTLocalInvocationIdILi2EN4sycl3_V12idILi2EEEE8initSizeEv: %agg.result"}
!77 = distinct !{!77, !"_ZN7__spirv28InitSizesSTLocalInvocationIdILi2EN4sycl3_V12idILi2EEEE8initSizeEv"}
!78 = distinct !{!78, !79, !"_ZN7__spirv21initLocalInvocationIdILi2EN4sycl3_V12idILi2EEEEET0_v: %agg.result"}
!79 = distinct !{!79, !"_ZN7__spirv21initLocalInvocationIdILi2EN4sycl3_V12idILi2EEEEET0_v"}
!80 = distinct !{!80, !81}
!81 = !{!"llvm.loop.mustprogress"}
!82 = !{!83, !65, i64 0}
!83 = !{!"_ZTSN4sycl3_V13ext6oneapi12experimental6matrix12joint_matrixINS0_9sub_groupEfLNS4_3useE2ELm8ELm16ELNS4_6layoutE3EEE", !65, i64 0}
!84 = distinct !{!84, !81}
!85 = distinct !{!85, !81}
!86 = !{!87, !65, i64 0}
!87 = !{!"_ZTSN4sycl3_V13ext6oneapi12experimental6matrix12joint_matrixINS0_9sub_groupENS2_8bfloat16ELNS4_3useE0ELm8ELm16ELNS4_6layoutE0EEE", !65, i64 0}
!88 = distinct !{!88, !81}
!89 = !{!90, !65, i64 0}
!90 = !{!"_ZTSN4sycl3_V13ext6oneapi12experimental6matrix12joint_matrixINS0_9sub_groupENS2_8bfloat16ELNS4_3useE1ELm16ELm16ELNS4_6layoutE2EEE", !65, i64 0}
!91 = distinct !{!91, !81}
!92 = !{!93}
!93 = distinct !{!93, !94, !"_ZN4sycl3_V13ext6oneapi12experimental6matrix16joint_matrix_madINS0_9sub_groupENS2_8bfloat16ES7_fLm8ELm16ELm16ELNS4_6layoutE0ELS8_2EEENS4_12joint_matrixIT_T2_LNS4_3useE2EXT3_EXT5_ELS8_3EEESA_RNS9_ISA_T0_LSC_0EXT3_EXT4_EXT6_EEERNS9_ISA_T1_LSC_1EXT4_EXT5_EXT7_EEERSD_: %agg.result"}
!94 = distinct !{!94, !"_ZN4sycl3_V13ext6oneapi12experimental6matrix16joint_matrix_madINS0_9sub_groupENS2_8bfloat16ES7_fLm8ELm16ELm16ELNS4_6layoutE0ELS8_2EEENS4_12joint_matrixIT_T2_LNS4_3useE2EXT3_EXT5_ELS8_3EEESA_RNS9_ISA_T0_LSC_0EXT3_EXT4_EXT6_EEERNS9_ISA_T1_LSC_1EXT4_EXT5_EXT7_EEERSD_"}
!95 = distinct !{!95, !81}
!96 = distinct !{!96, !81}
!97 = distinct !{!97, !81}
!98 = distinct !{!98, !81}
!99 = distinct !{!99, !81}
