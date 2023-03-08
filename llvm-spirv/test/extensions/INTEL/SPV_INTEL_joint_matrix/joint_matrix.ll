; RUN: llvm-as -opaque-pointers=0 < %s -o %t.bc
; RUN: llvm-spirv %t.bc -opaque-pointers=0 -spirv-ext=+SPV_INTEL_joint_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis -opaque-pointers=0 %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability JointMatrixINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_joint_matrix"
; CHECK-SPIRV: EntryPoint 6 [[#Kernel:]]

; CHECK-SPIRV-DAG: TypeInt [[#ShortTy:]] 16 0
; CHECK-SPIRV-DAG: TypeInt [[#CharTy:]] 8 0
; CHECK-SPIRV-DAG: TypeInt [[#IntTy:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#IntTy]] [[#Zero:]] 0
; CHECK-SPIRV-DAG: Constant [[#IntTy]] [[#Two:]] 2
; CHECK-SPIRV-DAG: Constant [[#IntTy]] [[#Three:]] 3
; CHECK-SPIRV-DAG: Constant [[#IntTy]] [[#Sixteen:]] 16
; CHECK-SPIRV-DAG: Constant [[#IntTy]] [[#FortyTwo:]] 42
; CHECK-SPIRV: TypeJointMatrixINTEL [[#CTy:]] [[#ShortTy]] [[#Two]] [[#Two]] [[#Zero]] [[#Three]]
; CHECK-SPIRV: TypeJointMatrixINTEL [[#ATy:]] [[#CharTy]] [[#Two]] [[#Sixteen]] [[#Zero]] [[#Three]] [[#Zero]]
; CHECK-SPIRV: TypeJointMatrixINTEL [[#BTy:]] [[#CharTy]] [[#Sixteen]] [[#Two]] [[#Three]] [[#Three]]

; CHECK-SPIRV: Function [[#]] [[#Kernel]]
; CHECK-SPIRV: FunctionParameter
; CHECK-SPIRV: FunctionParameter [[#]] [[#Stride:]]

; CHECK-SPIRV: Label [[#Entry:]]
; CHECK-SPIRV: JointMatrixLoadINTEL [[#CTy]] [[#CLoaded:]] [[#Cptr:]] [[#Stride]] [[#Zero]] [[#Three]] [[#Zero]]

; CHECK-SPIRV: Phi [[#CTy]] [[#C:]] [[#CLoaded]] [[#Entry]] [[#CMad:]] [[#ForBody:]]

; CHECK-SPIRV: Label [[#ForBody]]
; CHECK-SPIRV: JointMatrixLoadINTEL [[#ATy]] [[#A:]] [[#Aptr:]] [[#Stride]] [[#Zero]] [[#Three]] [[#Zero]]
; CHECK-SPIRV: JointMatrixLoadINTEL [[#BTy]] [[#B:]] [[#Bptr:]] [[#Stride]] [[#Zero]] [[#Three]] [[#Zero]]
; CHECK-SPIRV: JointMatrixMadINTEL [[#CTy]] [[#CMad]] [[#A]] [[#B]] [[#C]] [[#Three]]
; CHECK-SPIRV: JointMatrixSUMadINTEL [[#CTy]] [[#UnusedMad1:]] [[#A]] [[#B]] [[#C]] [[#Three]]
; CHECK-SPIRV: JointMatrixUSMadINTEL [[#CTy]] [[#UnusedMad2:]] [[#A]] [[#B]] [[#C]] [[#Three]]
; CHECK-SPIRV: JointMatrixUUMadINTEL [[#CTy]] [[#UnusedMad3:]] [[#A]] [[#B]] [[#C]] [[#Three]]

; CHECK-SPIRV: JointMatrixStoreINTEL [[#Cptr:]] [[#C]] [[#Stride]] [[#Zero]] [[#Three]] [[#Zero]]
; CHECK-SPIRV: CompositeConstruct [[#CTy]] [[#Cnew:]] [[#FortyTwo]]
; CHECK-SPIRV: Store [[#PtrToZero:]] [[#Zero]]
; CHECK-SPIRV: Load [[#]] [[#ZeroLoad:]] [[#PtrToZero]]
; CHECK-SPIRV: CompositeConstruct [[#CTy]] [[#CnewLoad:]] [[#ZeroLoad]]


; CHECK-LLVM: %spirv.JointMatrixINTEL._short_2_2_0_3
; CHECK-LLVM: %spirv.JointMatrixINTEL._char_2_16_0_3_0
; CHECK-LLVM: %spirv.JointMatrixINTEL._char_16_2_3_3

; CHECK-LLVM: [[CLoaded:%.*]] = call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(1)* @_Z77__spirv_JointMatrixLoadINTEL_RPU3AS139__spirv_JointMatrixINTEL__short_2_2_0_3PU3AS4sliii(i16 addrspace(4)* [[CPtr:%.*]], i64 [[Stride:%.*]], i32 0, i32 3, i32 0)
; CHECK-LLVM: [[C:%.*]] = phi %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(1)* [ [[CLoaded]], %entry ], [ [[CMad:%.*]], %for.body.i ]
; CHECK-LLVM: [[A:%.*]] = call spir_func %spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(1)* @_Z79__spirv_JointMatrixLoadINTEL_RPU3AS141__spirv_JointMatrixINTEL__char_2_16_0_3_0PU3AS4cliii(i8 addrspace(4)* [[APtr:%.*]], i64 [[Stride]], i32 0, i32 3, i32 0)
; CHECK-LLVM: [[B:%.*]] = call spir_func %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(1)* @_Z77__spirv_JointMatrixLoadINTEL_RPU3AS139__spirv_JointMatrixINTEL__char_16_2_3_3PU3AS4cliii(i8 addrspace(4)* [[BPtr:%.*]], i64 [[Stride]], i32 0, i32 3, i32 0)
; CHECK-LLVM: [[CMad1:%.*]] = call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(1)* @_Z27__spirv_JointMatrixMadINTELPU3AS141__spirv_JointMatrixINTEL__char_2_16_0_3_0PU3AS139__spirv_JointMatrixINTEL__char_16_2_3_3PU3AS139__spirv_JointMatrixINTEL__short_2_2_0_3i(%spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(1)* [[A]], %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(1)* [[B]], %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(1)* [[C]], i32 3)
; CHECK-LLVM: [[CMad2:%.*]] = call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(1)* @_Z29__spirv_JointMatrixSUMadINTELPU3AS141__spirv_JointMatrixINTEL__char_2_16_0_3_0PU3AS139__spirv_JointMatrixINTEL__char_16_2_3_3PU3AS139__spirv_JointMatrixINTEL__short_2_2_0_3i(%spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(1)* [[A]], %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(1)* [[B]], %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(1)* [[C]], i32 3)
; CHECK-LLVM: [[CMad3:%.*]] = call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(1)* @_Z29__spirv_JointMatrixUSMadINTELPU3AS141__spirv_JointMatrixINTEL__char_2_16_0_3_0PU3AS139__spirv_JointMatrixINTEL__char_16_2_3_3PU3AS139__spirv_JointMatrixINTEL__short_2_2_0_3i(%spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(1)* [[A]], %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(1)* [[B]], %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(1)* [[C]], i32 3)
; CHECK-LLVM: [[CMad4:%.*]] = call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(1)* @_Z29__spirv_JointMatrixUUMadINTELPU3AS141__spirv_JointMatrixINTEL__char_2_16_0_3_0PU3AS139__spirv_JointMatrixINTEL__char_16_2_3_3PU3AS139__spirv_JointMatrixINTEL__short_2_2_0_3i(%spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(1)* [[A]], %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(1)* [[B]], %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(1)* [[C]], i32 3)

; CHECK-LLVM: call spir_func void @_Z29__spirv_JointMatrixStoreINTELPU3AS4sPU3AS139__spirv_JointMatrixINTEL__short_2_2_0_3liii(i16 addrspace(4)* [[CPtr]], %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(1)* [[C]], i64 [[Stride]], i32 0, i32 3, i32 0)
; CHECK-LLVM: call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(1)* @_Z26__spirv_CompositeConstructi(i32 42)
; CHECK-LLVM: store i32 0, i32 addrspace(4)* [[StoredZero:%.*]], align 4
; CHECK-LLVM: [[LoadedZero:%.*]] = load i32, i32 addrspace(4)* [[StoredZero]], align 8
; CHECK-LLVM: call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(1)* @_Z26__spirv_CompositeConstructi(i32 [[LoadedZero]])

; ModuleID = 'joint_matrix_test-sycl-spir64-unknown-unknown.bc'
source_filename = "./joint_matrix_test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%spirv.JointMatrixINTEL._short_2_2_0_3 = type { [2 x [2 x [1 x [4 x [4 x i16]]]]]* }
%spirv.JointMatrixINTEL._char_2_16_0_3_0 = type { [2 x [16 x [1 x [4 x [1 x i8]]]]]* }
%spirv.JointMatrixINTEL._char_16_2_3_3 = type { [16 x [2 x [4 x [4 x i8]]]]* }

$_ZTSZ4mainE11matrix_test = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSZ4mainE11matrix_test(i16 addrspace(1)* %_arg_, i64 %_arg_1, i8 addrspace(1)* %_arg_3, i8 addrspace(1)* %_arg_5) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !5 !intel_reqd_sub_group_size !6 {
entry:
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32, !noalias !7
  %1 = extractelement <3 x i64> %0, i64 1
  %2 = extractelement <3 x i64> %0, i64 0
  %3 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInLocalInvocationId to <3 x i64> addrspace(4)*), align 32, !noalias !14
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
  %add.ptr.i51 = getelementptr inbounds i16, i16 addrspace(1)* %_arg_, i64 %mul6.i
  %add.ptr7.i52 = getelementptr inbounds i16, i16 addrspace(1)* %add.ptr.i51, i64 %sub5.i
  %add.ptr7.i = addrspacecast i16 addrspace(1)* %add.ptr7.i52 to i16 addrspace(4)*
  %call8.i = tail call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIsLm2ELm2ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(i16 addrspace(4)* %add.ptr7.i, i64 %_arg_1, i32 0, i32 3, i32 0) #3
  %add.ptr11.i53 = getelementptr inbounds i8, i8 addrspace(1)* %_arg_3, i64 %mul6.i
  %add.ptr16.i55 = getelementptr inbounds i8, i8 addrspace(1)* %_arg_5, i64 %sub5.i
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.body.i, %entry
  %k.0.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  %C.0.i = phi %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* [ %call8.i, %entry ], [ %call19.i, %for.body.i ]
  %cmp.i = icmp ult i32 %k.0.i, 32
  br i1 %cmp.i, label %for.body.i, label %_ZZ4mainENKUlN2cl4sycl7nd_itemILi2EEEE_clES2_.exit

for.body.i:                                       ; preds = %for.cond.i
  %idx.ext.i = zext i32 %k.0.i to i64
  %add.ptr12.i54 = getelementptr inbounds i8, i8 addrspace(1)* %add.ptr11.i53, i64 %idx.ext.i
  %add.ptr12.i = addrspacecast i8 addrspace(1)* %add.ptr12.i54 to i8 addrspace(4)*
  %call13.i = tail call spir_func %spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIaLm2ELm16ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(i8 addrspace(4)* %add.ptr12.i, i64 %_arg_1, i32 0, i32 3, i32 0) #3
  %mul14.i = shl nuw nsw i32 %k.0.i, 5
  %idx.ext15.i = zext i32 %mul14.i to i64
  %add.ptr17.i56 = getelementptr inbounds i8, i8 addrspace(1)* %add.ptr16.i55, i64 %idx.ext15.i
  %add.ptr17.i = addrspacecast i8 addrspace(1)* %add.ptr17.i56 to i8 addrspace(4)*
  %call18.i = tail call spir_func %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIaLm16ELm2ELN5__spv12MatrixLayoutE3ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(i8 addrspace(4)* %add.ptr17.i, i64 %_arg_1, i32 0, i32 3, i32 0) #3
  %call19.i = tail call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* @_Z27__spirv_JointMatrixMadINTELIasLm2ELm16ELm2ELN5__spv12MatrixLayoutE0ELS1_3ELS1_0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT6_EXT7_EEEPNS4_IT_XT1_EXT2_EXT4_EXT7_EEEPNS4_IS8_XT2_EXT3_EXT5_EXT7_EEES7_S3_(%spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(4)* %call13.i, %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(4)* %call18.i, %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* %C.0.i, i32 3) #3
  %call20.i = tail call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* @_Z29__spirv_JointMatrixSUMadINTELIasLm2ELm16ELm2ELN5__spv12MatrixLayoutE0ELS1_3ELS1_0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT6_EXT7_EEEPNS4_IT_XT1_EXT2_EXT4_EXT7_EEEPNS4_IS8_XT2_EXT3_EXT5_EXT7_EEES7_S3_(%spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(4)* %call13.i, %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(4)* %call18.i, %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* %C.0.i, i32 3) #3
  %call21.i = tail call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* @_Z29__spirv_JointMatrixUSMadINTELIasLm2ELm16ELm2ELN5__spv12MatrixLayoutE0ELS1_3ELS1_0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT6_EXT7_EEEPNS4_IT_XT1_EXT2_EXT4_EXT7_EEEPNS4_IS8_XT2_EXT3_EXT5_EXT7_EEES7_S3_(%spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(4)* %call13.i, %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(4)* %call18.i, %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* %C.0.i, i32 3) #3
  %call22.i = tail call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* @_Z29__spirv_JointMatrixUUMadINTELIasLm2ELm16ELm2ELN5__spv12MatrixLayoutE0ELS1_3ELS1_0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT6_EXT7_EEEPNS4_IT_XT1_EXT2_EXT4_EXT7_EEEPNS4_IS8_XT2_EXT3_EXT5_EXT7_EEES7_S3_(%spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(4)* %call13.i, %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(4)* %call18.i, %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* %C.0.i, i32 3) #3
  %add.i = add nuw nsw i32 %k.0.i, 16
  br label %for.cond.i, !llvm.loop !19

_ZZ4mainENKUlN2cl4sycl7nd_itemILi2EEEE_clES2_.exit: ; preds = %for.cond.i
  tail call spir_func void @_Z29__spirv_JointMatrixStoreINTELIsLm2ELm2ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEvPT_PNS0_24__spirv_JointMatrixINTELIS4_XT0_EXT1_EXT2_EXT3_EEEmS1_S3_i(i16 addrspace(4)* %add.ptr7.i, %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* %C.0.i, i64 %_arg_1, i32 0, i32 3, i32 0) #3
  %C.0.i.new = call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* @_Z26__spirv_CompositeConstructi(i32 42) #1
  %ref.tmp = alloca i32, align 4
  %ref.tmp.ascast = addrspacecast i32* %ref.tmp to i32 addrspace(4)*
  store i32 0, i32 addrspace(4)* %ref.tmp.ascast, align 4
  %zero = load i32, i32 addrspace(4)* %ref.tmp.ascast, align 8
  %C.0.i.new.load = call spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* @_Z26__spirv_CompositeConstructi(i32 %zero) #1

  ret void
}

; Function Attrs: convergent
declare dso_local spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIsLm2ELm2ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(i16 addrspace(4)*, i64, i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIaLm2ELm16ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(i8 addrspace(4)*, i64, i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIaLm16ELm2ELN5__spv12MatrixLayoutE3ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(i8 addrspace(4)*, i64, i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* @_Z27__spirv_JointMatrixMadINTELIasLm2ELm16ELm2ELN5__spv12MatrixLayoutE0ELS1_3ELS1_0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT6_EXT7_EEEPNS4_IT_XT1_EXT2_EXT4_EXT7_EEEPNS4_IS8_XT2_EXT3_EXT5_EXT7_EEES7_S3_(%spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(4)*, %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(4)*, %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)*, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* @_Z29__spirv_JointMatrixSUMadINTELIasLm2ELm16ELm2ELN5__spv12MatrixLayoutE0ELS1_3ELS1_0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT6_EXT7_EEEPNS4_IT_XT1_EXT2_EXT4_EXT7_EEEPNS4_IS8_XT2_EXT3_EXT5_EXT7_EEES7_S3_(%spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(4)*, %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(4)*, %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)*, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* @_Z29__spirv_JointMatrixUSMadINTELIasLm2ELm16ELm2ELN5__spv12MatrixLayoutE0ELS1_3ELS1_0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT6_EXT7_EEEPNS4_IT_XT1_EXT2_EXT4_EXT7_EEEPNS4_IS8_XT2_EXT3_EXT5_EXT7_EEES7_S3_(%spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(4)*, %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(4)*, %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)*, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* @_Z29__spirv_JointMatrixUUMadINTELIasLm2ELm16ELm2ELN5__spv12MatrixLayoutE0ELS1_3ELS1_0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT6_EXT7_EEEPNS4_IT_XT1_EXT2_EXT4_EXT7_EEEPNS4_IS8_XT2_EXT3_EXT5_EXT7_EEES7_S3_(%spirv.JointMatrixINTEL._char_2_16_0_3_0 addrspace(4)*, %spirv.JointMatrixINTEL._char_16_2_3_3 addrspace(4)*, %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)*, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z29__spirv_JointMatrixStoreINTELIsLm2ELm2ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEvPT_PNS0_24__spirv_JointMatrixINTELIS4_XT0_EXT1_EXT2_EXT3_EEEmS1_S3_i(i16 addrspace(4)*, %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)*, i64, i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %spirv.JointMatrixINTEL._short_2_2_0_3 addrspace(4)* @_Z26__spirv_CompositeConstructi(i32) #1

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #2

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="./joint_matrix_test.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #3 = { convergent }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 13.0.0 (https://github.com/intel/llvm.git b3243d9f711a1cd80681530d6017324796668d51)"}
!5 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!6 = !{i32 16}
!7 = !{!8, !10, !12}
!8 = distinct !{!8, !9, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv: %agg.result"}
!9 = distinct !{!9, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv"}
!10 = distinct !{!10, !11, !"_ZN7__spirvL22initGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v: %agg.result"}
!11 = distinct !{!11, !"_ZN7__spirvL22initGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v"}
!12 = distinct !{!12, !13, !"_ZN2cl4sycl6detail7Builder10getElementILi2EEEKNS0_7nd_itemIXT_EEEPS5_: %agg.result"}
!13 = distinct !{!13, !"_ZN2cl4sycl6detail7Builder10getElementILi2EEEKNS0_7nd_itemIXT_EEEPS5_"}
!14 = !{!15, !17, !12}
!15 = distinct !{!15, !16, !"_ZN7__spirv28InitSizesSTLocalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv: %agg.result"}
!16 = distinct !{!16, !"_ZN7__spirv28InitSizesSTLocalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv"}
!17 = distinct !{!17, !18, !"_ZN7__spirvL21initLocalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v: %agg.result"}
!18 = distinct !{!18, !"_ZN7__spirvL21initLocalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v"}
!19 = distinct !{!19, !20, !21}
!20 = !{!"llvm.loop.mustprogress"}
!21 = !{!"llvm.loop.unroll.disable"}


