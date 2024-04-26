// RUN: %clangxx -I %sycl_include -S -emit-llvm -fno-sycl-instrument-device-code -fsycl-device-only %s -o - | FileCheck %s

// This test checks the device code for various math operations on sycl::vec.
#include <sycl/sycl.hpp>

using namespace sycl;

// For testing binary operations.
#define CHECKBINOP(T, SUFFIX, OP)                                              \
  SYCL_EXTERNAL auto CheckDevCodeBINOP##SUFFIX(vec<T, 2> InVec##SUFFIX##2A,    \
                                               vec<T, 2> InVec##SUFFIX##2B) {  \
    return InVec##SUFFIX##2A OP InVec##SUFFIX##2B;                             \
  }

// For testing unary operators.
#define CHECKUOP(T, SUFFIX, OP, REF)                                           \
  SYCL_EXTERNAL auto CheckDevCodeUOP##SUFFIX(vec<T, 2> InVec##SUFFIX##2) {     \
    return OP InVec##SUFFIX##2;                                                \
  }

/********************** Binary Ops **********************/

// CHECK-LABEL: define dso_local spir_func void @_Z21CheckDevCodeBINOPINTAN4sycl3_V13vecIiLi2EEES2_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec") align 8 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec") align 8 [[INVECINTA2A:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec") align 8 [[INVECINTA2B:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] !srcloc [[META5:![0-9]+]] !sycl_fixed_targets [[META6:![0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META7:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i32>, ptr [[INVECINTA2A]], align 8, !tbaa [[TBAA10:![0-9]+]], !noalias [[META7]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i32>, ptr [[INVECINTA2B]], align 8, !tbaa [[TBAA10]], !noalias [[META7]]
// CHECK-NEXT:    [[ADD_I:%.*]] = add <2 x i32> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    store <2 x i32> [[ADD_I]], ptr addrspace(4) [[AGG_RESULT]], align 8, !tbaa [[TBAA10]], !alias.scope [[META7]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(int, INTA, +)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeBINOPBYTEAN4sycl3_V13vecISt4byteLi2EEES3_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.0") align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.0") align 2 [[INVECBYTEA2A:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.0") align 2 [[INVECBYTEA2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META13:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META14:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBYTEA2A]], align 2, !tbaa [[TBAA10]], !noalias [[META14]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i8>, ptr [[INVECBYTEA2B]], align 2, !tbaa [[TBAA10]], !noalias [[META14]]
// CHECK-NEXT:    [[ADD_I:%.*]] = add <2 x i8> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    store <2 x i8> [[ADD_I]], ptr addrspace(4) [[AGG_RESULT]], align 2, !tbaa [[TBAA10]], !alias.scope [[META14]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(std::byte, BYTEA, +)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeBINOPBOOLAN4sycl3_V13vecIbLi2EEES2_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.1") align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.1") align 2 [[INVECBOOLA2A:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.1") align 2 [[INVECBOOLA2B:%.*]]) local_unnamed_addr #[[ATTR1:[0-9]+]] !srcloc [[META17:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META18:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBOOLA2A]], align 2, !tbaa [[TBAA10]], !noalias [[META18]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i8>, ptr [[INVECBOOLA2B]], align 2, !tbaa [[TBAA10]], !noalias [[META18]]
// CHECK-NEXT:    [[ADD_I:%.*]] = add <2 x i8> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    br label [[FOR_COND_I_I:%.*]]
// CHECK:       for.cond.i.i:
// CHECK-NEXT:    [[VECINS_I_I6_I_I:%.*]] = phi <2 x i8> [ [[ADD_I]], [[ENTRY:%.*]] ], [ [[VECINS_I_I_I_I:%.*]], [[FOR_BODY_I_I:%.*]] ]
// CHECK-NEXT:    [[I_0_I_I:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[INC_I_I:%.*]], [[FOR_BODY_I_I]] ]
// CHECK-NEXT:    [[CMP_I_I:%.*]] = icmp ult i64 [[I_0_I_I]], 2
// CHECK-NEXT:    br i1 [[CMP_I_I]], label [[FOR_BODY_I_I]], label [[_ZN4SYCL3_V1PLERKNS0_3VECIBLI2EEES4__EXIT:%.*]]
// CHECK:       for.body.i.i:
// CHECK-NEXT:    [[CONV_I_I:%.*]] = trunc nuw nsw i64 [[I_0_I_I]] to i32
// CHECK-NEXT:    [[VECEXT_I_I_I_I:%.*]] = extractelement <2 x i8> [[VECINS_I_I6_I_I]], i32 [[CONV_I_I]]
// CHECK-NEXT:    [[TOBOOL_I_I_I_I:%.*]] = icmp ne i8 [[VECEXT_I_I_I_I]], 0
// CHECK-NEXT:    [[FROMBOOL_I_I:%.*]] = zext i1 [[TOBOOL_I_I_I_I]] to i8
// CHECK-NEXT:    [[VECINS_I_I_I_I]] = insertelement <2 x i8> [[VECINS_I_I6_I_I]], i8 [[FROMBOOL_I_I]], i32 [[CONV_I_I]]
// CHECK-NEXT:    [[INC_I_I]] = add nuw nsw i64 [[I_0_I_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I_I]], !llvm.loop [[LOOP21:![0-9]+]]
// CHECK:       _ZN4sycl3_V1plERKNS0_3vecIbLi2EEES4_.exit:
// CHECK-NEXT:    store <2 x i8> [[VECINS_I_I6_I_I]], ptr addrspace(4) [[AGG_RESULT]], align 2, !alias.scope [[META18]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(bool, BOOLA, +)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeBINOPHALFAN4sycl3_V13vecINS0_6detail9half_impl4halfELi2EEES5_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.2") align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.2") align 4 [[INVECHALFA2A:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.2") align 4 [[INVECHALFA2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META23:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META24:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x half>, ptr [[INVECHALFA2A]], align 4, !tbaa [[TBAA10]], !noalias [[META24]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x half>, ptr [[INVECHALFA2B]], align 4, !tbaa [[TBAA10]], !noalias [[META24]]
// CHECK-NEXT:    [[ADD_I:%.*]] = fadd <2 x half> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    store <2 x half> [[ADD_I]], ptr addrspace(4) [[AGG_RESULT]], align 4, !tbaa [[TBAA10]], !alias.scope [[META24]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(sycl::half, HALFA, +)

// CHECK-LABEL: define dso_local spir_func void @_Z20CheckDevCodeBINOPBFAN4sycl3_V13vecINS0_3ext6oneapi8bfloat16ELi2EEES5_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable sret(%"class.sycl::_V1::vec.3") align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.3") align 4 [[INVECBFA2A:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.3") align 4 [[INVECBFA2B:%.*]]) local_unnamed_addr #[[ATTR3:[0-9]+]] !srcloc [[META27:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[REF_TMP_I_I:%.*]] = alloca float, align 4
// CHECK-NEXT:    [[REF_TMP1_I:%.*]] = alloca %"class.sycl::_V1::ext::oneapi::bfloat16", align 2
// CHECK-NEXT:    [[REF_TMP3_I:%.*]] = alloca %"class.sycl::_V1::ext::oneapi::bfloat16", align 2
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META28:![0-9]+]])
// CHECK-NEXT:    [[REF_TMP1_ASCAST_I:%.*]] = addrspacecast ptr [[REF_TMP1_I]] to ptr addrspace(4)
// CHECK-NEXT:    [[REF_TMP3_ASCAST_I:%.*]] = addrspacecast ptr [[REF_TMP3_I]] to ptr addrspace(4)
// CHECK-NEXT:    [[REF_TMP_ASCAST_I_I:%.*]] = addrspacecast ptr [[REF_TMP_I_I]] to ptr addrspace(4)
// CHECK-NEXT:    [[AGG_RESULT_PROMOTED_I:%.*]] = load <2 x i16>, ptr addrspace(4) [[AGG_RESULT]], align 4, !alias.scope [[META28]]
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i16>, ptr [[INVECBFA2A]], align 4, !tbaa [[TBAA10]], !noalias [[META31:![0-9]+]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i16>, ptr [[INVECBFA2B]], align 4, !tbaa [[TBAA10]], !noalias [[META36:![0-9]+]]
// CHECK-NEXT:    br label [[FOR_COND_I:%.*]]
// CHECK:       for.cond.i:
// CHECK-NEXT:    [[VECINS_I_I10_I:%.*]] = phi <2 x i16> [ [[AGG_RESULT_PROMOTED_I]], [[ENTRY:%.*]] ], [ [[VECINS_I_I_I:%.*]], [[FOR_BODY_I:%.*]] ]
// CHECK-NEXT:    [[I_0_I:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[INC_I:%.*]], [[FOR_BODY_I]] ]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp ult i64 [[I_0_I]], 2
// CHECK-NEXT:    br i1 [[CMP_I]], label [[FOR_BODY_I]], label [[_ZN4SYCL3_V1PLERKNS0_3VECINS0_3EXT6ONEAPI8BFLOAT16ELI2EEES7__EXIT:%.*]]
// CHECK:       for.body.i:
// CHECK-NEXT:    [[CONV_I:%.*]] = trunc nuw nsw i64 [[I_0_I]] to i32
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 2, ptr nonnull [[REF_TMP1_I]]) #[[ATTR8:[0-9]+]], !noalias [[META28]]
// CHECK-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META41:![0-9]+]])
// CHECK-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META42:![0-9]+]])
// CHECK-NEXT:    [[VECEXT_I_I_I:%.*]] = extractelement <2 x i16> [[TMP0]], i32 [[CONV_I]]
// CHECK-NEXT:    store i16 [[VECEXT_I_I_I]], ptr [[REF_TMP1_I]], align 2, !alias.scope [[META43:![0-9]+]], !noalias [[META28]]
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 2, ptr nonnull [[REF_TMP3_I]]) #[[ATTR8]], !noalias [[META28]]
// CHECK-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META48:![0-9]+]])
// CHECK-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META49:![0-9]+]])
// CHECK-NEXT:    [[VECEXT_I_I9_I:%.*]] = extractelement <2 x i16> [[TMP1]], i32 [[CONV_I]]
// CHECK-NEXT:    store i16 [[VECEXT_I_I9_I]], ptr [[REF_TMP3_I]], align 2, !alias.scope [[META50:![0-9]+]], !noalias [[META28]]
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull [[REF_TMP_I_I]]) #[[ATTR8]], !noalias [[META55:![0-9]+]]
// CHECK-NEXT:    [[CALL_I_I_I_I:%.*]] = call spir_func noundef float @__devicelib_ConvertBF16ToFINTEL(ptr addrspace(4) noundef align 2 dereferenceable(2) [[REF_TMP1_ASCAST_I]]) #[[ATTR9:[0-9]+]], !noalias [[META55]]
// CHECK-NEXT:    [[CALL_I_I2_I_I:%.*]] = call spir_func noundef float @__devicelib_ConvertBF16ToFINTEL(ptr addrspace(4) noundef align 2 dereferenceable(2) [[REF_TMP3_ASCAST_I]]) #[[ATTR9]], !noalias [[META55]]
// CHECK-NEXT:    [[ADD_I_I:%.*]] = fadd float [[CALL_I_I_I_I]], [[CALL_I_I2_I_I]]
// CHECK-NEXT:    store float [[ADD_I_I]], ptr [[REF_TMP_I_I]], align 4, !tbaa [[TBAA58:![0-9]+]], !noalias [[META55]]
// CHECK-NEXT:    [[CALL_I_I3_I_I:%.*]] = call spir_func noundef zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4) [[REF_TMP_ASCAST_I_I]]) #[[ATTR9]], !noalias [[META55]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull [[REF_TMP_I_I]]) #[[ATTR8]], !noalias [[META55]]
// CHECK-NEXT:    [[VECINS_I_I_I]] = insertelement <2 x i16> [[VECINS_I_I10_I]], i16 [[CALL_I_I3_I_I]], i32 [[CONV_I]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 2, ptr nonnull [[REF_TMP3_I]]) #[[ATTR8]], !noalias [[META28]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 2, ptr nonnull [[REF_TMP1_I]]) #[[ATTR8]], !noalias [[META28]]
// CHECK-NEXT:    [[INC_I]] = add nuw nsw i64 [[I_0_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I]], !llvm.loop [[LOOP60:![0-9]+]]
// CHECK:       _ZN4sycl3_V1plERKNS0_3vecINS0_3ext6oneapi8bfloat16ELi2EEES7_.exit:
// CHECK-NEXT:    store <2 x i16> [[VECINS_I_I10_I]], ptr addrspace(4) [[AGG_RESULT]], align 4, !alias.scope [[META28]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(ext::oneapi::bfloat16, BFA, +)

// CHECK-LABEL: define dso_local spir_func void @_Z21CheckDevCodeBINOPINTLN4sycl3_V13vecIiLi2EEES2_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec") align 8 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec") align 8 [[INVECINTL2A:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec") align 8 [[INVECINTL2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META61:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META62:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i32>, ptr [[INVECINTL2A]], align 8, !tbaa [[TBAA10]], !noalias [[META62]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i32>, ptr [[INVECINTL2B]], align 8, !tbaa [[TBAA10]], !noalias [[META62]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp sgt <2 x i32> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK-NEXT:    store <2 x i32> [[SEXT_I]], ptr addrspace(4) [[AGG_RESULT]], align 8, !tbaa [[TBAA10]], !alias.scope [[META62]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(int, INTL, >)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeBINOPBYTELN4sycl3_V13vecISt4byteLi2EEES3_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.4") align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.0") align 2 [[INVECBYTEL2A:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.0") align 2 [[INVECBYTEL2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META65:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META66:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBYTEL2A]], align 2, !tbaa [[TBAA10]], !noalias [[META66]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i8>, ptr [[INVECBYTEL2B]], align 2, !tbaa [[TBAA10]], !noalias [[META66]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp sgt <2 x i8> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i8>
// CHECK-NEXT:    store <2 x i8> [[SEXT_I]], ptr addrspace(4) [[AGG_RESULT]], align 2, !tbaa [[TBAA10]], !alias.scope [[META66]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(std::byte, BYTEL, >)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeBINOPBOOLLN4sycl3_V13vecIbLi2EEES2_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.4") align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.1") align 2 [[INVECBOOLL2A:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.1") align 2 [[INVECBOOLL2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META69:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META70:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBOOLL2A]], align 2, !tbaa [[TBAA10]], !noalias [[META70]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i8>, ptr [[INVECBOOLL2B]], align 2, !tbaa [[TBAA10]], !noalias [[META70]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp sgt <2 x i8> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i8>
// CHECK-NEXT:    store <2 x i8> [[SEXT_I]], ptr addrspace(4) [[AGG_RESULT]], align 2, !tbaa [[TBAA10]], !alias.scope [[META70]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(bool, BOOLL, >)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeBINOPHALFLN4sycl3_V13vecINS0_6detail9half_impl4halfELi2EEES5_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.5") align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.2") align 4 [[INVECHALFL2A:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.2") align 4 [[INVECHALFL2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META73:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META74:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x half>, ptr [[INVECHALFL2A]], align 4, !tbaa [[TBAA10]], !noalias [[META74]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x half>, ptr [[INVECHALFL2B]], align 4, !tbaa [[TBAA10]], !noalias [[META74]]
// CHECK-NEXT:    [[CMP_I:%.*]] = fcmp ogt <2 x half> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i16>
// CHECK-NEXT:    store <2 x i16> [[SEXT_I]], ptr addrspace(4) [[AGG_RESULT]], align 4, !tbaa [[TBAA10]], !alias.scope [[META74]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(sycl::half, HALFL, >)

// CHECK-LABEL: define dso_local spir_func void @_Z20CheckDevCodeBINOPBFLN4sycl3_V13vecINS0_3ext6oneapi8bfloat16ELi2EEES5_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.5") align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.3") align 4 [[INVECBFL2A:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.3") align 4 [[INVECBFL2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META77:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META78:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i16>, ptr [[INVECBFL2A]], align 4, !tbaa [[TBAA10]], !noalias [[META78]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i16>, ptr [[INVECBFL2B]], align 4, !tbaa [[TBAA10]], !noalias [[META78]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp ugt <2 x i16> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i16>
// CHECK-NEXT:    store <2 x i16> [[SEXT_I]], ptr addrspace(4) [[AGG_RESULT]], align 4, !tbaa [[TBAA10]], !alias.scope [[META78]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(ext::oneapi::bfloat16, BFL, >)

/********************** Unary Ops **********************/

// CHECK-LABEL: define dso_local spir_func void @_Z21CheckDevCodeUOPINTNEGN4sycl3_V13vecIiLi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec") align 8 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec") align 8 [[INVECINTNEG2:%.*]]) local_unnamed_addr #[[ATTR1]] !srcloc [[META81:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[REF_TMP_I:%.*]] = alloca %"class.sycl::_V1::vec", align 8
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META82:![0-9]+]])
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 8, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META82]]
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i32>, ptr [[INVECINTNEG2]], align 8, !tbaa [[TBAA10]], !noalias [[META82]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp eq <2 x i32> [[TMP0]], zeroinitializer
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK-NEXT:    store <2 x i32> [[SEXT_I]], ptr [[REF_TMP_I]], align 8, !tbaa [[TBAA10]], !noalias [[META82]]
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META85:![0-9]+]])
// CHECK-NEXT:    br label [[FOR_COND_I_I_I:%.*]]
// CHECK:       for.cond.i.i.i:
// CHECK-NEXT:    [[I_0_I_I_I:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INC_I_I_I:%.*]], [[FOR_BODY_I_I_I:%.*]] ]
// CHECK-NEXT:    [[CMP_I_I_I:%.*]] = icmp ult i64 [[I_0_I_I_I]], 8
// CHECK-NEXT:    br i1 [[CMP_I_I_I]], label [[FOR_BODY_I_I_I]], label [[_ZN4SYCL3_V1NTERKNS0_3VECIILI2EEE_EXIT:%.*]]
// CHECK:       for.body.i.i.i:
// CHECK-NEXT:    [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[REF_TMP_I]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[ARRAYIDX_I_I_I]], align 1, !tbaa [[TBAA10]], !noalias [[META88:![0-9]+]]
// CHECK-NEXT:    [[ARRAYIDX1_I_I_I:%.*]] = getelementptr inbounds i8, ptr addrspace(4) [[AGG_RESULT]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    store i8 [[TMP1]], ptr addrspace(4) [[ARRAYIDX1_I_I_I]], align 1, !tbaa [[TBAA10]], !alias.scope [[META88]]
// CHECK-NEXT:    [[INC_I_I_I]] = add nuw nsw i64 [[I_0_I_I_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I_I_I]], !llvm.loop [[LOOP89:![0-9]+]]
// CHECK:       _ZN4sycl3_V1ntERKNS0_3vecIiLi2EEE.exit:
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 8, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META82]]
// CHECK-NEXT:    ret void
//
CHECKUOP(int, INTNEG, !, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z21CheckDevCodeUOPINTSUBN4sycl3_V13vecIiLi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec") align 8 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec") align 8 [[INVECINTSUB2:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META90:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META91:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i32>, ptr [[INVECINTSUB2]], align 8, !tbaa [[TBAA10]], !noalias [[META91]]
// CHECK-NEXT:    [[SUB_I:%.*]] = sub <2 x i32> zeroinitializer, [[TMP0]]
// CHECK-NEXT:    store <2 x i32> [[SUB_I]], ptr addrspace(4) [[AGG_RESULT]], align 8, !tbaa [[TBAA10]], !alias.scope [[META91]]
// CHECK-NEXT:    ret void
//
CHECKUOP(int, INTSUB, -, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeUOPBYTENEGN4sycl3_V13vecISt4byteLi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.4") align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.0") align 2 [[INVECBYTENEG2:%.*]]) local_unnamed_addr #[[ATTR1]] !srcloc [[META94:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[REF_TMP_I:%.*]] = alloca %"class.sycl::_V1::vec.0", align 2
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META95:![0-9]+]])
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 2, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META95]]
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBYTENEG2]], align 2, !tbaa [[TBAA10]], !noalias [[META95]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp eq <2 x i8> [[TMP0]], zeroinitializer
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i8>
// CHECK-NEXT:    store <2 x i8> [[SEXT_I]], ptr [[REF_TMP_I]], align 2, !tbaa [[TBAA10]], !noalias [[META95]]
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META98:![0-9]+]])
// CHECK-NEXT:    br label [[FOR_COND_I_I_I:%.*]]
// CHECK:       for.cond.i.i.i:
// CHECK-NEXT:    [[I_0_I_I_I:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INC_I_I_I:%.*]], [[FOR_BODY_I_I_I:%.*]] ]
// CHECK-NEXT:    [[CMP_I_I_I:%.*]] = icmp ult i64 [[I_0_I_I_I]], 2
// CHECK-NEXT:    br i1 [[CMP_I_I_I]], label [[FOR_BODY_I_I_I]], label [[_ZN4SYCL3_V1NTERKNS0_3VECIST4BYTELI2EEE_EXIT:%.*]]
// CHECK:       for.body.i.i.i:
// CHECK-NEXT:    [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[REF_TMP_I]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[ARRAYIDX_I_I_I]], align 1, !tbaa [[TBAA10]], !noalias [[META101:![0-9]+]]
// CHECK-NEXT:    [[ARRAYIDX1_I_I_I:%.*]] = getelementptr inbounds i8, ptr addrspace(4) [[AGG_RESULT]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    store i8 [[TMP1]], ptr addrspace(4) [[ARRAYIDX1_I_I_I]], align 1, !tbaa [[TBAA10]], !alias.scope [[META101]]
// CHECK-NEXT:    [[INC_I_I_I]] = add nuw nsw i64 [[I_0_I_I_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I_I_I]], !llvm.loop [[LOOP89]]
// CHECK:       _ZN4sycl3_V1ntERKNS0_3vecISt4byteLi2EEE.exit:
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 2, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META95]]
// CHECK-NEXT:    ret void
//
CHECKUOP(std::byte, BYTENEG, !, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeUOPBYTESUBN4sycl3_V13vecISt4byteLi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.0") align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.0") align 2 [[INVECBYTESUB2:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META102:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META103:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBYTESUB2]], align 2, !tbaa [[TBAA10]], !noalias [[META103]]
// CHECK-NEXT:    [[SUB_I:%.*]] = sub <2 x i8> zeroinitializer, [[TMP0]]
// CHECK-NEXT:    store <2 x i8> [[SUB_I]], ptr addrspace(4) [[AGG_RESULT]], align 2, !tbaa [[TBAA10]], !alias.scope [[META103]]
// CHECK-NEXT:    ret void
//
CHECKUOP(std::byte, BYTESUB, -, -1)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeUOPBOOLNEGN4sycl3_V13vecIbLi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.4") align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.1") align 2 [[INVECBOOLNEG2:%.*]]) local_unnamed_addr #[[ATTR1]] !srcloc [[META106:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[REF_TMP_I:%.*]] = alloca %"class.sycl::_V1::vec.1", align 2
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META107:![0-9]+]])
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 2, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META107]]
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBOOLNEG2]], align 2, !tbaa [[TBAA10]], !noalias [[META107]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp eq <2 x i8> [[TMP0]], zeroinitializer
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i8>
// CHECK-NEXT:    store <2 x i8> [[SEXT_I]], ptr [[REF_TMP_I]], align 2, !tbaa [[TBAA10]], !noalias [[META107]]
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META110:![0-9]+]])
// CHECK-NEXT:    br label [[FOR_COND_I_I_I:%.*]]
// CHECK:       for.cond.i.i.i:
// CHECK-NEXT:    [[I_0_I_I_I:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INC_I_I_I:%.*]], [[FOR_BODY_I_I_I:%.*]] ]
// CHECK-NEXT:    [[CMP_I_I_I:%.*]] = icmp ult i64 [[I_0_I_I_I]], 2
// CHECK-NEXT:    br i1 [[CMP_I_I_I]], label [[FOR_BODY_I_I_I]], label [[_ZN4SYCL3_V1NTERKNS0_3VECIBLI2EEE_EXIT:%.*]]
// CHECK:       for.body.i.i.i:
// CHECK-NEXT:    [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[REF_TMP_I]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[ARRAYIDX_I_I_I]], align 1, !tbaa [[TBAA10]], !noalias [[META113:![0-9]+]]
// CHECK-NEXT:    [[ARRAYIDX1_I_I_I:%.*]] = getelementptr inbounds i8, ptr addrspace(4) [[AGG_RESULT]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    store i8 [[TMP1]], ptr addrspace(4) [[ARRAYIDX1_I_I_I]], align 1, !tbaa [[TBAA10]], !alias.scope [[META113]]
// CHECK-NEXT:    [[INC_I_I_I]] = add nuw nsw i64 [[I_0_I_I_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I_I_I]], !llvm.loop [[LOOP89]]
// CHECK:       _ZN4sycl3_V1ntERKNS0_3vecIbLi2EEE.exit:
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 2, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META107]]
// CHECK-NEXT:    ret void
//
CHECKUOP(bool, BOOLNEG, !, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeUOPHALFNEGN4sycl3_V13vecINS0_6detail9half_impl4halfELi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.5") align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.2") align 4 [[INVECHALFNEG2:%.*]]) local_unnamed_addr #[[ATTR1]] !srcloc [[META114:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[REF_TMP_I:%.*]] = alloca %"class.sycl::_V1::vec.2", align 4
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META115:![0-9]+]])
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META115]]
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x half>, ptr [[INVECHALFNEG2]], align 4, !tbaa [[TBAA10]], !noalias [[META115]]
// CHECK-NEXT:    [[CMP_I:%.*]] = fcmp oeq <2 x half> [[TMP0]], zeroinitializer
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i16>
// CHECK-NEXT:    store <2 x i16> [[SEXT_I]], ptr [[REF_TMP_I]], align 4, !tbaa [[TBAA10]], !noalias [[META115]]
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META118:![0-9]+]])
// CHECK-NEXT:    br label [[FOR_COND_I_I_I:%.*]]
// CHECK:       for.cond.i.i.i:
// CHECK-NEXT:    [[I_0_I_I_I:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INC_I_I_I:%.*]], [[FOR_BODY_I_I_I:%.*]] ]
// CHECK-NEXT:    [[CMP_I_I_I:%.*]] = icmp ult i64 [[I_0_I_I_I]], 4
// CHECK-NEXT:    br i1 [[CMP_I_I_I]], label [[FOR_BODY_I_I_I]], label [[_ZN4SYCL3_V1NTERKNS0_3VECINS0_6DETAIL9HALF_IMPL4HALFELI2EEE_EXIT:%.*]]
// CHECK:       for.body.i.i.i:
// CHECK-NEXT:    [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[REF_TMP_I]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[ARRAYIDX_I_I_I]], align 1, !tbaa [[TBAA10]], !noalias [[META121:![0-9]+]]
// CHECK-NEXT:    [[ARRAYIDX1_I_I_I:%.*]] = getelementptr inbounds i8, ptr addrspace(4) [[AGG_RESULT]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    store i8 [[TMP1]], ptr addrspace(4) [[ARRAYIDX1_I_I_I]], align 1, !tbaa [[TBAA10]], !alias.scope [[META121]]
// CHECK-NEXT:    [[INC_I_I_I]] = add nuw nsw i64 [[I_0_I_I_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I_I_I]], !llvm.loop [[LOOP89]]
// CHECK:       _ZN4sycl3_V1ntERKNS0_3vecINS0_6detail9half_impl4halfELi2EEE.exit:
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META115]]
// CHECK-NEXT:    ret void
//
CHECKUOP(sycl::half, HALFNEG, !, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeUOPHALFSUBN4sycl3_V13vecINS0_6detail9half_impl4halfELi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.2") align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.2") align 4 [[INVECHALFSUB2:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META122:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META123:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x half>, ptr [[INVECHALFSUB2]], align 4, !tbaa [[TBAA10]], !noalias [[META123]]
// CHECK-NEXT:    [[FNEG_I:%.*]] = fneg <2 x half> [[TMP0]]
// CHECK-NEXT:    store <2 x half> [[FNEG_I]], ptr addrspace(4) [[AGG_RESULT]], align 4, !tbaa [[TBAA10]], !alias.scope [[META123]]
// CHECK-NEXT:    ret void
//
CHECKUOP(sycl::half, HALFSUB, -, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z20CheckDevCodeUOPBFNEGN4sycl3_V13vecINS0_3ext6oneapi8bfloat16ELi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.5") align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.3") align 4 [[INVECBFNEG2:%.*]]) local_unnamed_addr #[[ATTR3]] !srcloc [[META126:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RET_I:%.*]] = alloca %"class.sycl::_V1::vec.3", align 4
// CHECK-NEXT:    [[REF_TMP1_I:%.*]] = alloca float, align 4
// CHECK-NEXT:    [[REF_TMP2_I:%.*]] = alloca %"class.sycl::_V1::ext::oneapi::bfloat16", align 2
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META127:![0-9]+]])
// CHECK-NEXT:    [[REF_TMP1_ASCAST_I:%.*]] = addrspacecast ptr [[REF_TMP1_I]] to ptr addrspace(4)
// CHECK-NEXT:    [[REF_TMP2_ASCAST_I:%.*]] = addrspacecast ptr [[REF_TMP2_I]] to ptr addrspace(4)
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull [[RET_I]]) #[[ATTR8]], !noalias [[META127]]
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i16>, ptr [[INVECBFNEG2]], align 4, !tbaa [[TBAA10]], !noalias [[META130:![0-9]+]]
// CHECK-NEXT:    br label [[FOR_COND_I:%.*]]
// CHECK:       for.cond.i:
// CHECK-NEXT:    [[TMP1:%.*]] = phi <2 x i16> [ zeroinitializer, [[ENTRY:%.*]] ], [ [[VECINS_I_I_I:%.*]], [[FOR_BODY_I:%.*]] ]
// CHECK-NEXT:    [[I_0_I:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[INC_I:%.*]], [[FOR_BODY_I]] ]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp ult i64 [[I_0_I]], 2
// CHECK-NEXT:    br i1 [[CMP_I]], label [[FOR_BODY_I]], label [[FOR_COND_CLEANUP_I:%.*]]
// CHECK:       for.cond.cleanup.i:
// CHECK-NEXT:    store <2 x i16> [[TMP1]], ptr [[RET_I]], align 1, !noalias [[META127]]
// CHECK-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META135:![0-9]+]])
// CHECK-NEXT:    br label [[FOR_COND_I_I_I:%.*]]
// CHECK:       for.cond.i.i.i:
// CHECK-NEXT:    [[I_0_I_I_I:%.*]] = phi i64 [ 0, [[FOR_COND_CLEANUP_I]] ], [ [[INC_I_I_I:%.*]], [[FOR_BODY_I_I_I:%.*]] ]
// CHECK-NEXT:    [[CMP_I_I_I:%.*]] = icmp ult i64 [[I_0_I_I_I]], 4
// CHECK-NEXT:    br i1 [[CMP_I_I_I]], label [[FOR_BODY_I_I_I]], label [[_ZN4SYCL3_V1NTERKNS0_3VECINS0_3EXT6ONEAPI8BFLOAT16ELI2EEE_EXIT:%.*]]
// CHECK:       for.body.i.i.i:
// CHECK-NEXT:    [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[RET_I]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    [[TMP2:%.*]] = load i8, ptr [[ARRAYIDX_I_I_I]], align 1, !tbaa [[TBAA10]], !noalias [[META138:![0-9]+]]
// CHECK-NEXT:    [[ARRAYIDX1_I_I_I:%.*]] = getelementptr inbounds i8, ptr addrspace(4) [[AGG_RESULT]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    store i8 [[TMP2]], ptr addrspace(4) [[ARRAYIDX1_I_I_I]], align 1, !tbaa [[TBAA10]], !alias.scope [[META138]]
// CHECK-NEXT:    [[INC_I_I_I]] = add nuw nsw i64 [[I_0_I_I_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I_I_I]], !llvm.loop [[LOOP89]]
// CHECK:       for.body.i:
// CHECK-NEXT:    [[CONV_I:%.*]] = trunc nuw nsw i64 [[I_0_I]] to i32
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull [[REF_TMP1_I]]) #[[ATTR8]], !noalias [[META127]]
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 2, ptr nonnull [[REF_TMP2_I]]) #[[ATTR8]], !noalias [[META127]]
// CHECK-NEXT:    [[VECEXT_I_I_I:%.*]] = extractelement <2 x i16> [[TMP0]], i32 [[CONV_I]]
// CHECK-NEXT:    store i16 [[VECEXT_I_I_I]], ptr [[REF_TMP2_I]], align 2, !tbaa [[TBAA139:![0-9]+]], !alias.scope [[META141:![0-9]+]], !noalias [[META127]]
// CHECK-NEXT:    [[CALL_I_I_I:%.*]] = call spir_func noundef float @__devicelib_ConvertBF16ToFINTEL(ptr addrspace(4) noundef align 2 dereferenceable(2) [[REF_TMP2_ASCAST_I]]) #[[ATTR9]], !noalias [[META127]]
// CHECK-NEXT:    [[CMP_I_I:%.*]] = fcmp oeq float [[CALL_I_I_I]], 0.000000e+00
// CHECK-NEXT:    [[CONV4_I:%.*]] = uitofp i1 [[CMP_I_I]] to float
// CHECK-NEXT:    store float [[CONV4_I]], ptr [[REF_TMP1_I]], align 4, !tbaa [[TBAA58]], !noalias [[META127]]
// CHECK-NEXT:    [[CALL_I_I10_I:%.*]] = call spir_func noundef zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4) [[REF_TMP1_ASCAST_I]]) #[[ATTR9]], !noalias [[META127]]
// CHECK-NEXT:    [[VECINS_I_I_I]] = insertelement <2 x i16> [[TMP1]], i16 [[CALL_I_I10_I]], i32 [[CONV_I]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 2, ptr nonnull [[REF_TMP2_I]]) #[[ATTR8]], !noalias [[META127]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull [[REF_TMP1_I]]) #[[ATTR8]], !noalias [[META127]]
// CHECK-NEXT:    [[INC_I]] = add nuw nsw i64 [[I_0_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I]], !llvm.loop [[LOOP144:![0-9]+]]
// CHECK:       _ZN4sycl3_V1ntERKNS0_3vecINS0_3ext6oneapi8bfloat16ELi2EEE.exit:
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull [[RET_I]]) #[[ATTR8]], !noalias [[META127]]
// CHECK-NEXT:    ret void
//
CHECKUOP(ext::oneapi::bfloat16, BFNEG, !, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z20CheckDevCodeUOPBFSUBN4sycl3_V13vecINS0_3ext6oneapi8bfloat16ELi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.3") align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval(%"class.sycl::_V1::vec.3") align 4 [[INVECBFSUB2:%.*]]) local_unnamed_addr #[[ATTR5:[0-9]+]] !srcloc [[META145:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[REF_TMP_I_I:%.*]] = alloca float, align 4
// CHECK-NEXT:    [[V_I:%.*]] = alloca %"class.sycl::_V1::ext::oneapi::bfloat16", align 2
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META146:![0-9]+]])
// CHECK-NEXT:    [[V_ASCAST_I:%.*]] = addrspacecast ptr [[V_I]] to ptr addrspace(4)
// CHECK-NEXT:    store i32 0, ptr addrspace(4) [[AGG_RESULT]], align 4, !alias.scope [[META146]]
// CHECK-NEXT:    [[REF_TMP_ASCAST_I_I:%.*]] = addrspacecast ptr [[REF_TMP_I_I]] to ptr addrspace(4)
// CHECK-NEXT:    br label [[FOR_COND_I:%.*]]
// CHECK:       for.cond.i:
// CHECK-NEXT:    [[I_0_I:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INC_I:%.*]], [[FOR_COND_I]] ]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp ult i64 [[I_0_I]], 2
// CHECK-NEXT:    call void @llvm.assume(i1 [[CMP_I]])
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 2, ptr nonnull [[V_I]]) #[[ATTR8]], !noalias [[META146]]
// CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds <2 x i16>, ptr [[INVECBFSUB2]], i64 0, i64 [[I_0_I]]
// CHECK-NEXT:    [[VECEXT_I:%.*]] = load i16, ptr [[TMP0]], align 2, !noalias [[META146]]
// CHECK-NEXT:    store i16 [[VECEXT_I]], ptr [[V_I]], align 2, !tbaa [[TBAA149:![0-9]+]], !alias.scope [[META151:![0-9]+]], !noalias [[META146]]
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull [[REF_TMP_I_I]]) #[[ATTR8]], !noalias [[META154:![0-9]+]]
// CHECK-NEXT:    [[CALL_I_I:%.*]] = call spir_func float @__devicelib_ConvertBF16ToFINTEL(ptr addrspace(4) noundef align 2 dereferenceable(2) [[V_ASCAST_I]]) #[[ATTR9]], !noalias [[META154]]
// CHECK-NEXT:    [[FNEG_I_I:%.*]] = fneg float [[CALL_I_I]]
// CHECK-NEXT:    store float [[FNEG_I_I]], ptr [[REF_TMP_I_I]], align 4, !tbaa [[TBAA58]], !noalias [[META154]]
// CHECK-NEXT:    [[CALL_I_I_I_I:%.*]] = call spir_func noundef zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4) [[REF_TMP_ASCAST_I_I]]) #[[ATTR9]], !noalias [[META154]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull [[REF_TMP_I_I]]) #[[ATTR8]], !noalias [[META154]]
// CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds <2 x i16>, ptr addrspace(4) [[AGG_RESULT]], i64 0, i64 [[I_0_I]]
// CHECK-NEXT:    store i16 [[CALL_I_I_I_I]], ptr addrspace(4) [[TMP1]], align 2, !alias.scope [[META146]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 2, ptr nonnull [[V_I]]) #[[ATTR8]], !noalias [[META146]]
// CHECK-NEXT:    [[INC_I]] = add nuw nsw i64 [[I_0_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I]], !llvm.loop [[LOOP157:![0-9]+]]
//
CHECKUOP(ext::oneapi::bfloat16, BFSUB, -, 1)
