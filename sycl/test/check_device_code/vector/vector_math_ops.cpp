// RUN: %clangxx -I %sycl_include -S -emit-llvm -fno-sycl-instrument-device-code -fsycl-device-only %s -o - | FileCheck %s

// This test checks
// (1) the storage type of sycl::vec on device for all data types, and
// (2) the device code for various math operations on sycl::vec.
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

/*************** sycl::vec Storage Types ****************/

// CHECK: [[VECINT2:.*]] = type { <2 x i32> }
// CHECK: [[VECFLOAT2:.*]] = type { <2 x float> }
// CHECK: [[VECCHAR2:.*]] = type { <2 x i8> }
// CHECK: [[VECBYTE2:.*]] = type { <2 x i8> }
// CHECK: [[VECBOOL2:.*]] = type { <2 x i8> }
// CHECK: [[VECHALF2:.*]] = type { <2 x half> }
// CHECK: [[VECBF2:.*]] = type { <2 x i16> }

/*************** Binary Arithmetic Ops ******************/

// CHECK-LABEL: define dso_local spir_func void @_Z21CheckDevCodeBINOPINTAN4sycl3_V13vecIiLi2EEES2_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret([[VECINT2]]) align 8 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECINT2]]) align 8 [[INVECINTA2A:%.*]], ptr nocapture noundef readonly byval([[VECINT2]]) align 8 [[INVECINTA2B:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] !srcloc [[META5:![0-9]+]] !sycl_fixed_targets [[META6:![0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META7:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i32>, ptr [[INVECINTA2A]], align 8, !tbaa [[TBAA10:![0-9]+]], !noalias [[META7]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i32>, ptr [[INVECINTA2B]], align 8, !tbaa [[TBAA10]], !noalias [[META7]]
// CHECK-NEXT:    [[ADD_I:%.*]] = add <2 x i32> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    store <2 x i32> [[ADD_I]], ptr addrspace(4) [[AGG_RESULT]], align 8, !tbaa [[TBAA10]], !alias.scope [[META7]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(int, INTA, +)

// CHECK-LABEL: define dso_local spir_func void @_Z23CheckDevCodeBINOPFLOATAN4sycl3_V13vecIfLi2EEES2_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret([[VECFLOAT2]]) align 8 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECFLOAT2]]) align 8 [[INVECFLOATA2A:%.*]], ptr nocapture noundef readonly byval([[VECFLOAT2]]) align 8 [[INVECFLOATA2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META13:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META14:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x float>, ptr [[INVECFLOATA2A]], align 8, !tbaa [[TBAA10]], !noalias [[META14]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x float>, ptr [[INVECFLOATA2B]], align 8, !tbaa [[TBAA10]], !noalias [[META14]]
// CHECK-NEXT:    [[ADD_I:%.*]] = fadd <2 x float> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    store <2 x float> [[ADD_I]], ptr addrspace(4) [[AGG_RESULT]], align 8, !tbaa [[TBAA10]], !alias.scope [[META14]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(float, FLOATA, +)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeBINOPCHARAN4sycl3_V13vecIcLi2EEES2_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret([[VECCHAR2]]) align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECCHAR2]]) align 2 [[INVECCHARA2A:%.*]], ptr nocapture noundef readonly byval([[VECCHAR2]]) align 2 [[INVECCHARA2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META17:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META18:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECCHARA2A]], align 2, !tbaa [[TBAA10]], !noalias [[META18]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i8>, ptr [[INVECCHARA2B]], align 2, !tbaa [[TBAA10]], !noalias [[META18]]
// CHECK-NEXT:    [[ADD_I:%.*]] = add <2 x i8> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    store <2 x i8> [[ADD_I]], ptr addrspace(4) [[AGG_RESULT]], align 2, !tbaa [[TBAA10]], !alias.scope [[META18]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(char, CHARA, +)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeBINOPBYTEAN4sycl3_V13vecISt4byteLi2EEES3_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret([[VECBYTE2]]) align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECBYTE2]]) align 2 [[INVECBYTEA2A:%.*]], ptr nocapture noundef readonly byval([[VECBYTE2]]) align 2 [[INVECBYTEA2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META21:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META22:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBYTEA2A]], align 2, !tbaa [[TBAA10]], !noalias [[META22]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i8>, ptr [[INVECBYTEA2B]], align 2, !tbaa [[TBAA10]], !noalias [[META22]]
// CHECK-NEXT:    [[ADD_I:%.*]] = add <2 x i8> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    store <2 x i8> [[ADD_I]], ptr addrspace(4) [[AGG_RESULT]], align 2, !tbaa [[TBAA10]], !alias.scope [[META22]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(std::byte, BYTEA, +)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeBINOPBOOLAN4sycl3_V13vecIbLi2EEES2_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret([[VECBOOL2]]) align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECBOOL2]]) align 2 [[INVECBOOLA2A:%.*]], ptr nocapture noundef readonly byval([[VECBOOL2]]) align 2 [[INVECBOOLA2B:%.*]]) local_unnamed_addr #[[ATTR1:[0-9]+]] !srcloc [[META25:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META26:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBOOLA2A]], align 2, !tbaa [[TBAA10]], !noalias [[META26]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i8>, ptr [[INVECBOOLA2B]], align 2, !tbaa [[TBAA10]], !noalias [[META26]]
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
// CHECK-NEXT:    br label [[FOR_COND_I_I]], !llvm.loop [[LOOP29:![0-9]+]]
// CHECK:       _ZN4sycl3_V1plERKNS0_3vecIbLi2EEES4_.exit:
// CHECK-NEXT:    store <2 x i8> [[VECINS_I_I6_I_I]], ptr addrspace(4) [[AGG_RESULT]], align 2, !alias.scope [[META26]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(bool, BOOLA, +)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeBINOPHALFAN4sycl3_V13vecINS0_6detail9half_impl4halfELi2EEES5_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret([[VECHALF2]]) align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECHALF2]]) align 4 [[INVECHALFA2A:%.*]], ptr nocapture noundef readonly byval([[VECHALF2]]) align 4 [[INVECHALFA2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META31:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META32:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x half>, ptr [[INVECHALFA2A]], align 4, !tbaa [[TBAA10]], !noalias [[META32]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x half>, ptr [[INVECHALFA2B]], align 4, !tbaa [[TBAA10]], !noalias [[META32]]
// CHECK-NEXT:    [[ADD_I:%.*]] = fadd <2 x half> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    store <2 x half> [[ADD_I]], ptr addrspace(4) [[AGG_RESULT]], align 4, !tbaa [[TBAA10]], !alias.scope [[META32]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(sycl::half, HALFA, +)

// CHECK-LABEL: define dso_local spir_func void @_Z20CheckDevCodeBINOPBFAN4sycl3_V13vecINS0_3ext6oneapi8bfloat16ELi2EEES5_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable sret([[VECBF2]]) align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECBF2]]) align 4 [[INVECBFA2A:%.*]], ptr nocapture noundef readonly byval([[VECBF2]]) align 4 [[INVECBFA2B:%.*]]) local_unnamed_addr #[[ATTR3:[0-9]+]] !srcloc [[META35:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[REF_TMP_I_I:%.*]] = alloca float, align 4
// CHECK-NEXT:    [[REF_TMP1_I:%.*]] = alloca %"class.sycl::_V1::ext::oneapi::bfloat16", align 2
// CHECK-NEXT:    [[REF_TMP3_I:%.*]] = alloca %"class.sycl::_V1::ext::oneapi::bfloat16", align 2
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META36:![0-9]+]])
// CHECK-NEXT:    [[REF_TMP1_ASCAST_I:%.*]] = addrspacecast ptr [[REF_TMP1_I]] to ptr addrspace(4)
// CHECK-NEXT:    [[REF_TMP3_ASCAST_I:%.*]] = addrspacecast ptr [[REF_TMP3_I]] to ptr addrspace(4)
// CHECK-NEXT:    [[REF_TMP_ASCAST_I_I:%.*]] = addrspacecast ptr [[REF_TMP_I_I]] to ptr addrspace(4)
// CHECK-NEXT:    [[AGG_RESULT_PROMOTED_I:%.*]] = load <2 x i16>, ptr addrspace(4) [[AGG_RESULT]], align 4, !alias.scope [[META36]]
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i16>, ptr [[INVECBFA2A]], align 4, !tbaa [[TBAA10]], !noalias [[META39:![0-9]+]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i16>, ptr [[INVECBFA2B]], align 4, !tbaa [[TBAA10]], !noalias [[META44:![0-9]+]]
// CHECK-NEXT:    br label [[FOR_COND_I:%.*]]
// CHECK:       for.cond.i:
// CHECK-NEXT:    [[VECINS_I_I10_I:%.*]] = phi <2 x i16> [ [[AGG_RESULT_PROMOTED_I]], [[ENTRY:%.*]] ], [ [[VECINS_I_I_I:%.*]], [[FOR_BODY_I:%.*]] ]
// CHECK-NEXT:    [[I_0_I:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[INC_I:%.*]], [[FOR_BODY_I]] ]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp ult i64 [[I_0_I]], 2
// CHECK-NEXT:    br i1 [[CMP_I]], label [[FOR_BODY_I]], label [[_ZN4SYCL3_V1PLERKNS0_3VECINS0_3EXT6ONEAPI8BFLOAT16ELI2EEES7__EXIT:%.*]]
// CHECK:       for.body.i:
// CHECK-NEXT:    [[CONV_I:%.*]] = trunc nuw nsw i64 [[I_0_I]] to i32
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 2, ptr nonnull [[REF_TMP1_I]]) #[[ATTR8:[0-9]+]], !noalias [[META36]]
// CHECK-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META49:![0-9]+]])
// CHECK-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META50:![0-9]+]])
// CHECK-NEXT:    [[VECEXT_I_I_I:%.*]] = extractelement <2 x i16> [[TMP0]], i32 [[CONV_I]]
// CHECK-NEXT:    store i16 [[VECEXT_I_I_I]], ptr [[REF_TMP1_I]], align 2, !alias.scope [[META51:![0-9]+]], !noalias [[META36]]
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 2, ptr nonnull [[REF_TMP3_I]]) #[[ATTR8]], !noalias [[META36]]
// CHECK-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META56:![0-9]+]])
// CHECK-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META57:![0-9]+]])
// CHECK-NEXT:    [[VECEXT_I_I9_I:%.*]] = extractelement <2 x i16> [[TMP1]], i32 [[CONV_I]]
// CHECK-NEXT:    store i16 [[VECEXT_I_I9_I]], ptr [[REF_TMP3_I]], align 2, !alias.scope [[META58:![0-9]+]], !noalias [[META36]]
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull [[REF_TMP_I_I]]) #[[ATTR8]], !noalias [[META63:![0-9]+]]
// CHECK-NEXT:    [[CALL_I_I_I_I:%.*]] = call spir_func noundef float @__devicelib_ConvertBF16ToFINTEL(ptr addrspace(4) noundef align 2 dereferenceable(2) [[REF_TMP1_ASCAST_I]]) #[[ATTR9:[0-9]+]], !noalias [[META63]]
// CHECK-NEXT:    [[CALL_I_I2_I_I:%.*]] = call spir_func noundef float @__devicelib_ConvertBF16ToFINTEL(ptr addrspace(4) noundef align 2 dereferenceable(2) [[REF_TMP3_ASCAST_I]]) #[[ATTR9]], !noalias [[META63]]
// CHECK-NEXT:    [[ADD_I_I:%.*]] = fadd float [[CALL_I_I_I_I]], [[CALL_I_I2_I_I]]
// CHECK-NEXT:    store float [[ADD_I_I]], ptr [[REF_TMP_I_I]], align 4, !tbaa [[TBAA66:![0-9]+]], !noalias [[META63]]
// CHECK-NEXT:    [[CALL_I_I3_I_I:%.*]] = call spir_func noundef zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4) [[REF_TMP_ASCAST_I_I]]) #[[ATTR9]], !noalias [[META63]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull [[REF_TMP_I_I]]) #[[ATTR8]], !noalias [[META63]]
// CHECK-NEXT:    [[VECINS_I_I_I]] = insertelement <2 x i16> [[VECINS_I_I10_I]], i16 [[CALL_I_I3_I_I]], i32 [[CONV_I]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 2, ptr nonnull [[REF_TMP3_I]]) #[[ATTR8]], !noalias [[META36]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 2, ptr nonnull [[REF_TMP1_I]]) #[[ATTR8]], !noalias [[META36]]
// CHECK-NEXT:    [[INC_I]] = add nuw nsw i64 [[I_0_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I]], !llvm.loop [[LOOP68:![0-9]+]]
// CHECK:       _ZN4sycl3_V1plERKNS0_3vecINS0_3ext6oneapi8bfloat16ELi2EEES7_.exit:
// CHECK-NEXT:    store <2 x i16> [[VECINS_I_I10_I]], ptr addrspace(4) [[AGG_RESULT]], align 4, !alias.scope [[META36]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(ext::oneapi::bfloat16, BFA, +)

/***************** Binary Logical Ops *******************/

// CHECK-LABEL: define dso_local spir_func void @_Z21CheckDevCodeBINOPINTLN4sycl3_V13vecIiLi2EEES2_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret([[VECINT2]]) align 8 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECINT2]]) align 8 [[INVECINTL2A:%.*]], ptr nocapture noundef readonly byval([[VECINT2]]) align 8 [[INVECINTL2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META69:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META70:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i32>, ptr [[INVECINTL2A]], align 8, !tbaa [[TBAA10]], !noalias [[META70]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i32>, ptr [[INVECINTL2B]], align 8, !tbaa [[TBAA10]], !noalias [[META70]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp sgt <2 x i32> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK-NEXT:    store <2 x i32> [[SEXT_I]], ptr addrspace(4) [[AGG_RESULT]], align 8, !tbaa [[TBAA10]], !alias.scope [[META70]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(int, INTL, >)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeBINOPBYTELN4sycl3_V13vecISt4byteLi2EEES3_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.6") align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECBYTE2]]) align 2 [[INVECBYTEL2A:%.*]], ptr nocapture noundef readonly byval([[VECBYTE2]]) align 2 [[INVECBYTEL2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META73:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META74:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBYTEL2A]], align 2, !tbaa [[TBAA10]], !noalias [[META74]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i8>, ptr [[INVECBYTEL2B]], align 2, !tbaa [[TBAA10]], !noalias [[META74]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp sgt <2 x i8> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i8>
// CHECK-NEXT:    store <2 x i8> [[SEXT_I]], ptr addrspace(4) [[AGG_RESULT]], align 2, !tbaa [[TBAA10]], !alias.scope [[META74]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(std::byte, BYTEL, >)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeBINOPBOOLLN4sycl3_V13vecIbLi2EEES2_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.6") align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECBOOL2]]) align 2 [[INVECBOOLL2A:%.*]], ptr nocapture noundef readonly byval([[VECBOOL2]]) align 2 [[INVECBOOLL2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META77:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META78:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBOOLL2A]], align 2, !tbaa [[TBAA10]], !noalias [[META78]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i8>, ptr [[INVECBOOLL2B]], align 2, !tbaa [[TBAA10]], !noalias [[META78]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp sgt <2 x i8> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i8>
// CHECK-NEXT:    store <2 x i8> [[SEXT_I]], ptr addrspace(4) [[AGG_RESULT]], align 2, !tbaa [[TBAA10]], !alias.scope [[META78]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(bool, BOOLL, >)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeBINOPHALFLN4sycl3_V13vecINS0_6detail9half_impl4halfELi2EEES5_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.7") align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECHALF2]]) align 4 [[INVECHALFL2A:%.*]], ptr nocapture noundef readonly byval([[VECHALF2]]) align 4 [[INVECHALFL2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META81:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META82:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x half>, ptr [[INVECHALFL2A]], align 4, !tbaa [[TBAA10]], !noalias [[META82]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x half>, ptr [[INVECHALFL2B]], align 4, !tbaa [[TBAA10]], !noalias [[META82]]
// CHECK-NEXT:    [[CMP_I:%.*]] = fcmp ogt <2 x half> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i16>
// CHECK-NEXT:    store <2 x i16> [[SEXT_I]], ptr addrspace(4) [[AGG_RESULT]], align 4, !tbaa [[TBAA10]], !alias.scope [[META82]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(sycl::half, HALFL, >)

// FIXME: Why do we interpret BF16 as INT16 to perform logical operations?
// For arithmetic ops, we convert BF16 to float and then perform the operation.

// CHECK-LABEL: define dso_local spir_func void @_Z20CheckDevCodeBINOPBFLN4sycl3_V13vecINS0_3ext6oneapi8bfloat16ELi2EEES5_(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.7") align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECBF2]]) align 4 [[INVECBFL2A:%.*]], ptr nocapture noundef readonly byval([[VECBF2]]) align 4 [[INVECBFL2B:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META85:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META86:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i16>, ptr [[INVECBFL2A]], align 4, !tbaa [[TBAA10]], !noalias [[META86]]
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i16>, ptr [[INVECBFL2B]], align 4, !tbaa [[TBAA10]], !noalias [[META86]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp ugt <2 x i16> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i16>
// CHECK-NEXT:    store <2 x i16> [[SEXT_I]], ptr addrspace(4) [[AGG_RESULT]], align 4, !tbaa [[TBAA10]], !alias.scope [[META86]]
// CHECK-NEXT:    ret void
//
CHECKBINOP(ext::oneapi::bfloat16, BFL, >)

/********************** Unary Ops **********************/

// CHECK-LABEL: define dso_local spir_func void @_Z21CheckDevCodeUOPINTNEGN4sycl3_V13vecIiLi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret([[VECINT2]]) align 8 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECINT2]]) align 8 [[INVECINTNEG2:%.*]]) local_unnamed_addr #[[ATTR1]] !srcloc [[META89:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[REF_TMP_I:%.*]] = alloca [[VECINT2]], align 8
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META90:![0-9]+]])
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 8, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META90]]
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i32>, ptr [[INVECINTNEG2]], align 8, !tbaa [[TBAA10]], !noalias [[META90]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp eq <2 x i32> [[TMP0]], zeroinitializer
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK-NEXT:    store <2 x i32> [[SEXT_I]], ptr [[REF_TMP_I]], align 8, !tbaa [[TBAA10]], !noalias [[META90]]
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META93:![0-9]+]])
// CHECK-NEXT:    br label [[FOR_COND_I_I_I:%.*]]
// CHECK:       for.cond.i.i.i:
// CHECK-NEXT:    [[I_0_I_I_I:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INC_I_I_I:%.*]], [[FOR_BODY_I_I_I:%.*]] ]
// CHECK-NEXT:    [[CMP_I_I_I:%.*]] = icmp ult i64 [[I_0_I_I_I]], 8
// CHECK-NEXT:    br i1 [[CMP_I_I_I]], label [[FOR_BODY_I_I_I]], label [[_ZN4SYCL3_V1NTERKNS0_3VECIILI2EEE_EXIT:%.*]]
// CHECK:       for.body.i.i.i:
// CHECK-NEXT:    [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[REF_TMP_I]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[ARRAYIDX_I_I_I]], align 1, !tbaa [[TBAA10]], !noalias [[META96:![0-9]+]]
// CHECK-NEXT:    [[ARRAYIDX1_I_I_I:%.*]] = getelementptr inbounds i8, ptr addrspace(4) [[AGG_RESULT]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    store i8 [[TMP1]], ptr addrspace(4) [[ARRAYIDX1_I_I_I]], align 1, !tbaa [[TBAA10]], !alias.scope [[META96]]
// CHECK-NEXT:    [[INC_I_I_I]] = add nuw nsw i64 [[I_0_I_I_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I_I_I]], !llvm.loop [[LOOP97:![0-9]+]]
// CHECK:       _ZN4sycl3_V1ntERKNS0_3vecIiLi2EEE.exit:
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 8, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META90]]
// CHECK-NEXT:    ret void
//
CHECKUOP(int, INTNEG, !, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z21CheckDevCodeUOPINTSUBN4sycl3_V13vecIiLi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret([[VECINT2]]) align 8 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECINT2]]) align 8 [[INVECINTSUB2:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META98:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META99:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i32>, ptr [[INVECINTSUB2]], align 8, !tbaa [[TBAA10]], !noalias [[META99]]
// CHECK-NEXT:    [[SUB_I:%.*]] = sub <2 x i32> zeroinitializer, [[TMP0]]
// CHECK-NEXT:    store <2 x i32> [[SUB_I]], ptr addrspace(4) [[AGG_RESULT]], align 8, !tbaa [[TBAA10]], !alias.scope [[META99]]
// CHECK-NEXT:    ret void
//
CHECKUOP(int, INTSUB, -, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeUOPBYTENEGN4sycl3_V13vecISt4byteLi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.6") align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECBYTE2]]) align 2 [[INVECBYTENEG2:%.*]]) local_unnamed_addr #[[ATTR1]] !srcloc [[META102:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[REF_TMP_I:%.*]] = alloca [[VECBYTE2]], align 2
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META103:![0-9]+]])
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 2, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META103]]
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBYTENEG2]], align 2, !tbaa [[TBAA10]], !noalias [[META103]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp eq <2 x i8> [[TMP0]], zeroinitializer
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i8>
// CHECK-NEXT:    store <2 x i8> [[SEXT_I]], ptr [[REF_TMP_I]], align 2, !tbaa [[TBAA10]], !noalias [[META103]]
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META106:![0-9]+]])
// CHECK-NEXT:    br label [[FOR_COND_I_I_I:%.*]]
// CHECK:       for.cond.i.i.i:
// CHECK-NEXT:    [[I_0_I_I_I:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INC_I_I_I:%.*]], [[FOR_BODY_I_I_I:%.*]] ]
// CHECK-NEXT:    [[CMP_I_I_I:%.*]] = icmp ult i64 [[I_0_I_I_I]], 2
// CHECK-NEXT:    br i1 [[CMP_I_I_I]], label [[FOR_BODY_I_I_I]], label [[_ZN4SYCL3_V1NTERKNS0_3VECIST4BYTELI2EEE_EXIT:%.*]]
// CHECK:       for.body.i.i.i:
// CHECK-NEXT:    [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[REF_TMP_I]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[ARRAYIDX_I_I_I]], align 1, !tbaa [[TBAA10]], !noalias [[META109:![0-9]+]]
// CHECK-NEXT:    [[ARRAYIDX1_I_I_I:%.*]] = getelementptr inbounds i8, ptr addrspace(4) [[AGG_RESULT]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    store i8 [[TMP1]], ptr addrspace(4) [[ARRAYIDX1_I_I_I]], align 1, !tbaa [[TBAA10]], !alias.scope [[META109]]
// CHECK-NEXT:    [[INC_I_I_I]] = add nuw nsw i64 [[I_0_I_I_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I_I_I]], !llvm.loop [[LOOP97]]
// CHECK:       _ZN4sycl3_V1ntERKNS0_3vecISt4byteLi2EEE.exit:
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 2, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META103]]
// CHECK-NEXT:    ret void
//
CHECKUOP(std::byte, BYTENEG, !, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeUOPBYTESUBN4sycl3_V13vecISt4byteLi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret([[VECBYTE2]]) align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECBYTE2]]) align 2 [[INVECBYTESUB2:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META110:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META111:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBYTESUB2]], align 2, !tbaa [[TBAA10]], !noalias [[META111]]
// CHECK-NEXT:    [[SUB_I:%.*]] = sub <2 x i8> zeroinitializer, [[TMP0]]
// CHECK-NEXT:    store <2 x i8> [[SUB_I]], ptr addrspace(4) [[AGG_RESULT]], align 2, !tbaa [[TBAA10]], !alias.scope [[META111]]
// CHECK-NEXT:    ret void
//
CHECKUOP(std::byte, BYTESUB, -, -1)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeUOPBOOLNEGN4sycl3_V13vecIbLi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.6") align 2 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECBOOL2]]) align 2 [[INVECBOOLNEG2:%.*]]) local_unnamed_addr #[[ATTR1]] !srcloc [[META114:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[REF_TMP_I:%.*]] = alloca [[VECBOOL2]], align 2
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META115:![0-9]+]])
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 2, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META115]]
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i8>, ptr [[INVECBOOLNEG2]], align 2, !tbaa [[TBAA10]], !noalias [[META115]]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp eq <2 x i8> [[TMP0]], zeroinitializer
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i8>
// CHECK-NEXT:    store <2 x i8> [[SEXT_I]], ptr [[REF_TMP_I]], align 2, !tbaa [[TBAA10]], !noalias [[META115]]
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META118:![0-9]+]])
// CHECK-NEXT:    br label [[FOR_COND_I_I_I:%.*]]
// CHECK:       for.cond.i.i.i:
// CHECK-NEXT:    [[I_0_I_I_I:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INC_I_I_I:%.*]], [[FOR_BODY_I_I_I:%.*]] ]
// CHECK-NEXT:    [[CMP_I_I_I:%.*]] = icmp ult i64 [[I_0_I_I_I]], 2
// CHECK-NEXT:    br i1 [[CMP_I_I_I]], label [[FOR_BODY_I_I_I]], label [[_ZN4SYCL3_V1NTERKNS0_3VECIBLI2EEE_EXIT:%.*]]
// CHECK:       for.body.i.i.i:
// CHECK-NEXT:    [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[REF_TMP_I]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[ARRAYIDX_I_I_I]], align 1, !tbaa [[TBAA10]], !noalias [[META121:![0-9]+]]
// CHECK-NEXT:    [[ARRAYIDX1_I_I_I:%.*]] = getelementptr inbounds i8, ptr addrspace(4) [[AGG_RESULT]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    store i8 [[TMP1]], ptr addrspace(4) [[ARRAYIDX1_I_I_I]], align 1, !tbaa [[TBAA10]], !alias.scope [[META121]]
// CHECK-NEXT:    [[INC_I_I_I]] = add nuw nsw i64 [[I_0_I_I_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I_I_I]], !llvm.loop [[LOOP97]]
// CHECK:       _ZN4sycl3_V1ntERKNS0_3vecIbLi2EEE.exit:
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 2, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META115]]
// CHECK-NEXT:    ret void
//
CHECKUOP(bool, BOOLNEG, !, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeUOPHALFNEGN4sycl3_V13vecINS0_6detail9half_impl4halfELi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.7") align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECHALF2]]) align 4 [[INVECHALFNEG2:%.*]]) local_unnamed_addr #[[ATTR1]] !srcloc [[META122:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[REF_TMP_I:%.*]] = alloca [[VECHALF2]], align 4
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META123:![0-9]+]])
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META123]]
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x half>, ptr [[INVECHALFNEG2]], align 4, !tbaa [[TBAA10]], !noalias [[META123]]
// CHECK-NEXT:    [[CMP_I:%.*]] = fcmp oeq <2 x half> [[TMP0]], zeroinitializer
// CHECK-NEXT:    [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i16>
// CHECK-NEXT:    store <2 x i16> [[SEXT_I]], ptr [[REF_TMP_I]], align 4, !tbaa [[TBAA10]], !noalias [[META123]]
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META126:![0-9]+]])
// CHECK-NEXT:    br label [[FOR_COND_I_I_I:%.*]]
// CHECK:       for.cond.i.i.i:
// CHECK-NEXT:    [[I_0_I_I_I:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INC_I_I_I:%.*]], [[FOR_BODY_I_I_I:%.*]] ]
// CHECK-NEXT:    [[CMP_I_I_I:%.*]] = icmp ult i64 [[I_0_I_I_I]], 4
// CHECK-NEXT:    br i1 [[CMP_I_I_I]], label [[FOR_BODY_I_I_I]], label [[_ZN4SYCL3_V1NTERKNS0_3VECINS0_6DETAIL9HALF_IMPL4HALFELI2EEE_EXIT:%.*]]
// CHECK:       for.body.i.i.i:
// CHECK-NEXT:    [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[REF_TMP_I]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[ARRAYIDX_I_I_I]], align 1, !tbaa [[TBAA10]], !noalias [[META129:![0-9]+]]
// CHECK-NEXT:    [[ARRAYIDX1_I_I_I:%.*]] = getelementptr inbounds i8, ptr addrspace(4) [[AGG_RESULT]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    store i8 [[TMP1]], ptr addrspace(4) [[ARRAYIDX1_I_I_I]], align 1, !tbaa [[TBAA10]], !alias.scope [[META129]]
// CHECK-NEXT:    [[INC_I_I_I]] = add nuw nsw i64 [[I_0_I_I_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I_I_I]], !llvm.loop [[LOOP97]]
// CHECK:       _ZN4sycl3_V1ntERKNS0_3vecINS0_6detail9half_impl4halfELi2EEE.exit:
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull [[REF_TMP_I]]) #[[ATTR8]], !noalias [[META123]]
// CHECK-NEXT:    ret void
//
CHECKUOP(sycl::half, HALFNEG, !, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z22CheckDevCodeUOPHALFSUBN4sycl3_V13vecINS0_6detail9half_impl4halfELi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret([[VECHALF2]]) align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECHALF2]]) align 4 [[INVECHALFSUB2:%.*]]) local_unnamed_addr #[[ATTR0]] !srcloc [[META130:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META131:![0-9]+]])
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x half>, ptr [[INVECHALFSUB2]], align 4, !tbaa [[TBAA10]], !noalias [[META131]]
// CHECK-NEXT:    [[FNEG_I:%.*]] = fneg <2 x half> [[TMP0]]
// CHECK-NEXT:    store <2 x half> [[FNEG_I]], ptr addrspace(4) [[AGG_RESULT]], align 4, !tbaa [[TBAA10]], !alias.scope [[META131]]
// CHECK-NEXT:    ret void
//
CHECKUOP(sycl::half, HALFSUB, -, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z20CheckDevCodeUOPBFNEGN4sycl3_V13vecINS0_3ext6oneapi8bfloat16ELi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret(%"class.sycl::_V1::vec.7") align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECBF2]]) align 4 [[INVECBFNEG2:%.*]]) local_unnamed_addr #[[ATTR3]] !srcloc [[META134:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RET_I:%.*]] = alloca [[VECBF2]], align 4
// CHECK-NEXT:    [[REF_TMP1_I:%.*]] = alloca float, align 4
// CHECK-NEXT:    [[REF_TMP2_I:%.*]] = alloca %"class.sycl::_V1::ext::oneapi::bfloat16", align 2
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META135:![0-9]+]])
// CHECK-NEXT:    [[REF_TMP1_ASCAST_I:%.*]] = addrspacecast ptr [[REF_TMP1_I]] to ptr addrspace(4)
// CHECK-NEXT:    [[REF_TMP2_ASCAST_I:%.*]] = addrspacecast ptr [[REF_TMP2_I]] to ptr addrspace(4)
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull [[RET_I]]) #[[ATTR8]], !noalias [[META135]]
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i16>, ptr [[INVECBFNEG2]], align 4, !tbaa [[TBAA10]], !noalias [[META138:![0-9]+]]
// CHECK-NEXT:    br label [[FOR_COND_I:%.*]]
// CHECK:       for.cond.i:
// CHECK-NEXT:    [[TMP1:%.*]] = phi <2 x i16> [ zeroinitializer, [[ENTRY:%.*]] ], [ [[VECINS_I_I_I:%.*]], [[FOR_BODY_I:%.*]] ]
// CHECK-NEXT:    [[I_0_I:%.*]] = phi i64 [ 0, [[ENTRY]] ], [ [[INC_I:%.*]], [[FOR_BODY_I]] ]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp ult i64 [[I_0_I]], 2
// CHECK-NEXT:    br i1 [[CMP_I]], label [[FOR_BODY_I]], label [[FOR_COND_CLEANUP_I:%.*]]
// CHECK:       for.cond.cleanup.i:
// CHECK-NEXT:    store <2 x i16> [[TMP1]], ptr [[RET_I]], align 1, !noalias [[META135]]
// CHECK-NEXT:    call void @llvm.experimental.noalias.scope.decl(metadata [[META143:![0-9]+]])
// CHECK-NEXT:    br label [[FOR_COND_I_I_I:%.*]]
// CHECK:       for.cond.i.i.i:
// CHECK-NEXT:    [[I_0_I_I_I:%.*]] = phi i64 [ 0, [[FOR_COND_CLEANUP_I]] ], [ [[INC_I_I_I:%.*]], [[FOR_BODY_I_I_I:%.*]] ]
// CHECK-NEXT:    [[CMP_I_I_I:%.*]] = icmp ult i64 [[I_0_I_I_I]], 4
// CHECK-NEXT:    br i1 [[CMP_I_I_I]], label [[FOR_BODY_I_I_I]], label [[_ZN4SYCL3_V1NTERKNS0_3VECINS0_3EXT6ONEAPI8BFLOAT16ELI2EEE_EXIT:%.*]]
// CHECK:       for.body.i.i.i:
// CHECK-NEXT:    [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[RET_I]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    [[TMP2:%.*]] = load i8, ptr [[ARRAYIDX_I_I_I]], align 1, !tbaa [[TBAA10]], !noalias [[META146:![0-9]+]]
// CHECK-NEXT:    [[ARRAYIDX1_I_I_I:%.*]] = getelementptr inbounds i8, ptr addrspace(4) [[AGG_RESULT]], i64 [[I_0_I_I_I]]
// CHECK-NEXT:    store i8 [[TMP2]], ptr addrspace(4) [[ARRAYIDX1_I_I_I]], align 1, !tbaa [[TBAA10]], !alias.scope [[META146]]
// CHECK-NEXT:    [[INC_I_I_I]] = add nuw nsw i64 [[I_0_I_I_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I_I_I]], !llvm.loop [[LOOP97]]
// CHECK:       for.body.i:
// CHECK-NEXT:    [[CONV_I:%.*]] = trunc nuw nsw i64 [[I_0_I]] to i32
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull [[REF_TMP1_I]]) #[[ATTR8]], !noalias [[META135]]
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 2, ptr nonnull [[REF_TMP2_I]]) #[[ATTR8]], !noalias [[META135]]
// CHECK-NEXT:    [[VECEXT_I_I_I:%.*]] = extractelement <2 x i16> [[TMP0]], i32 [[CONV_I]]
// CHECK-NEXT:    store i16 [[VECEXT_I_I_I]], ptr [[REF_TMP2_I]], align 2, !tbaa [[TBAA147:![0-9]+]], !alias.scope [[META149:![0-9]+]], !noalias [[META135]]
// CHECK-NEXT:    [[CALL_I_I_I:%.*]] = call spir_func noundef float @__devicelib_ConvertBF16ToFINTEL(ptr addrspace(4) noundef align 2 dereferenceable(2) [[REF_TMP2_ASCAST_I]]) #[[ATTR9]], !noalias [[META135]]
// CHECK-NEXT:    [[CMP_I_I:%.*]] = fcmp oeq float [[CALL_I_I_I]], 0.000000e+00
// CHECK-NEXT:    [[CONV4_I:%.*]] = uitofp i1 [[CMP_I_I]] to float
// CHECK-NEXT:    store float [[CONV4_I]], ptr [[REF_TMP1_I]], align 4, !tbaa [[TBAA66]], !noalias [[META135]]
// CHECK-NEXT:    [[CALL_I_I10_I:%.*]] = call spir_func noundef zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4) [[REF_TMP1_ASCAST_I]]) #[[ATTR9]], !noalias [[META135]]
// CHECK-NEXT:    [[VECINS_I_I_I]] = insertelement <2 x i16> [[TMP1]], i16 [[CALL_I_I10_I]], i32 [[CONV_I]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 2, ptr nonnull [[REF_TMP2_I]]) #[[ATTR8]], !noalias [[META135]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull [[REF_TMP1_I]]) #[[ATTR8]], !noalias [[META135]]
// CHECK-NEXT:    [[INC_I]] = add nuw nsw i64 [[I_0_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I]], !llvm.loop [[LOOP152:![0-9]+]]
// CHECK:       _ZN4sycl3_V1ntERKNS0_3vecINS0_3ext6oneapi8bfloat16ELi2EEE.exit:
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull [[RET_I]]) #[[ATTR8]], !noalias [[META135]]
// CHECK-NEXT:    ret void
//
CHECKUOP(ext::oneapi::bfloat16, BFNEG, !, 1)

// CHECK-LABEL: define dso_local spir_func void @_Z20CheckDevCodeUOPBFSUBN4sycl3_V13vecINS0_3ext6oneapi8bfloat16ELi2EEE(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias nocapture writable writeonly sret([[VECBF2]]) align 4 [[AGG_RESULT:%.*]], ptr nocapture noundef readonly byval([[VECBF2]]) align 4 [[INVECBFSUB2:%.*]]) local_unnamed_addr #[[ATTR5:[0-9]+]] !srcloc [[META153:![0-9]+]] !sycl_fixed_targets [[META6]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[REF_TMP_I_I:%.*]] = alloca float, align 4
// CHECK-NEXT:    [[V_I:%.*]] = alloca %"class.sycl::_V1::ext::oneapi::bfloat16", align 2
// CHECK-NEXT:    tail call void @llvm.experimental.noalias.scope.decl(metadata [[META154:![0-9]+]])
// CHECK-NEXT:    [[V_ASCAST_I:%.*]] = addrspacecast ptr [[V_I]] to ptr addrspace(4)
// CHECK-NEXT:    store i32 0, ptr addrspace(4) [[AGG_RESULT]], align 4, !alias.scope [[META154]]
// CHECK-NEXT:    [[REF_TMP_ASCAST_I_I:%.*]] = addrspacecast ptr [[REF_TMP_I_I]] to ptr addrspace(4)
// CHECK-NEXT:    br label [[FOR_COND_I:%.*]]
// CHECK:       for.cond.i:
// CHECK-NEXT:    [[I_0_I:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INC_I:%.*]], [[FOR_COND_I]] ]
// CHECK-NEXT:    [[CMP_I:%.*]] = icmp ult i64 [[I_0_I]], 2
// CHECK-NEXT:    call void @llvm.assume(i1 [[CMP_I]])
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 2, ptr nonnull [[V_I]]) #[[ATTR8]], !noalias [[META154]]
// CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds <2 x i16>, ptr [[INVECBFSUB2]], i64 0, i64 [[I_0_I]]
// CHECK-NEXT:    [[VECEXT_I:%.*]] = load i16, ptr [[TMP0]], align 2, !noalias [[META154]]
// CHECK-NEXT:    store i16 [[VECEXT_I]], ptr [[V_I]], align 2, !tbaa [[TBAA157:![0-9]+]], !alias.scope [[META159:![0-9]+]], !noalias [[META154]]
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull [[REF_TMP_I_I]]) #[[ATTR8]], !noalias [[META162:![0-9]+]]
// CHECK-NEXT:    [[CALL_I_I:%.*]] = call spir_func float @__devicelib_ConvertBF16ToFINTEL(ptr addrspace(4) noundef align 2 dereferenceable(2) [[V_ASCAST_I]]) #[[ATTR9]], !noalias [[META162]]
// CHECK-NEXT:    [[FNEG_I_I:%.*]] = fneg float [[CALL_I_I]]
// CHECK-NEXT:    store float [[FNEG_I_I]], ptr [[REF_TMP_I_I]], align 4, !tbaa [[TBAA66]], !noalias [[META162]]
// CHECK-NEXT:    [[CALL_I_I_I_I:%.*]] = call spir_func noundef zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4) [[REF_TMP_ASCAST_I_I]]) #[[ATTR9]], !noalias [[META162]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull [[REF_TMP_I_I]]) #[[ATTR8]], !noalias [[META162]]
// CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds <2 x i16>, ptr addrspace(4) [[AGG_RESULT]], i64 0, i64 [[I_0_I]]
// CHECK-NEXT:    store i16 [[CALL_I_I_I_I]], ptr addrspace(4) [[TMP1]], align 2, !alias.scope [[META154]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 2, ptr nonnull [[V_I]]) #[[ATTR8]], !noalias [[META154]]
// CHECK-NEXT:    [[INC_I]] = add nuw nsw i64 [[I_0_I]], 1
// CHECK-NEXT:    br label [[FOR_COND_I]], !llvm.loop [[LOOP165:![0-9]+]]
//
CHECKUOP(ext::oneapi::bfloat16, BFSUB, -, 1)
