; RUN: opt -opaque-pointers < %s -passes=LowerESIMD -S | FileCheck %s

; This test checks we lower vector SPIRV globals with opaque pointers
; and a GEP operand

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

define spir_kernel void @"__spirv_GlobalInvocationId_xyz"() {
; CHECK-LABEL: @__spirv_GlobalInvocationId_xyz(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[DOTESIMD6:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK-NEXT:    [[LOCAL_ID_Y:%.*]] = extractelement <3 x i32> [[DOTESIMD6]], i32 1
; CHECK-NEXT:    [[LOCAL_ID_Y_CAST_TY:%.*]] = zext i32 [[LOCAL_ID_Y]] to i64
; CHECK-NEXT:    [[DOTESIMD7:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK-NEXT:    [[WGSIZE_Y:%.*]] = extractelement <3 x i32> [[DOTESIMD7]], i32 1
; CHECK-NEXT:    [[WGSIZE_Y_CAST_TY:%.*]] = zext i32 [[WGSIZE_Y]] to i64
; CHECK-NEXT:    [[GROUP_ID_Y:%.*]] = call i32 @llvm.genx.group.id.y()
; CHECK-NEXT:    [[GROUP_ID_Y_CAST_TY:%.*]] = zext i32 [[GROUP_ID_Y]] to i64
; CHECK-NEXT:    [[MUL8:%.*]] = mul i64 [[WGSIZE_Y_CAST_TY]], [[GROUP_ID_Y_CAST_TY]]
; CHECK-NEXT:    [[ADD9:%.*]] = add i64 [[LOCAL_ID_Y_CAST_TY]], [[MUL8]]
; CHECK-NEXT:    [[MUL10:%.*]] = shl nuw nsw i64 [[ADD9]], 5

; Verify that the attribute is deleted from GenX declaration
; CHECK-NOT: readnone
entry:
  %0 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, i64 8), align 8
  %mul.i = shl nuw nsw i64 %0, 5
 ret void
}
