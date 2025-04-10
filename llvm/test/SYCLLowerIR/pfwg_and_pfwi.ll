; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=LowerWGScope -S | FileCheck %s

; Check that allocas which correspond to PFWI lambda object and a local copy of the PFWG lambda object
; are properly handled by LowerWGScope pass. Check that WG-shared local "shadow" variables are created
; and before each PFWI invocation leader WI stores its private copy of the variable into the shadow,
; then all WIs load the shadow value into their private copies ("materialize" the private copy).

%struct.bar = type { i8 }
%struct.zot = type { %struct.widget, %struct.widget, %struct.widget, %struct.foo }
%struct.widget = type { %struct.barney }
%struct.barney = type { [3 x i64] }
%struct.foo = type { %struct.barney }
%struct.foo.0 = type { i8 }


define internal spir_func void @wibble(ptr addrspace(4) %arg, ptr byval(%struct.zot) align 8 %arg1) align 2 !work_group_scope !0 {
; CHECK-LABEL: @wibble(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    [[TMP0:%.*]] = alloca ptr addrspace(4), align 8
; CHECK-NEXT:    [[TMP1:%.*]] = alloca [[STRUCT_FOO_0:%.*]], align 1
; CHECK-NEXT:    [[TMP2:%.*]] = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationIndex, align 4
; CHECK-NEXT:    call void @_Z22__spirv_ControlBarrieriii(i32 2, i32 2, i32 272) #[[ATTR0:[0-9]+]]
; CHECK-NEXT:    [[CMPZ3:%.*]] = icmp eq i64 [[TMP2]], 0
; CHECK-NEXT:    br i1 [[CMPZ3]], label [[LEADER:%.*]], label [[MERGE:%.*]]
; CHECK:       leader:
; CHECK-NEXT:    call void @llvm.memcpy.p3.p0.i64(ptr addrspace(3) align 16 @ArgShadow, ptr align 8 [[ARG1:%.*]], i64 96, i1 false)
; CHECK-NEXT:    br label [[MERGE]]
; CHECK:       merge:
; CHECK-NEXT:    call void @_Z22__spirv_ControlBarrieriii(i32 2, i32 2, i32 272) #[[ATTR0]]
; CHECK-NEXT:    call void @llvm.memcpy.p0.p3.i64(ptr align 8 [[ARG1]], ptr addrspace(3) align 16 @ArgShadow, i64 96, i1 false)
; CHECK-NEXT:    [[TMP5:%.*]] = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationIndex, align 4
; CHECK-NEXT:    call void @_Z22__spirv_ControlBarrieriii(i32 2, i32 2, i32 272) #[[ATTR0]]
; CHECK-NEXT:    [[CMPZ:%.*]] = icmp eq i64 [[TMP5]], 0
; CHECK-NEXT:    br i1 [[CMPZ]], label [[WG_LEADER:%.*]], label [[WG_CF:%.*]]
; CHECK:       wg_leader:
; CHECK-NEXT:    store ptr addrspace(4) [[ARG:%.*]], ptr [[TMP0]], align 8
; CHECK-NEXT:    br label [[WG_CF]]
; CHECK:       wg_cf:
; CHECK-NEXT:    [[TMP6:%.*]] = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationIndex, align 4
; CHECK-NEXT:    call void @_Z22__spirv_ControlBarrieriii(i32 2, i32 2, i32 272) #[[ATTR0]]
; CHECK-NEXT:    [[CMPZ2:%.*]] = icmp eq i64 [[TMP6]], 0
; CHECK-NEXT:    br i1 [[CMPZ2]], label [[TESTMAT:%.*]], label [[LEADERMAT:%.*]]
; CHECK:       TestMat:
; CHECK-NEXT:    call void @llvm.memcpy.p3.p0.i64(ptr addrspace(3) align 8 @WGCopy.1, ptr align 1 [[TMP1]], i64 1, i1 false)
; CHECK-NEXT:    [[MAT_LD:%.*]] = load ptr addrspace(4), ptr [[TMP0]], align 8
; CHECK-NEXT:    store ptr addrspace(4) [[MAT_LD]], ptr addrspace(3) @WGCopy, align 8
; CHECK-NEXT:    br label [[LEADERMAT]]
; CHECK:       LeaderMat:
; CHECK-NEXT:    call void @_Z22__spirv_ControlBarrieriii(i32 2, i32 2, i32 272) #[[ATTR0]]
; CHECK-NEXT:    [[MAT_LD1:%.*]] = load ptr addrspace(4), ptr addrspace(3) @WGCopy, align 8
; CHECK-NEXT:    store ptr addrspace(4) [[MAT_LD1]], ptr [[TMP0]], align 8
; CHECK-NEXT:    call void @llvm.memcpy.p0.p3.i64(ptr align 1 [[TMP1]], ptr addrspace(3) align 8 @WGCopy.1, i64 1, i1 false)
; CHECK-NEXT:    call void @_Z22__spirv_ControlBarrieriii(i32 2, i32 2, i32 272) #[[ATTR0]]
; CHECK-NEXT:    [[TMP9:%.*]] = addrspacecast ptr [[ARG1]] to ptr addrspace(4)
; CHECK-NEXT:    call spir_func void @bar(ptr addrspace(4) [[TMP9]], ptr byval([[STRUCT_FOO_0]]) align 1 [[TMP1]])
; CHECK-NEXT:    ret void
;
bb:
  %0 = alloca ptr addrspace(4), align 8
  %1 = alloca %struct.foo.0, align 1
  store ptr addrspace(4) %arg, ptr %0, align 8
  %2 = addrspacecast ptr %arg1 to ptr addrspace(4)
  call spir_func void @bar(ptr addrspace(4) %2, ptr byval(%struct.foo.0) align 1 %1)
  ret void
}

define internal spir_func void @bar(ptr addrspace(4) %arg, ptr byval(%struct.foo.0) align 1 %arg1) align 2 !work_item_scope !0 !parallel_for_work_item !0 {
bb:
  ret void
}

!0 = !{}
