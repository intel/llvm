; RUN: opt < %s -LowerWGScope -S -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes=LowerWGScope -S | FileCheck %s

; Check that argument of the function marked with !work_group_scope
; attribute passed as byval is shared by leader work item via local
; memory to all work items

%struct.baz = type { i64 }

; CHECK: @[[SHADOW:[a-zA-Z0-9]+]] = internal unnamed_addr addrspace(3) global %struct.baz undef

define internal spir_func void @wibble(%struct.baz* byval(%struct.baz) %arg1) !work_group_scope !0 {
; CHECK-LABEL: @wibble(
; CHECK-NEXT:    [[TMP1:%.*]] = load i64, i64 addrspace(1)* @__spirv_BuiltInLocalInvocationIndex
; CHECK-NEXT:    call void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
; CHECK-NEXT:    [[CMPZ:%.*]] = icmp eq i64 [[TMP1]], 0
; CHECK-NEXT:    br i1 [[CMPZ]], label [[LEADER:%.*]], label [[MERGE:%.*]]
; CHECK:       leader:
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast %struct.baz* [[ARG1:%.*]] to i8*
; CHECK-NEXT:    call void @llvm.memcpy.p3i8.p0i8.i64(i8 addrspace(3)* align 8 bitcast (%struct.baz addrspace(3)* @[[SHADOW]] to i8 addrspace(3)*), i8* [[TMP2]], i64 8, i1 false)
; CHECK-NEXT:    br label [[MERGE]]
; CHECK:       merge:
; CHECK-NEXT:    call void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast %struct.baz* [[ARG1]] to i8*
; CHECK-NEXT:    call void @llvm.memcpy.p0i8.p3i8.i64(i8* [[TMP3]], i8 addrspace(3)* align 8 bitcast (%struct.baz addrspace(3)* @[[SHADOW]] to i8 addrspace(3)*), i64 8, i1 false)
; CHECK-NEXT:    ret void
;
  ret void
}

!0 = !{}
