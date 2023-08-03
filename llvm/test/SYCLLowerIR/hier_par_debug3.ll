; RUN: opt < %s -passes=debugify,LowerWGScope -S | FileCheck %s
;
; Check that debug info is not lost after the LowerWGScope pass.
; Case with additional control flow in pfwg.


%struct.snork = type { i32 }
%struct.eggs = type { i8 }
%struct.snork.0 = type { %struct.widget, %struct.widget, %struct.widget, %struct.ham }
%struct.widget = type { %struct.wibble }
%struct.wibble = type { [3 x i64] }
%struct.ham = type { %struct.wibble }

@global = internal addrspace(3) global [12 x %struct.snork] zeroinitializer, align 4

define internal spir_func void @spam(ptr addrspace(4) %arg, ptr byval(%struct.snork.0) align 8 %arg1) align 2 !work_group_scope !0 {
; CHECK-LABEL: define {{[^@]+}}@spam
; CHECK-SAME: (ptr addrspace(4) [[ARG:%.*]], ptr byval(%struct.snork.0) align 8 [[ARG1:%.*]])
entry:
  %tmp = alloca ptr addrspace(4), align 8
; CHECK:    [[TMP:%.*]] = alloca ptr addrspace(4), align 8
; CHECK:    call void @llvm.dbg.value(metadata ptr [[TMP]], [[META9:metadata !.*]], metadata !DIExpression())
  store ptr addrspace(4) %arg, ptr %tmp, align 8
  %tmp2 = load ptr addrspace(4), ptr %tmp, align 8
; CHECK:    [[TMP2:%.*]] = load ptr addrspace(4), ptr [[TMP]], align 8
; CHECK:    call void @llvm.dbg.value(metadata ptr addrspace(4) [[TMP2]], [[META11:metadata !.*]], metadata !DIExpression())
  br label %arrayctor.loop
arrayctor.loop:                                              ; preds = %arrayctor.loop, %entry
  %arrayctor.cur = phi ptr addrspace(4) [ addrspacecast (ptr addrspace(3) @global to ptr addrspace(4)), %entry ], [ %arrayctor.next, %arrayctor.loop ]
; CHECK:    [[ARRAYCTOR_CUR:%.*]] = phi ptr addrspace(4) [ addrspacecast (ptr addrspace(3) @global to ptr addrspace(4)), [[ENTRY:%.*]] ], [ [[ARRAYCTOR_NEXT:%.*]], [[ARRAYCTOR_LOOP:%.*]] ]
; CHECK:    call void @llvm.dbg.value(metadata ptr addrspace(4) [[ARRAYCTOR_CUR]], [[META12:metadata !.*]], metadata !DIExpression())
  call spir_func void @bar(ptr addrspace(4) %arrayctor.cur)
  %arrayctor.next = getelementptr inbounds %struct.snork, ptr addrspace(4) %arrayctor.cur, i64 1
; CHECK:    [[GEP_VAL:%.*]] = getelementptr inbounds %struct.snork, ptr addrspace(4) [[ARRAYCTOR_CUR]], i64 1
; CHECK:    call void @llvm.dbg.value(metadata ptr addrspace(4) [[GEP_VAL]], [[META13:metadata !.*]], metadata !DIExpression())
  %arrayctor.done = icmp eq ptr addrspace(4) %arrayctor.next, getelementptr inbounds (%struct.snork, ptr addrspace(4) addrspacecast (ptr addrspace(3) @global to ptr addrspace(4)), i64 12)
; CHECK:    [[ARRAYCTOR_DONE:%.*]] = icmp eq ptr addrspace(4) [[WG_VAL_ARRAYCTOR_NEXT:%.*]], getelementptr inbounds (%struct.snork, ptr addrspace(4) addrspacecast (ptr addrspace(3) @global to ptr addrspace(4)), i64 12)
; CHECK:    call void @llvm.dbg.value(metadata i1 [[ARRAYCTOR_DONE]], [[META14:metadata !.*]], metadata !DIExpression())
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void
}

define linkonce_odr dso_local spir_func void @bar(ptr addrspace(4) %arg) unnamed_addr align 2 {
bb:
  %tmp = alloca ptr addrspace(4), align 8
  store ptr addrspace(4) %arg, ptr %tmp, align 8
  %tmp1 = load ptr addrspace(4), ptr %tmp, align 8
  store i32 0, ptr addrspace(4) %tmp1, align 4
  ret void
}

!0 = !{}
