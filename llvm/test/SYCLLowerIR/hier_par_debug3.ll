; RUN: opt < %s -debugify -LowerWGScope -S | FileCheck %s
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

define internal spir_func void @spam(%struct.eggs addrspace(4)* %arg, %struct.snork.0* byval(%struct.snork.0) align 8 %arg1) align 2 !work_group_scope !0 {
; CHECK-LABEL: define {{[^@]+}}@spam
; CHECK-SAME: (%struct.eggs addrspace(4)* [[ARG:%.*]], %struct.snork.0* byval(%struct.snork.0) align 8 [[ARG1:%.*]])
entry:
  %tmp = alloca %struct.eggs addrspace(4)*, align 8
; CHECK:    [[TMP:%.*]] = alloca %struct.eggs addrspace(4)*, align 8
; CHECK:    call void @llvm.dbg.value(metadata %struct.eggs addrspace(4)** [[TMP]], [[META9:metadata !.*]], metadata !DIExpression())
  store %struct.eggs addrspace(4)* %arg, %struct.eggs addrspace(4)** %tmp, align 8
  %tmp2 = load %struct.eggs addrspace(4)*, %struct.eggs addrspace(4)** %tmp, align 8
; CHECK:    [[TMP2:%.*]] = load %struct.eggs addrspace(4)*, %struct.eggs addrspace(4)** [[TMP]], align 8
; CHECK:    call void @llvm.dbg.value(metadata %struct.eggs addrspace(4)* [[TMP2]], [[META11:metadata !.*]], metadata !DIExpression())
  br label %arrayctor.loop
arrayctor.loop:                                              ; preds = %arrayctor.loop, %entry
  %arrayctor.cur = phi %struct.snork addrspace(4)* [ getelementptr inbounds ([12 x %struct.snork], [12 x %struct.snork] addrspace(4)* addrspacecast ([12 x %struct.snork] addrspace(3)* @global to [12 x %struct.snork] addrspace(4)*), i32 0, i32 0), %entry ], [ %arrayctor.next, %arrayctor.loop ]
; CHECK:    [[ARRAYCTOR_CUR:%.*]] = phi [[STRUCT_SNORK:%.*]] addrspace(4)* [ getelementptr inbounds ([12 x %struct.snork], [12 x %struct.snork] addrspace(4)* addrspacecast ([12 x %struct.snork] addrspace(3)* @global to [12 x %struct.snork] addrspace(4)*), i32 0, i32 0), [[ENTRY:%.*]] ], [ [[ARRAYCTOR_NEXT:%.*]], [[ARRAYCTOR_LOOP:%.*]] ]
; CHECK:    call void @llvm.dbg.value(metadata %struct.snork addrspace(4)* [[ARRAYCTOR_CUR]], [[META12:metadata !.*]], metadata !DIExpression())
  call spir_func void @bar(%struct.snork addrspace(4)* %arrayctor.cur)
  %arrayctor.next = getelementptr inbounds %struct.snork, %struct.snork addrspace(4)* %arrayctor.cur, i64 1
; CHECK:    [[GEP_VAL:%.*]] = getelementptr inbounds %struct.snork, %struct.snork addrspace(4)* [[ARRAYCTOR_CUR]], i64 1
; CHECK:    call void @llvm.dbg.value(metadata %struct.snork addrspace(4)* [[GEP_VAL]], [[META13:metadata !.*]], metadata !DIExpression())
  %arrayctor.done = icmp eq %struct.snork addrspace(4)* %arrayctor.next, getelementptr inbounds (%struct.snork, %struct.snork addrspace(4)* getelementptr inbounds ([12 x %struct.snork], [12 x %struct.snork] addrspace(4)* addrspacecast ([12 x %struct.snork] addrspace(3)* @global to [12 x %struct.snork] addrspace(4)*), i32 0, i32 0), i64 12)
; CHECK:    [[ARRAYCTOR_DONE:%.*]] = icmp eq %struct.snork addrspace(4)* [[WG_VAL_ARRAYCTOR_NEXT:%.*]], getelementptr inbounds ([12 x %struct.snork], [12 x %struct.snork] addrspace(4)* addrspacecast ([12 x %struct.snork] addrspace(3)* @global to [12 x %struct.snork] addrspace(4)*), i64 1, i64 0)
; CHECK:    call void @llvm.dbg.value(metadata i1 [[ARRAYCTOR_DONE]], [[META14:metadata !.*]], metadata !DIExpression())
  br i1 %arrayctor.done, label %arrayctor.cont, label %arrayctor.loop

arrayctor.cont:                                   ; preds = %arrayctor.loop
  ret void
}

define linkonce_odr dso_local spir_func void @bar(%struct.snork addrspace(4)* %arg) unnamed_addr align 2 {
bb:
  %tmp = alloca %struct.snork addrspace(4)*, align 8
  store %struct.snork addrspace(4)* %arg, %struct.snork addrspace(4)** %tmp, align 8
  %tmp1 = load %struct.snork addrspace(4)*, %struct.snork addrspace(4)** %tmp, align 8
  %tmp2 = getelementptr inbounds %struct.snork, %struct.snork addrspace(4)* %tmp1, i32 0, i32 0
  store i32 0, i32 addrspace(4)* %tmp2, align 4
  ret void
}

!0 = !{}
