; RUN: opt < %s -debugify -LowerWGScope -S | FileCheck %s

; Check that debug info is not lost after the LowerWGScope pass.
; Typical case of pfwg + pfwi usage.

%struct.bar = type { i8 }
%struct.zot = type { %struct.widget, %struct.widget, %struct.widget, %struct.foo }
%struct.widget = type { %struct.barney }
%struct.barney = type { [3 x i64] }
%struct.foo = type { %struct.barney }
%struct.foo.0 = type { i8 }

define internal spir_func void @wibble(%struct.bar addrspace(4)* %arg, %struct.zot* byval(%struct.zot) align 8 %arg1) align 2 !work_group_scope !0 {
; CHECK-LABEL: define {{[^@]+}}@wibble
; CHECK-SAME: (%struct.bar addrspace(4)* [[ARG:%.*]], %struct.zot* byval(%struct.zot) align 8 [[ARG1:%.*]])
bb:
  %tmp = alloca %struct.bar addrspace(4)*, align 8
; CHECK:    [[TMP:%.*]] = alloca %struct.bar addrspace(4)*, align 8
; CHECK:    call void @llvm.dbg.value(metadata %struct.bar addrspace(4)** [[TMP]], [[META9:metadata !.*]], metadata !DIExpression())
  %tmp1 = alloca %struct.foo.0, align 1
; CHECK:    [[TMP1:%.*]] = alloca %struct.foo.0, align 1
; CHECK:    call void @llvm.dbg.value(metadata %struct.foo.0* [[TMP1]], [[META11:metadata !.*]], metadata !DIExpression())
  store %struct.bar addrspace(4)* %arg, %struct.bar addrspace(4)** %tmp, align 8
  %tmp3 = load %struct.bar addrspace(4)*, %struct.bar addrspace(4)** %tmp, align 8
; CHECK:    [[TMP3:%.*]] = load %struct.bar addrspace(4)*, %struct.bar addrspace(4)** [[TMP]], align 8
; CHECK:    call void @llvm.dbg.value(metadata %struct.bar addrspace(4)* [[TMP3]], [[META12:metadata !.*]], metadata !DIExpression())
  %tmp4 = addrspacecast %struct.zot* %arg1 to %struct.zot addrspace(4)*
; CHECK:    [[TMP4:%.*]] = addrspacecast %struct.zot* [[ARG1]] to %struct.zot addrspace(4)*
; CHECK:    call void @llvm.dbg.value(metadata %struct.zot addrspace(4)* [[TMP4]], [[META13:metadata !.*]], metadata !DIExpression())
  call spir_func void @bar(%struct.zot addrspace(4)* %tmp4, %struct.foo.0* byval(%struct.foo.0) align 1 %tmp1)
  ret void
}

define internal spir_func void @bar(%struct.zot addrspace(4)* %arg, %struct.foo.0* byval(%struct.foo.0) align 1 %arg1) align 2 !work_item_scope !0 !parallel_for_work_item !0 {
; CHECK-LABEL: define {{[^@]+}}@bar
; CHECK:    call void @llvm.dbg.value(metadata i32 0, [[META23:metadata !.*]], metadata !DIExpression())
bb:
  ret void
}

!0 = !{}
