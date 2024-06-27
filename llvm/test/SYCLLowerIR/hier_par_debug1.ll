; RUN: opt < %s -passes=debugify,LowerWGScope -S | FileCheck %s

; Check that debug info is not lost after the LowerWGScope pass.
; Typical case of pfwg + pfwi usage.

%struct.bar = type { i8 }
%struct.zot = type { %struct.widget, %struct.widget, %struct.widget, %struct.foo }
%struct.widget = type { %struct.barney }
%struct.barney = type { [3 x i64] }
%struct.foo = type { %struct.barney }
%struct.foo.0 = type { i8 }

define internal spir_func void @wibble(ptr addrspace(4) %arg, ptr byval(%struct.zot) align 8 %arg1) align 2 !work_group_scope !0 {
; CHECK-LABEL: define {{[^@]+}}@wibble
; CHECK-SAME: (ptr addrspace(4) [[ARG:%.*]], ptr byval(%struct.zot) align 8 [[ARG1:%.*]])
bb:
  %tmp = alloca ptr addrspace(4), align 8
; CHECK:    [[TMP:%.*]] = alloca ptr addrspace(4), align 8, !dbg [[DBG1:!.*]]
; CHECK:    #dbg_value(ptr [[TMP]], [[META9:!.*]], !DIExpression(), [[DBG1]])
  %tmp1 = alloca %struct.foo.0, align 1
; CHECK:    [[TMP1:%.*]] = alloca %struct.foo.0, align 1, !dbg [[DBG2:!.*]]
; CHECK:    #dbg_value(ptr [[TMP1]], [[META11:!.*]], !DIExpression(), [[DBG2]])
  store ptr addrspace(4) %arg, ptr %tmp, align 8
  %tmp3 = load ptr addrspace(4), ptr %tmp, align 8
; CHECK:    [[TMP3:%.*]] = load ptr addrspace(4), ptr [[TMP]], align 8, !dbg [[DBG3:!.*]]
; CHECK:    #dbg_value(ptr addrspace(4) [[TMP3]], [[META12:!.*]], !DIExpression(), [[DBG3]])
  %tmp4 = addrspacecast ptr %arg1 to ptr addrspace(4)
; CHECK:    [[TMP4:%.*]] = addrspacecast ptr [[ARG1]] to ptr addrspace(4), !dbg [[DBG4:!.*]]
; CHECK:    #dbg_value(ptr addrspace(4) [[TMP4]], [[META13:!.*]], !DIExpression(), [[DBG4]])
  call spir_func void @bar(ptr addrspace(4) %tmp4, ptr byval(%struct.foo.0) align 1 %tmp1)
  ret void
}

define internal spir_func void @bar(ptr addrspace(4) %arg, ptr byval(%struct.foo.0) align 1 %arg1) align 2 !work_item_scope !0 !parallel_for_work_item !0 {
; CHECK-LABEL: define {{[^@]+}}@bar
; CHECK-SAME:  !dbg [[DBGBAR:!.*]]
bb:
  ret void
}

!0 = !{}
