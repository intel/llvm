; RUN: opt < %s -passes=debugify,LowerWGScope -S | FileCheck %s
;
; Check that debug info is not lost after LowerWGScope pass.
; Case with private_mem usage.

%struct.snork = type { i8 }
%struct.wobble = type { %struct.spam, %struct.spam, %struct.spam, %struct.snork.0 }
%struct.spam = type { %struct.foo }
%struct.foo = type { [2 x i64] }
%struct.snork.0 = type { %struct.foo }
%struct.ham = type { %struct.pluto }
%struct.pluto = type { i32, i32 }
%struct.wibble = type { ptr addrspace(4) }

define internal spir_func void @wibble(ptr addrspace(4) dereferenceable_or_null(1) %arg, ptr byval(%struct.wobble) align 8 %arg1) align 2 !work_group_scope !0 {
; CHECK-LABEL: define {{[^@]+}}@wibble
; CHECK-SAME: (ptr addrspace(4) dereferenceable_or_null(1) [[ARG:%.*]], ptr byval(%struct.wobble) align 8 [[ARG1:%.*]])
;
bb:
  %tmp = alloca ptr addrspace(4), align 8
; CHECK:    [[TMP:%.*]] = alloca ptr addrspace(4), align 8, !dbg [[DBG1:!.*]]
; CHECK:    #dbg_value(ptr [[TMP]], [[META9:!.*]], !DIExpression(), [[DBG1]])
  %tmp2 = addrspacecast ptr %tmp to ptr addrspace(4)
; CHECK:    [[TMP2:%.*]] = addrspacecast ptr [[TMP]] to ptr addrspace(4), !dbg [[DBG2:!.*]]
; CHECK:    #dbg_value(ptr addrspace(4) [[TMP2]], [[META11:!.*]], !DIExpression(), [[DBG2]])
  %tmp3 = alloca %struct.ham, align 4, !work_item_scope !0
; CHECK:    [[TMP3:%.*]] = alloca %struct.ham, align 4, !dbg [[DBG3:!.*]], !work_item_scope ![[#]]
; CHECK:    #dbg_value(ptr [[TMP3]], [[META12:!.*]], !DIExpression(), [[DBG3]])
  %tmp4 = addrspacecast ptr %tmp3 to ptr addrspace(4)
; CHECK:    [[TMP4:%.*]] = addrspacecast ptr [[TMP3]] to ptr addrspace(4), !dbg [[DBG4:!.*]]
; CHECK:    #dbg_value(ptr addrspace(4) [[TMP4]], [[META13:!.*]], !DIExpression(), [[DBG4]])
  %tmp5 = alloca %struct.spam, align 8
; CHECK:    [[TMP5:%.*]] = alloca %struct.spam, align 8, !dbg [[DBG5:!.*]]
; CHECK:    #dbg_value(ptr [[TMP5]], [[META14:!.*]], !DIExpression(), [[DBG5]])
  %tmp6 = addrspacecast ptr %tmp5 to ptr addrspace(4)
; CHECK:    [[TMP6:%.*]] = addrspacecast ptr [[TMP5]] to ptr addrspace(4), !dbg [[DBG6:!.*]]
; CHECK:    #dbg_value(ptr addrspace(4) [[TMP6]], [[META15:!.*]], !DIExpression(), [[DBG6]])
  %tmp7 = alloca %struct.wibble, align 8
; CHECK:    [[TMP7:%.*]] = alloca %struct.wibble, align 8, !dbg [[DBG7:!.*]]
; CHECK:    #dbg_value(ptr [[TMP7]], [[META16:!.*]], !DIExpression(), [[DBG7]])
  %tmp8 = addrspacecast ptr %tmp7 to ptr addrspace(4)
; CHECK:    [[TMP8:%.*]] = addrspacecast ptr [[TMP7]] to ptr addrspace(4), !dbg [[DBG8:!.*]]
; CHECK:    #dbg_value(ptr addrspace(4) [[TMP8]], [[META17:!.*]], !DIExpression(), [[DBG8]])
  store ptr addrspace(4) %arg, ptr addrspace(4) %tmp2, align 8
  %tmp9 = addrspacecast ptr %arg1 to ptr addrspace(4)
; CHECK:    [[TMP9:%.*]] = addrspacecast ptr [[ARG1]] to ptr addrspace(4), !dbg [[DBG9:!.*]]
; CHECK:    #dbg_value(ptr addrspace(4) [[TMP9]], [[META18:!.*]], !DIExpression(), [[DBG9]])
  call spir_func void @eggs(ptr addrspace(4) dereferenceable_or_null(8) %tmp4, ptr addrspace(4) align 8 dereferenceable(64) %tmp9)
  call spir_func void @snork(ptr addrspace(4) dereferenceable_or_null(16) %tmp6, i64 7, i64 3)
  store ptr addrspace(4) %tmp4, ptr addrspace(4) %tmp8, align 8
  %tmp11 = addrspacecast ptr addrspace(4) %tmp6 to ptr
; CHECK:    [[TMP11:%.*]] = addrspacecast ptr addrspace(4) [[TMP6]] to ptr, !dbg [[DBG11:!.*]]
; CHECK:    #dbg_value(ptr [[TMP11]], [[META20:!.*]], !DIExpression(), [[DBG11]])
  %tmp12 = addrspacecast ptr addrspace(4) %tmp8 to ptr
  call spir_func void @wombat(ptr addrspace(4) dereferenceable_or_null(64) %tmp9, ptr byval(%struct.spam) align 8 %tmp11, ptr byval(%struct.wibble) align 8 %tmp12)
; CHECK:    [[TMP12:%.*]] = addrspacecast ptr addrspace(4) [[TMP8]] to ptr, !dbg [[DBG12:!.*]]
; CHECK:    #dbg_value(ptr [[TMP12]], [[META21:!.*]], !DIExpression(), [[DBG12]])
  ret void
}

define linkonce_odr dso_local spir_func void @eggs(ptr addrspace(4) dereferenceable_or_null(8) %arg, ptr addrspace(4) align 8 dereferenceable(64) %arg1) align 2 {
bb:
  ret void
}

define internal spir_func void @wombat(ptr addrspace(4) dereferenceable_or_null(64) %arg, ptr byval(%struct.spam) align 8 %arg1, ptr byval(%struct.wibble) align 8 %arg2) align 2 !work_item_scope !0 !parallel_for_work_item !0 {
bb:
; CHECK-LABEL: define {{[^@]+}}@wombat
; CHECK:       !dbg [[DBGWOMBAT:!.*]]
  ret void
}

define linkonce_odr dso_local spir_func void @snork(ptr addrspace(4) dereferenceable_or_null(16) %arg, i64 %arg1, i64 %arg2) align 2 {
bb:
  ret void
}

!0 = !{}
