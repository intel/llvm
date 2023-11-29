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
; CHECK:    [[TMP:%.*]] = alloca ptr addrspace(4), align 8
; CHECK:    call void @llvm.dbg.value(metadata ptr [[TMP]], [[META9:metadata !.*]], metadata !DIExpression())
  %tmp2 = addrspacecast ptr %tmp to ptr addrspace(4)
; CHECK:    [[TMP2:%.*]] = addrspacecast ptr [[TMP]] to ptr addrspace(4)
; CHECK:    call void @llvm.dbg.value(metadata ptr addrspace(4) [[TMP2]], [[META11:metadata !.*]], metadata !DIExpression())
  %tmp3 = alloca %struct.ham, align 4, !work_item_scope !0
; CHECK:    [[TMP3:%.*]] = alloca %struct.ham, align 4, [[DBG24:!dbg !.*]], !work_item_scope ![[#]]
; CHECK:    call void @llvm.dbg.value(metadata ptr [[TMP3]], [[META12:metadata !.*]], metadata !DIExpression())
  %tmp4 = addrspacecast ptr %tmp3 to ptr addrspace(4)
; CHECK:    [[TMP4:%.*]] = addrspacecast ptr [[TMP3]] to ptr addrspace(4)
; CHECK:    call void @llvm.dbg.value(metadata ptr addrspace(4) [[TMP4]], [[META13:metadata !.*]], metadata !DIExpression())
  %tmp5 = alloca %struct.spam, align 8
; CHECK:    [[TMP5:%.*]] = alloca %struct.spam, align 8
; CHECK:    call void @llvm.dbg.value(metadata ptr [[TMP5]], [[META14:metadata !.*]], metadata !DIExpression())
  %tmp6 = addrspacecast ptr %tmp5 to ptr addrspace(4)
; CHECK:    [[TMP6:%.*]] = addrspacecast ptr [[TMP5]] to ptr addrspace(4)
; CHECK:    call void @llvm.dbg.value(metadata ptr addrspace(4) [[TMP6]], [[META15:metadata !.*]], metadata !DIExpression())
  %tmp7 = alloca %struct.wibble, align 8
; CHECK:    [[TMP7:%.*]] = alloca %struct.wibble, align 8
; CHECK:    call void @llvm.dbg.value(metadata ptr [[TMP7]], [[META16:metadata !.*]], metadata !DIExpression())
  %tmp8 = addrspacecast ptr %tmp7 to ptr addrspace(4)
; CHECK:    [[TMP8:%.*]] = addrspacecast ptr [[TMP7]] to ptr addrspace(4)
; CHECK:    call void @llvm.dbg.value(metadata ptr addrspace(4) [[TMP8]], [[META17:metadata !.*]], metadata !DIExpression())
  store ptr addrspace(4) %arg, ptr addrspace(4) %tmp2, align 8
  %tmp9 = addrspacecast ptr %arg1 to ptr addrspace(4)
; CHECK:    [[TMP9:%.*]] = addrspacecast ptr [[ARG1]] to ptr addrspace(4)
; CHECK:    call void @llvm.dbg.value(metadata ptr addrspace(4) [[TMP9]], [[META18:metadata !.*]], metadata !DIExpression())
  call spir_func void @eggs(ptr addrspace(4) dereferenceable_or_null(8) %tmp4, ptr addrspace(4) align 8 dereferenceable(64) %tmp9)
  call spir_func void @snork(ptr addrspace(4) dereferenceable_or_null(16) %tmp6, i64 7, i64 3)
  store ptr addrspace(4) %tmp4, ptr addrspace(4) %tmp8, align 8
  %tmp11 = addrspacecast ptr addrspace(4) %tmp6 to ptr
; CHECK:    [[TMP11:%.*]] = addrspacecast ptr addrspace(4) [[TMP6]] to ptr
; CHECK:    call void @llvm.dbg.value(metadata ptr [[TMP11]], [[META20:metadata !.*]], metadata !DIExpression())
  %tmp12 = addrspacecast ptr addrspace(4) %tmp8 to ptr
  call spir_func void @wombat(ptr addrspace(4) dereferenceable_or_null(64) %tmp9, ptr byval(%struct.spam) align 8 %tmp11, ptr byval(%struct.wibble) align 8 %tmp12)
; CHECK:    [[TMP12:%.*]] = addrspacecast ptr addrspace(4) [[TMP8]] to ptr
; CHECK:    call void @llvm.dbg.value(metadata ptr [[TMP12]], [[META21:metadata !.*]], metadata !DIExpression())
  ret void
}

define linkonce_odr dso_local spir_func void @eggs(ptr addrspace(4) dereferenceable_or_null(8) %arg, ptr addrspace(4) align 8 dereferenceable(64) %arg1) align 2 {
bb:
  ret void
}

define internal spir_func void @wombat(ptr addrspace(4) dereferenceable_or_null(64) %arg, ptr byval(%struct.spam) align 8 %arg1, ptr byval(%struct.wibble) align 8 %arg2) align 2 !work_item_scope !0 !parallel_for_work_item !0 {
bb:
; CHECK:    call void @llvm.dbg.value(metadata i32 0, [[META42:metadata !.*]], metadata !DIExpression())
  ret void
}

define linkonce_odr dso_local spir_func void @snork(ptr addrspace(4) dereferenceable_or_null(16) %arg, i64 %arg1, i64 %arg2) align 2 {
bb:
  ret void
}

!0 = !{}
