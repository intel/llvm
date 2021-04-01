; RUN: opt < %s -debugify -LowerWGScope -S | FileCheck %s
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
%struct.wibble = type { %struct.ham addrspace(4)* }

define internal spir_func void @wibble(%struct.snork addrspace(4)* dereferenceable_or_null(1) %arg, %struct.wobble* byval(%struct.wobble) align 8 %arg1) align 2 !work_group_scope !0 {
; CHECK-LABEL: define {{[^@]+}}@wibble
; CHECK-SAME: (%struct.snork addrspace(4)* dereferenceable_or_null(1) [[ARG:%.*]], %struct.wobble* byval(%struct.wobble) align 8 [[ARG1:%.*]])
;
bb:
  %tmp = alloca %struct.snork addrspace(4)*, align 8
; CHECK:    [[TMP:%.*]] = alloca %struct.snork addrspace(4)*, align 8
; CHECK:    call void @llvm.dbg.value(metadata %struct.snork addrspace(4)** [[TMP]], [[META9:metadata !.*]], metadata !DIExpression())
  %tmp2 = addrspacecast %struct.snork addrspace(4)** %tmp to %struct.snork addrspace(4)* addrspace(4)*
; CHECK:    [[TMP2:%.*]] = addrspacecast %struct.snork addrspace(4)** [[TMP]] to %struct.snork addrspace(4)* addrspace(4)*
; CHECK:    call void @llvm.dbg.value(metadata %struct.snork addrspace(4)* addrspace(4)* [[TMP2]], [[META11:metadata !.*]], metadata !DIExpression())
  %tmp3 = alloca %struct.ham, align 4, !work_item_scope !0
; CHECK:    [[TMP3:%.*]] = alloca %struct.ham, align 4, [[DBG24:!dbg !.*]], !work_item_scope !2
; CHECK:    call void @llvm.dbg.value(metadata %struct.ham* [[TMP3]], [[META12:metadata !.*]], metadata !DIExpression())
  %tmp4 = addrspacecast %struct.ham* %tmp3 to %struct.ham addrspace(4)*
; CHECK:    [[TMP4:%.*]] = addrspacecast %struct.ham* [[TMP3]] to %struct.ham addrspace(4)*
; CHECK:    call void @llvm.dbg.value(metadata %struct.ham addrspace(4)* [[TMP4]], [[META13:metadata !.*]], metadata !DIExpression())
  %tmp5 = alloca %struct.spam, align 8
; CHECK:    [[TMP5:%.*]] = alloca %struct.spam, align 8
; CHECK:    call void @llvm.dbg.value(metadata %struct.spam* [[TMP5]], [[META14:metadata !.*]], metadata !DIExpression())
  %tmp6 = addrspacecast %struct.spam* %tmp5 to %struct.spam addrspace(4)*
; CHECK:    [[TMP6:%.*]] = addrspacecast %struct.spam* [[TMP5]] to %struct.spam addrspace(4)*
; CHECK:    call void @llvm.dbg.value(metadata %struct.spam addrspace(4)* [[TMP6]], [[META15:metadata !.*]], metadata !DIExpression())
  %tmp7 = alloca %struct.wibble, align 8
; CHECK:    [[TMP7:%.*]] = alloca %struct.wibble, align 8
; CHECK:    call void @llvm.dbg.value(metadata %struct.wibble* [[TMP7]], [[META16:metadata !.*]], metadata !DIExpression())
  %tmp8 = addrspacecast %struct.wibble* %tmp7 to %struct.wibble addrspace(4)*
; CHECK:    [[TMP8:%.*]] = addrspacecast %struct.wibble* [[TMP7]] to %struct.wibble addrspace(4)*
; CHECK:    call void @llvm.dbg.value(metadata %struct.wibble addrspace(4)* [[TMP8]], [[META17:metadata !.*]], metadata !DIExpression())
  store %struct.snork addrspace(4)* %arg, %struct.snork addrspace(4)* addrspace(4)* %tmp2, align 8
  %tmp9 = addrspacecast %struct.wobble* %arg1 to %struct.wobble addrspace(4)*
; CHECK:    [[TMP9:%.*]] = addrspacecast %struct.wobble* [[ARG1]] to %struct.wobble addrspace(4)*
; CHECK:    call void @llvm.dbg.value(metadata %struct.wobble addrspace(4)* [[TMP9]], [[META18:metadata !.*]], metadata !DIExpression())
  call spir_func void @eggs(%struct.ham addrspace(4)* dereferenceable_or_null(8) %tmp4, %struct.wobble addrspace(4)* align 8 dereferenceable(64) %tmp9)
  call spir_func void @snork(%struct.spam addrspace(4)* dereferenceable_or_null(16) %tmp6, i64 7, i64 3)
  %tmp10 = getelementptr inbounds %struct.wibble, %struct.wibble addrspace(4)* %tmp8, i32 0, i32 0
; CHECK:    [[TMP10:%.*]] = getelementptr inbounds %struct.wibble, %struct.wibble addrspace(4)* [[TMP8]], i32 0, i32 0
; CHECK:    call void @llvm.dbg.value(metadata %struct.ham addrspace(4)* addrspace(4)* [[TMP10]], [[META19:metadata !.*]], metadata !DIExpression())
  store %struct.ham addrspace(4)* %tmp4, %struct.ham addrspace(4)* addrspace(4)* %tmp10, align 8
  %tmp11 = addrspacecast %struct.spam addrspace(4)* %tmp6 to %struct.spam*
; CHECK:    [[TMP11:%.*]] = addrspacecast %struct.spam addrspace(4)* [[TMP6]] to %struct.spam*
; CHECK:    call void @llvm.dbg.value(metadata %struct.spam* [[TMP11]], [[META20:metadata !.*]], metadata !DIExpression())
  %tmp12 = addrspacecast %struct.wibble addrspace(4)* %tmp8 to %struct.wibble*
  call spir_func void @wombat(%struct.wobble addrspace(4)* dereferenceable_or_null(64) %tmp9, %struct.spam* byval(%struct.spam) align 8 %tmp11, %struct.wibble* byval(%struct.wibble) align 8 %tmp12)
; CHECK:    [[TMP12:%.*]] = addrspacecast %struct.wibble addrspace(4)* [[TMP8]] to %struct.wibble*
; CHECK:    call void @llvm.dbg.value(metadata %struct.wibble* [[TMP12]], [[META21:metadata !.*]], metadata !DIExpression())
  ret void
}

define linkonce_odr dso_local spir_func void @eggs(%struct.ham addrspace(4)* dereferenceable_or_null(8) %arg, %struct.wobble addrspace(4)* align 8 dereferenceable(64) %arg1) align 2 {
bb:
  ret void
}

define internal spir_func void @wombat(%struct.wobble addrspace(4)* dereferenceable_or_null(64) %arg, %struct.spam* byval(%struct.spam) align 8 %arg1, %struct.wibble* byval(%struct.wibble) align 8 %arg2) align 2 !work_item_scope !0 !parallel_for_work_item !0 {
bb:
; CHECK:    call void @llvm.dbg.value(metadata i32 0, [[META42:metadata !.*]], metadata !DIExpression())
  ret void
}

define linkonce_odr dso_local spir_func void @snork(%struct.spam addrspace(4)* dereferenceable_or_null(16) %arg, i64 %arg1, i64 %arg2) align 2 {
bb:
  ret void
}

!0 = !{}
