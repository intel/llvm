; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "n32:64"

define void @MainKernel(i32 %iNumSteps, i32 %tid, i32 %base) {
; CHECK-LABEL: @MainKernel(
; CHECK-NEXT:    [[CALLA:%.*]] = alloca [258 x float], align 4
; CHECK-NEXT:    [[CALLB:%.*]] = alloca [258 x float], align 4
; CHECK-NEXT:    [[CONV_I:%.*]] = uitofp i32 [[INUMSTEPS:%.*]] to float
; CHECK-NEXT:    [[CONV_I12:%.*]] = zext i32 [[TID:%.*]] to i64
; CHECK-NEXT:    [[ARRAYIDX3:%.*]] = getelementptr inbounds nuw [258 x float], ptr [[CALLA]], i64 0, i64 [[CONV_I12]]
; CHECK-NEXT:    store float [[CONV_I]], ptr [[ARRAYIDX3]], align 4
; CHECK-NEXT:    [[ARRAYIDX6:%.*]] = getelementptr inbounds nuw [258 x float], ptr [[CALLB]], i64 0, i64 [[CONV_I12]]
; CHECK-NEXT:    store float [[CONV_I]], ptr [[ARRAYIDX6]], align 4
; CHECK-NEXT:    [[CMP7:%.*]] = icmp eq i32 [[TID]], 0
; CHECK-NEXT:    br i1 [[CMP7]], label [[DOTBB1:%.*]], label [[DOTBB2:%.*]]
; CHECK:       .bb1:
; CHECK-NEXT:    [[ARRAYIDX10:%.*]] = getelementptr inbounds nuw i8, ptr [[CALLA]], i64 1024
; CHECK-NEXT:    store float [[CONV_I]], ptr [[ARRAYIDX10]], align 4
; CHECK-NEXT:    [[ARRAYIDX11:%.*]] = getelementptr inbounds nuw i8, ptr [[CALLB]], i64 1024
; CHECK-NEXT:    store float 0.000000e+00, ptr [[ARRAYIDX11]], align 4
; CHECK-NEXT:    br label [[DOTBB2]]
; CHECK:       .bb2:
; CHECK-NEXT:    [[CMP135:%.*]] = icmp sgt i32 [[INUMSTEPS]], 0
; CHECK-NEXT:    br i1 [[CMP135]], label [[DOTBB3:%.*]], label [[DOTBB8:%.*]]
; CHECK:       .bb3:
; CHECK-NEXT:    [[TMP1:%.*]] = phi float [ [[TMP10:%.*]], [[DOTBB12:%.*]] ], [ [[CONV_I]], [[DOTBB2]] ]
; CHECK-NEXT:    [[TMP2:%.*]] = phi float [ [[TMP11:%.*]], [[DOTBB12]] ], [ [[CONV_I]], [[DOTBB2]] ]
; CHECK-NEXT:    [[I12_06:%.*]] = phi i32 [ [[SUB:%.*]], [[DOTBB12]] ], [ [[INUMSTEPS]], [[DOTBB2]] ]
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ugt i32 [[I12_06]], [[BASE:%.*]]
; CHECK-NEXT:    [[ADD:%.*]] = add nuw i32 [[I12_06]], 1
; CHECK-NEXT:    [[CONV_I9:%.*]] = sext i32 [[ADD]] to i64
; CHECK-NEXT:    [[ARRAYIDX20:%.*]] = getelementptr inbounds [258 x float], ptr [[CALLA]], i64 0, i64 [[CONV_I9]]
; CHECK-NEXT:    [[ARRAYIDX24:%.*]] = getelementptr inbounds [258 x float], ptr [[CALLB]], i64 0, i64 [[CONV_I9]]
; CHECK-NEXT:    [[CMP40:%.*]] = icmp ult i32 [[I12_06]], [[BASE]]
; CHECK-NEXT:    br i1 [[TMP3]], label [[DOTBB4:%.*]], label [[DOTBB5:%.*]]
; CHECK:       .bb4:
; CHECK-NEXT:    [[TMP4:%.*]] = load float, ptr [[ARRAYIDX20]], align 4
; CHECK-NEXT:    [[TMP5:%.*]] = load float, ptr [[ARRAYIDX24]], align 4
; CHECK-NEXT:    [[ADD33:%.*]] = fadd float [[TMP5]], [[TMP4]]
; CHECK-NEXT:    [[ADD33_1:%.*]] = fadd float [[ADD33]], [[TMP1]]
; CHECK-NEXT:    [[ADD33_2:%.*]] = fadd float [[ADD33_1]], [[TMP2]]
; CHECK-NEXT:    br label [[DOTBB5]]
; CHECK:       .bb5:
; CHECK-NEXT:    [[TMP6:%.*]] = phi float [ [[ADD33_1]], [[DOTBB4]] ], [ [[TMP1]], [[DOTBB3]] ]
; CHECK-NEXT:    [[TMP7:%.*]] = phi float [ [[ADD33_2]], [[DOTBB4]] ], [ [[TMP2]], [[DOTBB3]] ]
; CHECK-NEXT:    br i1 [[CMP40]], label [[DOTBB6:%.*]], label [[DOTBB7:%.*]]
; CHECK:       .bb6:
; CHECK-NEXT:    store float [[TMP7]], ptr [[ARRAYIDX3]], align 4
; CHECK-NEXT:    store float [[TMP6]], ptr [[ARRAYIDX6]], align 4
; CHECK-NEXT:    br label [[DOTBB7]]
; CHECK:       .bb7:
; CHECK-NEXT:    br i1 [[TMP3]], label [[DOTBB9:%.*]], label [[DOTBB10:%.*]]
; CHECK:       .bb8:
; CHECK-NEXT:    ret void
; CHECK:       .bb9:
; CHECK-NEXT:    [[TMP8:%.*]] = load float, ptr [[ARRAYIDX20]], align 4
; CHECK-NEXT:    [[TMP9:%.*]] = load float, ptr [[ARRAYIDX24]], align 4
; CHECK-NEXT:    [[ADD33_112:%.*]] = fadd float [[TMP9]], [[TMP8]]
; CHECK-NEXT:    [[ADD33_1_1:%.*]] = fadd float [[ADD33_112]], [[TMP6]]
; CHECK-NEXT:    [[ADD33_2_1:%.*]] = fadd float [[ADD33_1_1]], [[TMP7]]
; CHECK-NEXT:    br label [[DOTBB10]]
; CHECK:       .bb10:
; CHECK-NEXT:    [[TMP10]] = phi float [ [[ADD33_1_1]], [[DOTBB9]] ], [ [[TMP6]], [[DOTBB7]] ]
; CHECK-NEXT:    [[TMP11]] = phi float [ [[ADD33_2_1]], [[DOTBB9]] ], [ [[TMP7]], [[DOTBB7]] ]
; CHECK-NEXT:    br i1 [[CMP40]], label [[DOTBB11:%.*]], label [[DOTBB12]]
; CHECK:       .bb11:
; CHECK-NEXT:    store float [[TMP11]], ptr [[ARRAYIDX3]], align 4
; CHECK-NEXT:    store float [[TMP10]], ptr [[ARRAYIDX6]], align 4
; CHECK-NEXT:    br label [[DOTBB12]]
; CHECK:       .bb12:
; CHECK-NEXT:    [[SUB]] = add nsw i32 [[I12_06]], -4
; CHECK-NEXT:    [[CMP13:%.*]] = icmp sgt i32 [[I12_06]], 4
; CHECK-NEXT:    br i1 [[CMP13]], label [[DOTBB3]], label [[DOTBB8]]
;
  %callA = alloca [258 x float], align 4
  %callB = alloca [258 x float], align 4
  %conv.i = uitofp i32 %iNumSteps to float
  %1 = bitcast float %conv.i to i32
  %conv.i12 = zext i32 %tid to i64
  %arrayidx3 = getelementptr inbounds [258 x float], ptr %callA, i64 0, i64 %conv.i12
  store i32 %1, ptr %arrayidx3, align 4
  %arrayidx6 = getelementptr inbounds [258 x float], ptr %callB, i64 0, i64 %conv.i12
  store i32 %1, ptr %arrayidx6, align 4
  %cmp7 = icmp eq i32 %tid, 0
  br i1 %cmp7, label %.bb1, label %.bb2

.bb1:
  %arrayidx10 = getelementptr inbounds [258 x float], ptr %callA, i64 0, i64 256
  store float %conv.i, ptr %arrayidx10, align 4
  %arrayidx11 = getelementptr inbounds [258 x float], ptr %callB, i64 0, i64 256
  store float 0.000000e+00, ptr %arrayidx11, align 4
  br label %.bb2

.bb2:
  %cmp135 = icmp sgt i32 %iNumSteps, 0
  br i1 %cmp135, label %.bb3, label %.bb8

.bb3:
  %rA.sroa.8.0 = phi i32 [ %rA.sroa.8.2, %.bb12 ], [ %1, %.bb2 ]
  %rA.sroa.0.0 = phi i32 [ %rA.sroa.0.2, %.bb12 ], [ %1, %.bb2 ]
  %i12.06 = phi i32 [ %sub, %.bb12 ], [ %iNumSteps, %.bb2 ]
  %2 = icmp ugt i32 %i12.06, %base
  %add = add i32 %i12.06, 1
  %conv.i9 = sext i32 %add to i64
  %arrayidx20 = getelementptr inbounds [258 x float], ptr %callA, i64 0, i64 %conv.i9
  %arrayidx24 = getelementptr inbounds [258 x float], ptr %callB, i64 0, i64 %conv.i9
  %cmp40 = icmp ult i32 %i12.06, %base
  br i1 %2, label %.bb4, label %.bb5

.bb4:
  %3 = load i32, ptr %arrayidx20, align 4
  %4 = load i32, ptr %arrayidx24, align 4
  %5 = bitcast i32 %4 to float
  %6 = bitcast i32 %3 to float
  %add33 = fadd float %5, %6
  %7 = bitcast i32 %rA.sroa.8.0 to float
  %add33.1 = fadd float %add33, %7
  %8 = bitcast float %add33.1 to i32
  %9 = bitcast i32 %rA.sroa.0.0 to float
  %add33.2 = fadd float %add33.1, %9
  %10 = bitcast float %add33.2 to i32
  br label %.bb5

.bb5:
  %rA.sroa.8.1 = phi i32 [ %8, %.bb4 ], [ %rA.sroa.8.0, %.bb3 ]
  %rA.sroa.0.1 = phi i32 [ %10, %.bb4 ], [ %rA.sroa.0.0, %.bb3 ]
  br i1 %cmp40, label %.bb6, label %.bb7

.bb6:
  store i32 %rA.sroa.0.1, ptr %arrayidx3, align 4
  store i32 %rA.sroa.8.1, ptr %arrayidx6, align 4
  br label %.bb7

.bb7:
  br i1 %2, label %.bb9, label %.bb10

.bb8:
  ret void

.bb9:
  %11 = load i32, ptr %arrayidx20, align 4
  %12 = load i32, ptr %arrayidx24, align 4
  %13 = bitcast i32 %12 to float
  %14 = bitcast i32 %11 to float
  %add33.112 = fadd float %13, %14
  %15 = bitcast i32 %rA.sroa.8.1 to float
  %add33.1.1 = fadd float %add33.112, %15
  %16 = bitcast float %add33.1.1 to i32
  %17 = bitcast i32 %rA.sroa.0.1 to float
  %add33.2.1 = fadd float %add33.1.1, %17
  %18 = bitcast float %add33.2.1 to i32
  br label %.bb10

.bb10:
  %rA.sroa.8.2 = phi i32 [ %16, %.bb9 ], [ %rA.sroa.8.1, %.bb7 ]
  %rA.sroa.0.2 = phi i32 [ %18, %.bb9 ], [ %rA.sroa.0.1, %.bb7 ]
  br i1 %cmp40, label %.bb11, label %.bb12

.bb11:
  store i32 %rA.sroa.0.2, ptr %arrayidx3, align 4
  store i32 %rA.sroa.8.2, ptr %arrayidx6, align 4
  br label %.bb12

.bb12:
  %sub = add i32 %i12.06, -4
  %cmp13 = icmp sgt i32 %sub, 0
  br i1 %cmp13, label %.bb3, label %.bb8
}

declare i32 @get_i32()
declare i3 @get_i3()
declare void @bar()

define i37 @zext_from_legal_to_illegal_type(i32 %x) {
; CHECK-LABEL: @zext_from_legal_to_illegal_type(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], 42
; CHECK-NEXT:    br i1 [[CMP]], label [[T:%.*]], label [[F:%.*]]
; CHECK:       t:
; CHECK-NEXT:    [[Y:%.*]] = call i32 @get_i32()
; CHECK-NEXT:    br label [[EXIT:%.*]]
; CHECK:       f:
; CHECK-NEXT:    call void @bar()
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    [[P:%.*]] = phi i32 [ [[Y]], [[T]] ], [ 3, [[F]] ]
; CHECK-NEXT:    [[R:%.*]] = zext i32 [[P]] to i37
; CHECK-NEXT:    ret i37 [[R]]
;
entry:
  %cmp = icmp eq i32 %x, 42
  br i1 %cmp, label %t, label %f

t:
  %y = call i32 @get_i32()
  br label %exit

f:
  call void @bar()
  br label %exit

exit:
  %p = phi i32 [ %y, %t ], [ 3, %f ]
  %r = zext i32 %p to i37
  ret i37 %r
}

define i37 @zext_from_illegal_to_illegal_type(i32 %x) {
; CHECK-LABEL: @zext_from_illegal_to_illegal_type(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], 42
; CHECK-NEXT:    br i1 [[CMP]], label [[T:%.*]], label [[F:%.*]]
; CHECK:       t:
; CHECK-NEXT:    [[Y:%.*]] = call i3 @get_i3()
; CHECK-NEXT:    br label [[EXIT:%.*]]
; CHECK:       f:
; CHECK-NEXT:    call void @bar()
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    [[P:%.*]] = phi i3 [ [[Y]], [[T]] ], [ 3, [[F]] ]
; CHECK-NEXT:    [[R:%.*]] = zext i3 [[P]] to i37
; CHECK-NEXT:    ret i37 [[R]]
;
entry:
  %cmp = icmp eq i32 %x, 42
  br i1 %cmp, label %t, label %f

t:
  %y = call i3 @get_i3()
  br label %exit

f:
  call void @bar()
  br label %exit

exit:
  %p = phi i3 [ %y, %t ], [ 3, %f ]
  %r = zext i3 %p to i37
  ret i37 %r
}

define i64 @zext_from_legal_to_legal_type(i32 %x) {
; CHECK-LABEL: @zext_from_legal_to_legal_type(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], 42
; CHECK-NEXT:    br i1 [[CMP]], label [[T:%.*]], label [[F:%.*]]
; CHECK:       t:
; CHECK-NEXT:    [[Y:%.*]] = call i32 @get_i32()
; CHECK-NEXT:    [[TMP0:%.*]] = zext i32 [[Y]] to i64
; CHECK-NEXT:    br label [[EXIT:%.*]]
; CHECK:       f:
; CHECK-NEXT:    call void @bar()
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    [[P:%.*]] = phi i64 [ [[TMP0]], [[T]] ], [ 3, [[F]] ]
; CHECK-NEXT:    ret i64 [[P]]
;
entry:
  %cmp = icmp eq i32 %x, 42
  br i1 %cmp, label %t, label %f

t:
  %y = call i32 @get_i32()
  br label %exit

f:
  call void @bar()
  br label %exit

exit:
  %p = phi i32 [ %y, %t ], [ 3, %f ]
  %r = zext i32 %p to i64
  ret i64 %r
}

define i64 @zext_from_illegal_to_legal_type(i32 %x) {
; CHECK-LABEL: @zext_from_illegal_to_legal_type(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], 42
; CHECK-NEXT:    br i1 [[CMP]], label [[T:%.*]], label [[F:%.*]]
; CHECK:       t:
; CHECK-NEXT:    [[Y:%.*]] = call i3 @get_i3()
; CHECK-NEXT:    [[TMP0:%.*]] = zext i3 [[Y]] to i64
; CHECK-NEXT:    br label [[EXIT:%.*]]
; CHECK:       f:
; CHECK-NEXT:    call void @bar()
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    [[P:%.*]] = phi i64 [ [[TMP0]], [[T]] ], [ 3, [[F]] ]
; CHECK-NEXT:    ret i64 [[P]]
;
entry:
  %cmp = icmp eq i32 %x, 42
  br i1 %cmp, label %t, label %f

t:
  %y = call i3 @get_i3()
  br label %exit

f:
  call void @bar()
  br label %exit

exit:
  %p = phi i3 [ %y, %t ], [ 3, %f ]
  %r = zext i3 %p to i64
  ret i64 %r
}

define i8 @trunc_in_loop_exit_block() {
; CHECK-LABEL: @trunc_in_loop_exit_block(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[IV_NEXT:%.*]], [[LOOP_LATCH:%.*]] ]
; CHECK-NEXT:    [[PHI:%.*]] = phi i32 [ 1, [[ENTRY]] ], [ [[IV_NEXT]], [[LOOP_LATCH]] ]
; CHECK-NEXT:    [[CMP:%.*]] = icmp samesign ult i32 [[IV]], 100
; CHECK-NEXT:    br i1 [[CMP]], label [[LOOP_LATCH]], label [[EXIT:%.*]]
; CHECK:       loop.latch:
; CHECK-NEXT:    [[IV_NEXT]] = add nuw nsw i32 [[IV]], 1
; CHECK-NEXT:    br label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    [[TRUNC:%.*]] = trunc i32 [[PHI]] to i8
; CHECK-NEXT:    ret i8 [[TRUNC]]
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %phi = phi i32 [ 1, %entry ], [ %iv.next, %loop.latch ]
  %cmp = icmp ult i32 %iv, 100
  br i1 %cmp, label %loop.latch, label %exit

loop.latch:
  %iv.next = add i32 %iv, 1
  br label %loop

exit:
  %trunc = trunc i32 %phi to i8
  ret i8 %trunc
}

define i32 @zext_in_loop_and_exit_block(i8 %step, i32 %end) {
; CHECK-LABEL: @zext_in_loop_and_exit_block(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i8 [ 0, [[ENTRY:%.*]] ], [ [[IV_NEXT:%.*]], [[LOOP_LATCH:%.*]] ]
; CHECK-NEXT:    [[IV_EXT:%.*]] = zext i8 [[IV]] to i32
; CHECK-NEXT:    [[CMP_NOT:%.*]] = icmp eq i32 [[END:%.*]], [[IV_EXT]]
; CHECK-NEXT:    br i1 [[CMP_NOT]], label [[EXIT:%.*]], label [[LOOP_LATCH]]
; CHECK:       loop.latch:
; CHECK-NEXT:    [[IV_NEXT]] = add i8 [[IV]], [[STEP:%.*]]
; CHECK-NEXT:    br label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    [[EXT:%.*]] = zext i8 [[IV]] to i32
; CHECK-NEXT:    ret i32 [[EXT]]
;
entry:
  br label %loop

loop:
  %iv = phi i8 [ 0, %entry ], [ %iv.next.trunc, %loop.latch ]
  %iv.ext = zext i8 %iv to i32
  %cmp = icmp ne i32 %iv.ext, %end
  br i1 %cmp, label %loop.latch, label %exit

loop.latch:
  %step.ext = zext i8 %step to i32
  %iv.next = add i32 %iv.ext, %step.ext
  %iv.next.trunc = trunc i32 %iv.next to i8
  br label %loop

exit:
  %ext = zext i8 %iv to i32
  ret i32 %ext
}
