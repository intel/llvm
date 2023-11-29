; Source:
; /*** for ***/
; void for_count()
; {
;     __attribute__((opencl_unroll_hint(1)))
;     for( int i = 0; i < 1024; ++i) {
;        if(i%2) continue;
;        int x = i;
;     }
; }
;
; /*** while ***/
; void while_count()
; {
;     int i = 1024;
;     __attribute__((opencl_unroll_hint(8)))
;     while(i-->0) {
;       if(i%2) continue;
;       int x = i;
;     }
; }
;
; /*** do ***/
; void do_count()
; {
;     int i = 1024;
;     __attribute__((opencl_unroll_hint))
;     do {
;       if(i%2) continue;
;       int x = i;
;    } while(i--> 0);
; }
;
; for_count_unusual() is a synthetically written function
;
; Command:
; clang -cc1 -triple spir64 -O0 LoopUnroll.cl -emit-llvm -o /test/SPIRV/transcoding/LoopUnroll.ll
;
; unroll_full() test was generated from the following source with -O2
; void foo();
;
; void unroll_full() {
;   #pragma clang unroll(full)
;   for (int i = 0; i != 1024; ++i) {
;     foo();
;   }
; }

; RUN: llvm-as < %s > %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-max-version=1.1
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV-NEGATIVE

; Check SPIR-V versions in a format magic number + version
; CHECK-SPIRV: 119734787 66560
; CHECK-SPIRV-NEGATIVE: 119734787 65536

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK-SPIRV: Function
; Function Attrs: noinline nounwind optnone
define spir_func void @for_count() #0 {
entry:
; CHECK-SPIRV: Label
  %i = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
; CHECK-SPIRV: Label [[#HEADER:]]
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 1024
; Per SPIRV spec p3.23 "DontUnroll" loop control = 0x2
; CHECK-SPIRV: LoopMerge [[#MERGEBLOCK:]] [[#CONTINUE:]] 2
; CHECK-SPIRV: BranchConditional [[#]] [[#]] [[#MERGEBLOCK]]
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
; CHECK-SPIRV: Label
  %1 = load i32, ptr %i, align 4
  %rem = srem i32 %1, 2
  %tobool = icmp ne i32 %rem, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
; CHECK-SPIRV: Label
  br label %for.inc

if.end:                                           ; preds = %for.body
; CHECK-SPIRV: Label
  %2 = load i32, ptr %i, align 4
  store i32 %2, ptr %x, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end, %if.then
; CHECK-SPIRV: Label [[#CONTINUE]]
  %3 = load i32, ptr %i, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !5
; CHECK-LLVM: br label %for.cond, !llvm.loop ![[#UNROLLDISABLE:]]
; CHECK-SPIRV: Branch [[#HEADER]]

for.end:                                          ; preds = %for.cond
; CHECK-SPIRV: Label [[#MERGEBLOCK]]
  ret void
}

; CHECK-SPIRV: Function
; Function Attrs: noinline nounwind optnone
define spir_func void @while_count() #0 {
entry:
; CHECK-SPIRV: Label
  %i = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 1024, ptr %i, align 4
  br label %while.cond

while.cond:                                       ; preds = %if.end, %if.then, %entry
; CHECK-SPIRV: Label [[#HEADER:]]
  %0 = load i32, ptr %i, align 4
  %dec = add nsw i32 %0, -1
  store i32 %dec, ptr %i, align 4
  %cmp = icmp sgt i32 %0, 0
; Per SPIRV spec p3.23 "Unroll" loop control = 0x1
; CHECK-SPIRV: LoopMerge [[#MERGEBLOCK:]] [[#CONTINUE:]] 256 8
; CHECK-SPIRV: BranchConditional [[#]] [[#]] [[#MERGEBLOCK]]
; CHECK-SPIRV-NEGATIVE-NOT: LoopMerge {{.*}} 256
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
; CHECK-SPIRV: Label
  %1 = load i32, ptr %i, align 4
  %rem = srem i32 %1, 2
  %tobool = icmp ne i32 %rem, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %while.body
; CHECK-SPIRV: Label
; CHECK-LLVM: br label %while.cond, !llvm.loop ![[#UNROLLCOUNT:]]
  br label %while.cond, !llvm.loop !7

; loop-simplify pass will create extra basic block which is the only one in
; loop having a back-edge to the header
; CHECK-SPIRV: [[#CONTINUE]]
; CHECK-SPIRV: Branch [[#HEADER]]

if.end:                                           ; preds = %while.body
; CHECK-SPIRV: Label
  %2 = load i32, ptr %i, align 4
  store i32 %2, ptr %x, align 4
  br label %while.cond, !llvm.loop !7

while.end:                                        ; preds = %while.cond
; CHECK-SPIRV: [[#MERGEBLOCK]]
  ret void
}

; CHECK-SPIRV: Function
; Function Attrs: noinline nounwind optnone
define spir_func void @do_count() #0 {
entry:
; CHECK-SPIRV: Label
  %i = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 1024, ptr %i, align 4
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
; CHECK-SPIRV: Label [[#HEADER:]]
  %0 = load i32, ptr %i, align 4
  %rem = srem i32 %0, 2
  %tobool = icmp ne i32 %rem, 0
; Per SPIRV spec p3.23 "Unroll" loop control = 0x1
; CHECK-SPIRV: LoopMerge [[#MERGEBLOCK:]] [[#CONTINUE:]] 1
; CHECK-SPIRV: BranchConditional
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %do.body
; CHECK-SPIRV: Label
  br label %do.cond

if.end:                                           ; preds = %do.body
; CHECK-SPIRV: Label
  %1 = load i32, ptr %i, align 4
  store i32 %1, ptr %x, align 4
  br label %do.cond

do.cond:                                          ; preds = %if.end, %if.then
; CHECK-SPIRV: Label [[#CONTINUE]]
  %2 = load i32, ptr %i, align 4
  %dec = add nsw i32 %2, -1
  store i32 %dec, ptr %i, align 4
  %cmp = icmp sgt i32 %2, 0
; CHECK-SPIRV: BranchConditional [[#]] [[#HEADER]] [[#MERGEBLOCK]]
; CHECK-LLVM: br i1 %cmp, label %do.body, label %do.end, !llvm.loop ![[#UNROLLENABLE1:]]
  br i1 %cmp, label %do.body, label %do.end, !llvm.loop !9

do.end:                                           ; preds = %do.cond
; CHECK-SPIRV: Label [[#MERGEBLOCK]]
  ret void
}

; CHECK-SPIRV: Function
; Function Attrs: noinline nounwind optnone
define spir_func void @for_count_unusual() #0 {
entry:
; CHECK-SPIRV: Label
  %i = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 1024, ptr %i, align 4
  br label %for.body

for.body:                                          ; preds = %for.cond, %entry
; CHECK-SPIRV: Label [[#HEADER:]]
  %0 = load i32, ptr %i, align 4
  %rem = srem i32 %0, 2
  %tobool = icmp ne i32 %rem, 0
; Per SPIRV spec p3.23 "Unroll" loop control = 0x1
; CHECK-SPIRV: LoopMerge [[#MERGEBLOCK:]] [[#CONTINUE:]] 1
; CHECK-SPIRV: BranchConditional
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
; CHECK-SPIRV: Label
  br label %for.cond

if.end:                                           ; preds = %for.body
; CHECK-SPIRV: Label
  %1 = load i32, ptr %i, align 4
  store i32 %1, ptr %x, align 4
  br label %for.cond

for.cond:                                          ; preds = %if.end, %if.then
; CHECK-SPIRV: Label [[#CONTINUE]]
  %2 = load i32, ptr %i, align 4
  %dec = add nsw i32 %2, -1
  store i32 %dec, ptr %i, align 4
  %cmp = icmp sgt i32 %2, 0
; CHECK-SPIRV: BranchConditional [[#]] [[#MERGEBLOCK]] [[#HEADER]]
; CHECK-LLVM: br i1 %cmp, label %for.end, label %for.body, !llvm.loop ![[#UNROLLENABLE2:]]
  br i1 %cmp, label %for.end, label %for.body, !llvm.loop !9

for.end:                                           ; preds = %for.cond
; CHECK-SPIRV: Label [[#MERGEBLOCK]]
  ret void
}

; CHECK-SPIRV: Function
; CHECK-SPIRV: Label
; CHECK-SPIRV: Branch [[#Header:]]
; CHECK-SPIRV: LoopMerge [[#Return:]] [[#Header]] 257 1
; CHECK-SPIRV: BranchConditional [[#]] [[#Return]] [[#Header]]
; CHECK-SPIRV: Label [[#Return]]
; Function Attrs: noinline nounwind optnone
define spir_func void @unroll_full() {
  br label %2

1:                                                ; preds = %2
  ret void

2:                                                ; preds = %0, %2
  %3 = phi i32 [ 0, %0 ], [ %4, %2 ]
  tail call void @_Z3foov()
  %4 = add nuw nsw i32 %3, 1
  %5 = icmp eq i32 %4, 1024
  ; CHECK-LLVM: br i1 %[[#]], label %[[#]], label %[[#]], !llvm.loop ![[#FULL:]]
  br i1 %5, label %1, label %2, !llvm.loop !11
}

declare void @_Z3foov()

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{}
!4 = !{!"clang version 5.0.1 (cfe/trunk)"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.unroll.disable"}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.unroll.count", i32 8}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.enable"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.unroll.full"}

; CHECK-LLVM: ![[#UNROLLDISABLE]] = distinct !{![[#UNROLLDISABLE]], ![[#DISABLE:]]}
; CHECK-LLVM: ![[#DISABLE]] = !{!"llvm.loop.unroll.disable"}
; CHECK-LLVM: ![[#UNROLLCOUNT]] = distinct !{![[#UNROLLCOUNT]], ![[#COUNT:]]}
; CHECK-LLVM: ![[#COUNT]] = !{!"llvm.loop.unroll.count", i32 8}
; CHECK-LLVM: ![[#UNROLLENABLE1]] = distinct !{![[#UNROLLENABLE1]], ![[#ENABLE:]]}
; CHECK-LLVM: ![[#ENABLE]] = !{!"llvm.loop.unroll.enable"}
; CHECK-LLVM: ![[#UNROLLENABLE2]] = distinct !{![[#UNROLLENABLE2]], ![[#ENABLE]]}
; CHECK-LLVM: ![[#FULL]] = distinct !{![[#FULL]], ![[#UNROLLFULL:]]}
; CHECK-LLVM: ![[#UNROLLFULL]] = !{!"llvm.loop.unroll.full"}
