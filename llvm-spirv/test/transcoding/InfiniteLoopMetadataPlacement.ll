; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_unstructured_loop_controls %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv --to-text -o - | FileCheck %s --check-prefix=CHECK-SPV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-REV-LLVM

; ModuleID = 'llvm_loop_test.cpp'
source_filename = "llvm_loop_test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

$_ZTS12WhileOneTest = comdat any

; CHECK-SPV: {{[0-9]+}} Name [[WH_COND:[0-9]+]] "while.cond"

; Function Attrs: inlinehint nounwind
define weak_odr dso_local spir_kernel void @_ZTS12WhileOneTest() #0 comdat !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %i = alloca i32, align 4
  %s = alloca i32, align 4
  %0 = bitcast i32* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #2
  store i32 0, i32* %i, align 4, !tbaa !7
  %1 = bitcast i32* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #2
  store i32 0, i32* %s, align 4, !tbaa !7
  br label %while.cond

; CHECK-SPV-NOT: {{[0-9]+}} LoopControlINTEL
; CHECK-SPV-NOT: {{[0-9]+}} LoopMerge

while.cond:                                       ; preds = %if.end, %entry
; CHECK-SPV:      {{[0-9]+}} Label [[WH_COND]]
; CHECK-SPV-NEXT: {{[0-9]+}} LoopControlINTEL 1
; CHECK-SPV-NEXT: {{[0-9]+}} Branch
  br label %while.body

; CHECK-SPV-NOT: {{[0-9]+}} LoopControlINTEL
; CHECK-SPV-NOT: {{[0-9]+}} LoopMerge

while.body:                                       ; preds = %while.cond
  %2 = load i32, i32* %i, align 4, !tbaa !7
  %cmp = icmp sge i32 %2, 16
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %while.body
  call spir_func void @_Z1fv() #0
  br label %while.end

if.else:                                          ; preds = %while.body
  %3 = load i32, i32* %i, align 4, !tbaa !7
  %4 = load i32, i32* %s, align 4, !tbaa !7
  %add = add nsw i32 %4, %3
  store i32 %add, i32* %s, align 4, !tbaa !7
  br label %if.end

; CHECK-REV-LLVM-NOT: br {{.*}}, !llvm.loop

if.end:                                           ; preds = %if.else
; CHECK-REV-LLVM: if.end:
  %5 = load i32, i32* %i, align 4, !tbaa !7
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4, !tbaa !7
  br label %while.cond, !llvm.loop !9
; CHECK-REV-LLVM: br label %while.cond, !llvm.loop ![[MD_UNROLL:[0-9]+]]

; CHECK-REV-LLVM-NOT: br {{.*}}, !llvm.loop

while.end:                                        ; preds = %if.then
  %6 = bitcast i32* %s to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %6) #2
  %7 = bitcast i32* %i to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %7) #2
  ret void
}

; Function Attrs: nounwind
define spir_func void @_Z1fv() #0 {
entry:
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 9.0.0"}
!4 = !{}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !5, i64 0}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.enable"}

; CHECK-REV-LLVM: ![[MD_UNROLL]] = distinct !{![[MD_UNROLL]], ![[MD_unroll_enable:[0-9]+]]}
; CHECK-REV-LLVM: ![[MD_unroll_enable]] = !{!"llvm.loop.unroll.enable"}
