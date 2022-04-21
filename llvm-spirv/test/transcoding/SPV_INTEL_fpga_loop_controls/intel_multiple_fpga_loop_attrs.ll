; RUN: llvm-as < %s > %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_loop_controls -o - -spirv-text | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_loop_controls -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc -o - -spirv-text | FileCheck %s --check-prefix=CHECK-SPIRV-NEGATIVE

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-NEGATIVE

; CHECK-SPIRV: Capability FPGALoopControlsINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_fpga_loop_controls"
; CHECK-SPIRV-NEGATIVE-NOT: Capability FPGALoopControlsINTEL
; CHECK-SPIRV-NEGATIVE-NOT: Extension "SPV_INTEL_fpga_loop_controls"
; CHECK-SPIRV: 4522248 3 2 1 1 16 3 0
; CHECK-SPIRV-NEGATIVE: LoopMerge {{[0-9]+}} {{[0-9]+}} 264 3 2

; CHECK-LLVM: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD:[0-9]+]]
; CHECK-LLVM: ![[MD]] = distinct !{![[MD]], ![[MD_ivdep:[0-9]+]], ![[MD_unroll:[0-9]+]], ![[MD_ii:[0-9]+]], ![[MD_access:[0-9]+]], ![[MD_si:[0-9]+]]}
; CHECK-LLVM: ![[MD_ivdep]] = !{!"llvm.loop.ivdep.safelen", i32 3}
; CHECK-LLVM: ![[MD_unroll]] = !{!"llvm.loop.unroll.count", i32 2}
; CHECK-LLVM: ![[MD_ii]] = !{!"llvm.loop.ii.count", i32 1}
; CHECK-LLVM: ![[MD_access]] = !{!"llvm.loop.parallel_access_indices", !{{[0-9]+}}, i32 3}
; CHECK-LLVM: ![[MD_si]] = !{!"llvm.loop.intel.speculated.iterations.count", i32 0}

; CHECK-LLVM-NEGATIVE: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD:[0-9]+]]
; CHECK-LLVM-NEGATIVE: ![[MD]] = distinct !{![[MD]], ![[MD_ivdep:[0-9]+]], ![[MD_unroll:[0-9]+]]}
; CHECK-LLVM-NEGATIVE: ![[MD_ivdep]] = !{!"llvm.loop.ivdep.safelen", i32 3}
; CHECK-LLVM-NEGATIVE: ![[MD_unroll]] = !{!"llvm.loop.unroll.count", i32 2}

; ModuleID = 'intel-fpga-loops.cpp'
source_filename = "intel-fpga-loops.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-linux"

%class.anon = type { i8 }

; Function Attrs: nounwind
define spir_func void @_Z4testv() #0 {
entry:
  %a = alloca [10 x i32], align 4
  %i = alloca i32, align 4
  %0 = bitcast [10 x i32]* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* %0) #4
  %1 = bitcast i32* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #4
  store i32 0, i32* %i, align 4, !tbaa !2
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, i32* %i, align 4, !tbaa !2
  %cmp = icmp ne i32 %2, 10
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %3 = bitcast i32* %i to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %3) #4
  br label %for.end

for.body:                                         ; preds = %for.cond
  %4 = load i32, i32* %i, align 4, !tbaa !2
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* %a, i64 0, i64 %idxprom, !llvm.index.group !6
  store i32 0, i32* %arrayidx, align 4, !tbaa !2
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4, !tbaa !2
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4, !tbaa !2
  br label %for.cond, !llvm.loop !7

for.end:                                          ; preds = %for.cond.cleanup
  %6 = bitcast [10 x i32]* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 40, i8* %6) #4
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: norecurse nounwind
define i32 @main() #2 {
entry:
  %retval = alloca i32, align 4
  %agg.tmp = alloca %class.anon, align 1
  store i32 0, i32* %retval, align 4
  call spir_func void @"_Z18kernel_single_taskIZ4mainE15kernel_functionZ4mainE3$_0EvT0_"(%class.anon* byval(%class.anon) align 1 %agg.tmp)
  ret i32 0
}

; Function Attrs: nounwind
define internal spir_func void @"_Z18kernel_single_taskIZ4mainE15kernel_functionZ4mainE3$_0EvT0_"(%class.anon* byval(%class.anon) align 1 %kernelFunc) #0 {
entry:
  %0 = addrspacecast %class.anon* %kernelFunc to %class.anon addrspace(4)*
  call spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon addrspace(4)* %0)
  ret void
}

; Function Attrs: inlinehint nounwind
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon addrspace(4)* %this) #3 align 2 {
entry:
  %this.addr = alloca %class.anon addrspace(4)*, align 8
  store %class.anon addrspace(4)* %this, %class.anon addrspace(4)** %this.addr, align 8, !tbaa !13
  %this1 = load %class.anon addrspace(4)*, %class.anon addrspace(4)** %this.addr, align 8
  call spir_func void @_Z4testv()
  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.0"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = distinct !{}
!7 = distinct !{!7, !8, !9, !10, !11, !12}
!8 = !{!"llvm.loop.parallel_access_indices", !6, i32 3}
!9 = !{!"llvm.loop.ivdep.safelen", i32 3}
!10 = !{!"llvm.loop.ii.count", i32 1}
!11 = !{!"llvm.loop.intel.speculated.iterations.count", i32 0}
!12 = !{!"llvm.loop.unroll.count", i32 2}
!13 = !{!14, !14, i64 0}
!14 = !{!"any pointer", !4, i64 0}
