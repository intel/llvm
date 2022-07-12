; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir-unknown-unknown"

; Function Attrs: convergent nofree norecurse nounwind uwtable
define dso_local spir_kernel void @test_builtin_readnone(double* nocapture readonly %a, double* nocapture %b) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %0 = load double, double* %a, align 8, !tbaa !7
  %call = tail call double @_Z3expd(double %0) #2
  store double %call, double* %b, align 8, !tbaa !7
  %1 = load double, double* %a, align 8, !tbaa !7
  %call1 = tail call double @_Z3cosd(double %1) #2
  store double %call1, double* %b, align 8, !tbaa !7
  ret void
}

; Function Attrs: convergent nounwind readnone
; CHECK-LLVM: declare{{.*}}@_Z3expd{{.*}}#[[#Attrs:]]
declare dso_local double @_Z3expd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
; CHECK-LLVM: declare{{.*}}@_Z3cosd{{.*}}#[[#Attrs]]
declare dso_local double @_Z3cosd(double) local_unnamed_addr #1

attributes #0 = { convergent nofree norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
; CHECK-LLVM: attributes #[[#Attrs]] {{.*}} readnone
attributes #1 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind readnone }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 12.0.0 (https://github.com/intel/llvm 275e05b9dc13deb44eb7c765d23e65358d6bd077)"}
!3 = !{i32 1, i32 1}
!4 = !{!"none", !"none"}
!5 = !{!"double*", !"double*"}
!6 = !{!"", !""}
!7 = !{!8, !8, i64 0}
!8 = !{!"double", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
