; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_optimization_hints -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV-NO-EXT
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM-NO-EXT

; CHECK-SPIRV: Capability OptimizationHintsINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_optimization_hints"
; CHECK-SPIRV: Name [[COMPARE:[0-9]+]] "cmp"
; CHECK-SPIRV: INotEqual {{[0-9]+}} [[COMPARE]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: AssumeTrueINTEL [[COMPARE]]

; CHECK-LLVM: %cmp = icmp ne i32 %0, 0
; CHECK-LLVM: call void @llvm.assume(i1 %cmp)

; CHECK-SPIRV-NO-EXT-NOT: Capability OptimizationHintsINTEL
; CHECK-SPIRV-NO-EXT-NOT: Extension "SPV_INTEL_optimization_hints"
; CHECK-SPIRV-NO-EXT: Name [[COMPARE:[0-9]+]] "cmp"
; CHECK-SPIRV-NO-EXT: INotEqual {{[0-9]+}} [[COMPARE]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV-NO-EXT-NOT: AssumeTrueINTEL [[COMPARE]]

; CHECK-LLVM-NO-EXT: %cmp = icmp ne i32 %0, 0
; CHECK-LLVM-NO-EXT-NOT: call void @llvm.assume(i1 %cmp)

; ModuleID = 'assume.cpp'
source_filename = "assume.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%class.anon = type { i8 }

; Function Attrs: nounwind
define spir_func void @_Z3fooi(i32 %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4, !tbaa !2
  %0 = load i32, i32* %x.addr, align 4, !tbaa !2
  %cmp = icmp ne i32 %0, 0
  call void @llvm.assume(i1 %cmp)
  ret void
}

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1) #1

; Function Attrs: norecurse nounwind
define i32 @main() #2 {
entry:
  %retval = alloca i32, align 4
  %agg.tmp = alloca %class.anon, align 1
  store i32 0, i32* %retval, align 4
  call spir_func void @"_Z18kernel_single_taskIZ4mainE11fake_kernelZ4mainE3$_0EvT0_"(%class.anon* byval(%class.anon) align 1 %agg.tmp)
  ret i32 0
}

; Function Attrs: nounwind
define internal spir_func void @"_Z18kernel_single_taskIZ4mainE11fake_kernelZ4mainE3$_0EvT0_"(%class.anon* byval(%class.anon) align 1 %kernelFunc) #0 {
entry:
  call spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon* %kernelFunc)
  ret void
}

; Function Attrs: inlinehint nounwind
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon* %this) #3 align 2 {
entry:
  %this.addr = alloca %class.anon*, align 8
  %a = alloca i32, align 4
  store %class.anon* %this, %class.anon** %this.addr, align 8, !tbaa !6
  %this1 = load %class.anon*, %class.anon** %this.addr, align 8
  %0 = bitcast i32* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #5
  store i32 1, i32* %a, align 4, !tbaa !2
  %1 = load i32, i32* %a, align 4, !tbaa !2
  call spir_func void @_Z3fooi(i32 %1)
  %2 = bitcast i32* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %2) #5
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #4

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind willreturn }
attributes #2 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind willreturn }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}
