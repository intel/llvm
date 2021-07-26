; Test checks debug info of producer is preserved from LLVM IR to spirv
; and spirv to LLVM IR translation.

; Original .cpp source:
;
;  int main() {
;    return 0;
;  }

; Command line:
; ./clang -cc1 -debug-info-kind=standalone -v s.cpp -S -emit-llvm -triple spir

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll --check-prefix CHECK-LLVM

; ModuleID = 's.bc'
source_filename = "s.cpp"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir"

; Function Attrs: noinline norecurse nounwind optnone
define i32 @main() #0 !dbg !8 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 0, !dbg !13
}

attributes #0 = { noinline norecurse nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!7}

; CHECK-LLVM: !DICompileUnit
; CHECK-LLVM-SAME: producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 16a50c9e642fd085e5ceb68c403b71b5b2e0607c)"
; CHECK-LLVM-NOT: producer: "spirv"
; CHECK-SPIRV: ModuleProcessed "Debug info producer: clang version 13.0.0 (https://github.com/llvm/llvm-project.git 16a50c9e642fd085e5ceb68c403b71b5b2e0607c)"

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 16a50c9e642fd085e5ceb68c403b71b5b2e0607c)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "oneAPI")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"ThinLTO", i32 0}
!6 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!7 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 16a50c9e642fd085e5ceb68c403b71b5b2e0607c)"}
!8 = distinct !DISubprogram(name: "main", scope: !9, file: !9, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DIFile(filename: "s.cpp", directory: "C:\\")
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 2, column: 2, scope: !8)
