; Test checks debug info of Checksumkind & Checksum is preserved from LLVM IR to
; SPIR-V and SPIR-V to LLVM IR translation.

; Original .cpp source:
;
;  int main() {
;    return 0;
;  }

; Command line:
; ./clang -cc1 -debug-info-kind=standalone -S -emit-llvm -triple spir -gcodeview -gcodeview-ghash main.cpp

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll --check-prefix CHECK-LLVM

; ModuleID = 'source.bc'
source_filename = "main.cpp"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir"

; Function Attrs: noinline norecurse nounwind optnone
define i32 @main() #0 !dbg !10 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 0, !dbg !15
}

attributes #0 = { noinline norecurse nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6, !7, !8}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!9}

; CHECK-LLVM: !DIFile(filename: "main.cpp"
; CHECK-LLVM-SAME: checksumkind: CSK_MD5, checksum: "7bb56387968a9caa6e9e35fff94eaf7b"
; CHECK-SPIRV: String [[#REG:]] "//__CSK_MD5:7bb56387968a9caa6e9e35fff94eaf7b"
; CHECK-SPIRV: DebugSource
; CHECK-SPIRV-SAME: [[#REG]]

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 7d09e1d7cf27ce781e83f9d388a7a3e1e6487ead)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "oneAPI", checksumkind: CSK_MD5, checksum: "7bb56387968a9caa6e9e35fff94eaf7b")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"CodeViewGHash", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 1, !"ThinLTO", i32 0}
!8 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!9 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 7d09e1d7cf27ce781e83f9d388a7a3e1e6487ead)"}
!10 = distinct !DISubprogram(name: "main", scope: !11, file: !11, line: 1, type: !12, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!11 = !DIFile(filename: "main.cpp", directory: "C:\\", checksumkind: CSK_MD5, checksum: "7bb56387968a9caa6e9e35fff94eaf7b")
!12 = !DISubroutineType(types: !13)
!13 = !{!14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 2, column: 3, scope: !10)