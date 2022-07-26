; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=x86_64-pc-linux-gnu -split-dwarf-file=foo.dwo %t.ll -o - | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"
; Derived from:

; int main (void) {
;    return 0;
; }

; Check that we get a symbol off of the debug_info section when using split dwarf and pubnames.

; CHECK: .LpubTypes_start0:
; CHECK-NEXT: .short    2                       # DWARF Version
; CHECK-NEXT: .long     .Lcu_begin0             # Offset of Compilation Unit Info

; Function Attrs: nounwind uwtable
define i32 @main() #0 !dbg !4 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0, !dbg !10
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 (trunk 189287) (llvm/trunk 189296)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "foo.c", directory: "/usr/local/google/home/echristo/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "main", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "foo.c", directory: "/usr/local/google/home/echristo/tmp")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 3}
!10 = !DILocation(line: 2, scope: !4)
!11 = !{i32 1, !"Debug Info Version", i32 3}
