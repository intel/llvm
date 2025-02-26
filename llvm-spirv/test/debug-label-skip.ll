; RUN: llvm-as --preserve-input-debuginfo-format %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv

source_filename = "debug-label-bitcode.c"

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

; Function Attrs: noinline nounwind optnone
define i32 @foo(i32 signext %a, i32 signext %b) !dbg !4 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %sum = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  br label %top

top:                                              ; preds = %entry
  call void @llvm.dbg.label(metadata !9), !dbg !10
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %0, %1
  store i32 %add, ptr %sum, align 4
  br label %done

done:                                             ; preds = %top
  call void @llvm.dbg.label(metadata !11), !dbg !12
  %2 = load i32, ptr %sum, align 4
  ret i32 %2
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.label(metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang 6.0.0", isOptimized: false, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "debug-label-bitcode.c", directory: "./")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !5)
!5 = !{!9}
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8, !8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DILabel(scope: !4, name: "top", file: !1, line: 4)
!10 = !DILocation(line: 4, column: 1, scope: !4)
!11 = !DILabel(scope: !4, name: "done", file: !1, line: 7)
!12 = !DILocation(line: 7, column: 1, scope: !4)
