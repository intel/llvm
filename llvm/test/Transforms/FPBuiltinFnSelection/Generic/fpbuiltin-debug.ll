; RUN: opt -fpbuiltin-fn-selection -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: @test_fdiv
; CHECK: %{{.*}} = call float @llvm.nvvm.div.approx.f(float %{{.*}}, float %{{.*}}){{.*}}, !dbg ![[#Loc1:]]
; CHECK: %{{.*}} = fdiv float %{{.*}}, %{{.*}}, !dbg ![[#Loc2:]]
; CHECK-DAG: [[#Loc1]] = !DILocation(line: 4, column: 1, scope: ![[#]])
; CHECK-DAG: [[#Loc2]] = !DILocation(line: 5, column: 1, scope: ![[#]])
define void @test_fdiv(float %d1, <2 x float> %v2d1,
                       float %d2, <2 x float> %v2d2) {
entry:
  %t0 = call float @llvm.fpbuiltin.fdiv.f32(float %d1, float %d2) #0, !dbg !16
  %t1 = call float @llvm.fpbuiltin.fdiv.f32(float %d1, float %d2) #1, !dbg !17
  ret void
}

declare float @llvm.fpbuiltin.fdiv.f32(float, float)

attributes #0 = { "fpbuiltin-max-error"="2.5" }
attributes #1 = { "fpbuiltin-max-error"="0.5" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "example.c", directory: "/home")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 21.0.0git (https://github.com/llvm/llvm-project.git"}
!10 = distinct !DISubprogram(name: "test_fdiv", scope: !11, file: !11, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!11 = !DIFile(filename: "example.c", directory: "/home")
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14, !14}
!14 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!15 = !{}
!16 = !DILocation(line: 4, column: 1, scope: !10)
!17 = !DILocation(line: 5, column: 1, scope: !10)
