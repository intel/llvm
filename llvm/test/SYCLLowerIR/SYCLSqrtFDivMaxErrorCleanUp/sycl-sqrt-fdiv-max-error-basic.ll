; Test checks if @llvm.fpbuiltin.fdiv and @llvm.fpbuiltin.sqrt are removed from
; the module.

; RUN: opt -passes=sycl-sqrt-fdiv-max-error-clean-up < %s -S | FileCheck %s

; CHECK-NOT: llvm.fpbuiltin.fdiv.f32
; CHECK-NOT: llvm.fpbuiltin.sqrt.f32
; CHECK-NOT: fpbuiltin-max-error

; CHECK: test_fp_max_error_decoration(float [[F1:[%0-9a-z.]+]], float [[F2:[%0-9a-z.]+]])
; CHECK: [[V1:[%0-9a-z.]+]] = fdiv float [[F1]], [[F2]]
; CHECK: call float @llvm.sqrt.f32(float [[V1]])

; CHECK: test_fp_max_error_decoration_fast(float [[F1:[%0-9a-z.]+]], float [[F2:[%0-9a-z.]+]])
; CHECK: [[V1:[%0-9a-z.]+]] = fdiv fast float [[F1]], [[F2]]
; CHECK: call fast float @llvm.sqrt.f32(float [[V1]])

; CHECK: test_fp_max_error_decoration_debug(float [[F1:[%0-9a-z.]+]], float [[F2:[%0-9a-z.]+]])
; CHECK: [[V1:[%0-9a-z.]+]] = fdiv float [[F1]], [[F2]], !dbg ![[#Loc1:]]
; CHECK: call float @llvm.sqrt.f32(float [[V1]]), !dbg ![[#Loc2:]]

; CHECK: [[#Loc1]] = !DILocation(line: 1, column: 1, scope: ![[#]])
; CHECK: [[#Loc2]] = !DILocation(line: 2, column: 1, scope: ![[#]])

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define void @test_fp_max_error_decoration(float %f1, float %f2) {
entry:
  %v1 = call float @llvm.fpbuiltin.fdiv.f32(float %f1, float %f2) #0
  %v2 = call float @llvm.fpbuiltin.sqrt.f32(float %v1) #1
  ret void
}

define void @test_fp_max_error_decoration_fast(float %f1, float %f2) {
entry:
  %v1 = call fast float @llvm.fpbuiltin.fdiv.f32(float %f1, float %f2) #0
  %v2 = call fast float @llvm.fpbuiltin.sqrt.f32(float %v1) #1
  ret void
}

define void @test_fp_max_error_decoration_debug(float %f1, float %f2) {
entry:
  %v1 = call float @llvm.fpbuiltin.fdiv.f32(float %f1, float %f2) #0, !dbg !7
  %v2 = call float @llvm.fpbuiltin.sqrt.f32(float %v1) #1, !dbg !8
  ret void
}

declare float @llvm.fpbuiltin.fdiv.f32(float, float)

declare float @llvm.fpbuiltin.sqrt.f32(float)

attributes #0 = { "fpbuiltin-max-error"="2.5" }
attributes #1 = { "fpbuiltin-max-error"="3.0" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "2a034da6937f5b9cf6dd2d89127f57fd")
!2 = distinct !DISubprogram(name: "test_fp_max_error_decoration_debug", scope: !1, file: !1, line: 1, type: !3, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!3 = !DISubroutineType(types: !4)
!4 = !{!5, !6, !6}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!7 = !DILocation(line: 1, column: 1, scope: !2)
!8 = !DILocation(line: 2, column: 1, scope: !2)
!9 = !{i32 2, !"Debug Info Version", i32 3}
