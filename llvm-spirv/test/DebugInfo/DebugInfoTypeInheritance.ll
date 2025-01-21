; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck %s --input-file %t.spt --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-OCL
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll --check-prefix CHECK-LLVM

; RUN: llvm-spirv %t.bc -spirv-text --spirv-debug-info-version=nonsemantic-shader-100 -o %t.spt
; RUN: FileCheck %s --input-file %t.spt --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NONSEM
; RUN: llvm-spirv %t.bc --spirv-debug-info-version=nonsemantic-shader-100 -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll --check-prefix CHECK-LLVM

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt --experimental-debuginfo-iterators=1
; RUN: FileCheck %s --input-file %t.spt --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-OCL
; RUN: llvm-spirv %t.bc -o %t.spv --experimental-debuginfo-iterators=1
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc --experimental-debuginfo-iterators=1
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll --check-prefix CHECK-LLVM

; RUN: llvm-spirv %t.bc -spirv-text --spirv-debug-info-version=nonsemantic-shader-100 -o %t.spt --experimental-debuginfo-iterators=1
; RUN: FileCheck %s --input-file %t.spt --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NONSEM
; RUN: llvm-spirv %t.bc --spirv-debug-info-version=nonsemantic-shader-100 -o %t.spv --experimental-debuginfo-iterators=1
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc --experimental-debuginfo-iterators=1
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll --check-prefix CHECK-LLVM

; CHECK-SPIRV: String [[#Str_C:]] "C"
; CHECK-SPIRV: String [[#Str_B:]] "B"
; CHECK-SPIRV: String [[#Str_A:]] "A"

; CHECK-SPIRV: [[#Class_A:]] [[#]] DebugTypeComposite [[#Str_A]]

; CHECK-SPIRV-OCL: [[#B_inherits_A:]] [[#]] DebugTypeInheritance [[#Class_B:]] [[#Class_A]] [[#]] [[#]] [[#]] {{$}}
; CHECK-SPIRV-NONSEM: [[#B_inherits_A:]] [[#]] DebugTypeInheritance [[#Class_A]] [[#]] [[#]] [[#]] {{$}}
; CHECK-SPIRV: [[#Class_B:]] [[#]] DebugTypeComposite [[#Str_B]] {{.*}} [[#B_inherits_A]]

; CHECK-SPIRV-OCL: [[#C_inherits_B:]] [[#]] DebugTypeInheritance [[#Class_C:]] [[#Class_B]] [[#]] [[#]] [[#]] {{$}}
; CHECK-SPIRV-NONSEM: [[#C_inherits_B:]] [[#]] DebugTypeInheritance [[#Class_B]] [[#]] [[#]] [[#]] {{$}}
; CHECK-SPIRV: [[#Class_C:]] [[#]] DebugTypeComposite [[#Str_C]] {{.*}} [[#C_inherits_B]]

; CHECK-LLVM: ![[#Class_C:]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C"{{.*}}identifier: "_ZTS1C")
; CHECK-LLVM: !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[#Class_C]], baseType: ![[#Class_B:]]
; CHECK-LLVM: ![[#Class_B]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "B"{{.*}}identifier: "_ZTS1B")
; CHECK-LLVM: !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[#Class_B]], baseType: ![[#Class_A:]]
; CHECK-LLVM: ![[#Class_A]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A"{{.*}}identifier: "_ZTS1A")

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

%class.C = type { i8 }

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z3foov() #0 !dbg !10 {
  %1 = alloca %class.C, align 1
  call void @llvm.dbg.declare(metadata ptr %1, metadata !16, metadata !DIExpression()), !dbg !24
  ret i32 0, !dbg !25
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project.git 1f8a33c19c79fd4649a07eb70ea394c60a8ce316)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/app/example.cpp", directory: "/app")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 1f8a33c19c79fd4649a07eb70ea394c60a8ce316)"}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !11, file: !11, line: 4, type: !12, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!11 = !DIFile(filename: "example.cpp", directory: "/app")
!12 = !DISubroutineType(types: !13)
!13 = !{!14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{}
!16 = !DILocalVariable(name: "c", scope: !10, file: !11, line: 7, type: !17)
!17 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C", file: !11, line: 3, size: 8, flags: DIFlagTypePassByValue, elements: !18, identifier: "_ZTS1C")
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !17, baseType: !20, flags: DIFlagPublic, extraData: i32 0)
!20 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "B", file: !11, line: 2, size: 8, flags: DIFlagTypePassByValue, elements: !21, identifier: "_ZTS1B")
!21 = !{!22}
!22 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !20, baseType: !23, flags: DIFlagPublic, extraData: i32 0)
!23 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !11, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !15, identifier: "_ZTS1A")
!24 = !DILocation(line: 7, column: 11, scope: !10)
!25 = !DILocation(line: 8, column: 3, scope: !10)
