;; compiled from (null added manually):
;; template <auto Func>
;; int foo() {
;;   int result = Func();
;;   return result;
;; };
;;
;; long get() { return 42; }
;;
;; void boo() {
;;   int val = foo<get>();
;; }

; REQUIRES: object-emission

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll
; RUN: FileCheck < %t.ll %s --check-prefix=CHECK-LLVM

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %t.ll | llvm-dwarfdump -v -debug-info - | FileCheck %s --check-prefix=CHECK-DWARF

; CHECK-LLVM: ![[#]] = !DITemplateValueParameter(name: "Func", type: ![[#Type:]], value: ptr null)
; CHECK-LLVM: ![[#Type]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[#]], size: 64)

; CHECK-DWARF: DW_TAG_subprogram
; CHECK-DWARF: DW_AT_name{{.*}}"foo<&get>"

; CHECK-DWARF: DW_TAG_template_value_parameter
; CHECK-DWARF: DW_AT_type {{.*}} "int (*)()"
; CHECK-DWARF: DW_AT_name {{.*}} "Func"

; ModuleID = '/app/example.cpp'
source_filename = "/app/example.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

$_Z3fooIXadL_Z3getvEEEiv = comdat any

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z3getv() #0 !dbg !10 {
  ret i32 42, !dbg !16
}

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @_Z3boov() #1 !dbg !17 {
  %1 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %1, metadata !20, metadata !DIExpression()), !dbg !21
  %2 = call noundef i32 @_Z3fooIXadL_Z3getvEEEiv(), !dbg !22
  store i32 %2, ptr %1, align 4, !dbg !21
  ret void, !dbg !23
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef i32 @_Z3fooIXadL_Z3getvEEEiv() #0 comdat !dbg !24 {
  %1 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %1, metadata !28, metadata !DIExpression()), !dbg !29
  %2 = call noundef i32 @_Z3getv(), !dbg !30
  store i32 %2, ptr %1, align 4, !dbg !29
  %3 = load i32, ptr %1, align 4, !dbg !31
  ret i32 %3, !dbg !32
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project.git ef38880ce03bc1f1fb3606c5a629151f3d0e975e)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/app/example.cpp", directory: "/app")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git ef38880ce03bc1f1fb3606c5a629151f3d0e975e)"}
!10 = distinct !DISubprogram(name: "get", linkageName: "_Z3getv", scope: !11, file: !11, line: 7, type: !12, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!11 = !DIFile(filename: "example.cpp", directory: "/app")
!12 = !DISubroutineType(types: !13)
!13 = !{!14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{}
!16 = !DILocation(line: 7, column: 13, scope: !10)
!17 = distinct !DISubprogram(name: "boo", linkageName: "_Z3boov", scope: !11, file: !11, line: 9, type: !18, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !DILocalVariable(name: "val", scope: !17, file: !11, line: 10, type: !14)
!21 = !DILocation(line: 10, column: 9, scope: !17)
!22 = !DILocation(line: 10, column: 15, scope: !17)
!23 = !DILocation(line: 11, column: 1, scope: !17)
!24 = distinct !DISubprogram(name: "foo<&get>", linkageName: "_Z3fooIXadL_Z3getvEEEiv", scope: !11, file: !11, line: 2, type: !12, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, templateParams: !25, retainedNodes: !15)
!25 = !{!26}
!26 = !DITemplateValueParameter(name: "Func", type: !27, value: null)
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!28 = !DILocalVariable(name: "result", scope: !24, file: !11, line: 3, type: !14)
!29 = !DILocation(line: 3, column: 9, scope: !24)
!30 = !DILocation(line: 3, column: 18, scope: !24)
!31 = !DILocation(line: 4, column: 12, scope: !24)
!32 = !DILocation(line: 4, column: 5, scope: !24)
