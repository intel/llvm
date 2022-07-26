; Source code:
; class A {};
; struct B {};

; int main() {
;   A a;
;   B b;
;   return 0;
; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; ModuleID = 'main.cpp'
source_filename = "main.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

%class.A = type { i8 }
%struct.B = type { i8 }

; Function Attrs: noinline norecurse nounwind optnone mustprogress
define dso_local i32 @main() #0 !dbg !6 {
entry:
  %retval = alloca i32, align 4
  %a = alloca %class.A, align 1
  %b = alloca %struct.B, align 1
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata %class.A* %a, metadata !11, metadata !DIExpression()), !dbg !13
  call void @llvm.dbg.declare(metadata %struct.B* %b, metadata !14, metadata !DIExpression()), !dbg !16
  ret i32 0, !dbg !17
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline norecurse nounwind optnone mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!5}

; CHECK-LLVM: !DICompositeType
; CHECK-LLVM-SAME: tag: DW_TAG_class_type
; CHECK-LLVM-SAME: name: "A"
; CHECK-LLVM: !DICompositeType
; CHECK-LLVM-SAME: tag: DW_TAG_structure_type
; CHECK-LLVM-SAME: name: "B"

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 16a50c9e642fd085e5ceb68c403b71b5b2e0607c)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "/export/users")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 16a50c9e642fd085e5ceb68c403b71b5b2e0607c)"}
!6 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 4, type: !8, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "main.cpp", directory: "/export/users")
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "a", scope: !6, file: !7, line: 5, type: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !7, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: "_ZTS1A")
!13 = !DILocation(line: 5, column: 7, scope: !6)
!14 = !DILocalVariable(name: "b", scope: !6, file: !7, line: 6, type: !15)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !7, line: 2, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: "_ZTS1B")
!16 = !DILocation(line: 6, column: 7, scope: !6)
!17 = !DILocation(line: 7, column: 5, scope: !6)
