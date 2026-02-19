; This test exercises the case where the DICompileUnit has as !DIMacro without a DIMacroFile

; Test round-trip translation of debug macro information:
; LLVM IR -> SPIR-V -> LLVM IR

; RUN: llvm-spirv --spirv-debug-info-version=ocl-100 %s -o %t.spv
; RUN: spirv-val %t.spv

; RUN: spirv-dis %t.spv -o %t.spvasm
; RUN: FileCheck %s --input-file %t.spvasm --check-prefix CHECK-SPIRV-OCL

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll

; RUN: llvm-spirv --spirv-ext=+SPV_KHR_non_semantic_info --spirv-debug-info-version=nonsemantic-shader-100 %s -o %t.spv
; RUN: spirv-val %t.spv

; RUN: spirv-dis %t.spv -o %t.spvasm
; RUN: FileCheck %s --input-file %t.spvasm --check-prefix CHECK-SPIRV-NON-SEMANTIC-100

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll

; RUN: llvm-spirv --spirv-ext=+SPV_KHR_non_semantic_info --spirv-debug-info-version=nonsemantic-shader-200 %s -o %t.spv
; RUN: spirv-val %t.spv

; RUN: spirv-dis %t.spv -o %t.spvasm
; RUN: FileCheck %s --input-file %t.spvasm --check-prefix CHECK-SPIRV-NON-SEMANTIC-200

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll

; CHECK-NOT: !DIMacroFile
; CHECK-DAG: !DIMacro(type: DW_MACINFO_define, line: 1, name: "SIZE", value: "5")

; CHECK-SPIRV-OCL-DAG: %[[size_name:.*]] = OpString "SIZE"
; CHECK-SPIRV-OCL-DAG: %[[size_value:.*]] = OpString "5"
; CHECK-SPIRV-OCL-DAG: %[[#]] = OpExtInst %void %[[#]] DebugMacroDef %[[#]] 1 %[[size_name]] %[[size_value]]

; CHECK-SPIRV-NON-SEMANTIC-100-DAG: %[[type:.*]] = OpTypeInt 32 0
; CHECK-SPIRV-NON-SEMANTIC-100-DAG: %[[uint_1_reg:.*]] = OpConstant %[[type]] 1
; CHECK-SPIRV-NON-SEMANTIC-100-DAG: %[[size_name:.*]] = OpString "SIZE"
; CHECK-SPIRV-NON-SEMANTIC-100-DAG: %[[size_value:.*]] = OpString "5"
; CHECK-SPIRV-NON-SEMANTIC-100-DAG: %[[#]] = OpExtInst %void %[[#]] DebugMacroDef %[[#]] %[[uint_1_reg]] %[[size_name]] %[[size_value]]

; CHECK-SPIRV-NON-SEMANTIC-200-DAG: %[[type:.*]] = OpTypeInt 32 0
; CHECK-SPIRV-NON-SEMANTIC-200-DAG: %[[uint_1_reg:.*]] = OpConstant %[[type]] 1
; CHECK-SPIRV-NON-SEMANTIC-200-DAG: %[[size_name:.*]] = OpString "SIZE"
; CHECK-SPIRV-NON-SEMANTIC-200-DAG: %[[size_value:.*]] = OpString "5"
; CHECK-SPIRV-NON-SEMANTIC-200-DAG: %[[#]] = OpExtInst %void %[[#]] 32 %[[#]] %[[uint_1_reg]] %[[size_name]] %[[size_value]]

target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_func i32 @main() #0 {
entry:
  ret i32 0
}

attributes #0 = { nounwind }

!llvm.module.flags = !{!0, !1, !12}
!llvm.dbg.cu = !{!4}
!spirv.MemoryModel = !{!13}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!14}
!opencl.spir.version = !{!15}
!opencl.used.extensions = !{!16}
!opencl.used.optional.core.features = !{!16}
!spirv.Generator = !{!17}

!0 = !{i32 7, !"Dwarf Version", i32 0}
!1 = !{i32 2, !"Source Lang Literal", !2}
!2 = !{!3}
!3 = !{!4, i32 12}
!4 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !5, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, macros: !6)
!5 = !DIFile(filename: "def.c", directory: "/tmp")
!6 = !{!10}
!10 = !DIMacro(type: DW_MACINFO_define, line: 1, name: "SIZE", value: "5")
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 2, i32 2}
!14 = !{i32 4, i32 100000}
!15 = !{i32 1, i32 2}
!16 = !{}
!17 = !{i16 7, i16 0}

