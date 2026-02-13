; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-preserve-auxdata --spirv-max-version=1.5
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-EXT
; RUN: llvm-spirv -r --spirv-preserve-auxdata %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r %t.spv -o %t.rev.without.bc
; RUN: llvm-dis %t.rev.without.bc -o - | FileCheck %s --implicit-check-not="{{foo|bar|baz}}"

; RUN: llvm-spirv %t.bc -o %t.spv --spirv-preserve-auxdata
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NOEXT
; RUN: llvm-spirv -r --spirv-preserve-auxdata %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r %t.spv -o %t.rev.without.bc
; RUN: llvm-dis %t.rev.without.bc -o - | FileCheck %s --implicit-check-not="{{foo|bar|baz}}"

; Check SPIR-V versions in a format magic number + version
; CHECK-SPIRV-EXT: 119734787 65536
; CHECK-SPIRV-EXT: Extension "SPV_KHR_non_semantic_info"
; CHECK-SPIRV-NOEXT: 119734787 67072

; CHECK-SPIRV: ExtInstImport [[#Import:]] "NonSemantic.AuxData"

; CHECK-SPIRV: String [[#MDName:]] "absolute_symbol"

; CHECK-SPIRV: Name [[#GVName:]] "a"

; CHECK-SPIRV: TypeInt [[#Int32T:]] 64 0
; CHECK-SPIRV: Constant [[#Int32T]] [[#MDValue0:]] 0
; CHECK-SPIRV: Constant [[#Int32T]] [[#MDValue1:]] 16

; CHECK-SPIRV: TypeVoid [[#VoidT:]]

; CHECK-SPIRV: ExtInst [[#VoidT]] [[#ValInst:]] [[#Import]] NonSemanticAuxDataGlobalVariableMetadata [[#GVName]] [[#MDName]] [[#MDValue0]] [[#MDValue1]] {{$}}

target triple = "spir64-unknown-unknown"

; CHECK-LLVM: @a = external addrspace(1) global i8, !absolute_symbol ![[#LLVMVal:]]
@a = external addrspace(1) global i8, !absolute_symbol !0

; CHECK-LLVM: ![[#LLVMVal]] = !{i64 0, i64 16}
!0 = !{i64 0, i64 16}
