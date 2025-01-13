; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-preserve-auxdata --spirv-max-version=1.5 -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-EXT
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-preserve-auxdata
; RUN: llvm-spirv -r --spirv-preserve-auxdata %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc -spirv-text --spirv-preserve-auxdata -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NOEXT
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-preserve-auxdata
; RUN: llvm-spirv -r --spirv-preserve-auxdata %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; Check SPIR-V versions in a format magic number + version
; CHECK-SPIRV-EXT: 119734787 65536
; CHECK-SPIRV-EXT: Extension "SPV_KHR_non_semantic_info"
; CHECK-SPIRV-NOEXT: 119734787 67072

; CHECK-SPIRV: ExtInstImport [[#Import:]] "NonSemantic.AuxData"

; CHECK-SPIRV: String [[#Attr0:]] "nounwind"

; CHECK-SPIRV: Name [[#Fcn0:]] "foo"

; CHECK-SPIRV: TypeVoid [[#VoidT:]]

; CHECK-SPIRV: ExtInst [[#VoidT]] [[#Attr0Inst:]] [[#Import]] NonSemanticAuxDataFunctionAttribute [[#Fcn0]] [[#Attr0]] {{$}}

target triple = "spir64-unknown-unknown"

; CHECK-LLVM: define spir_func void @foo() #[[#Fcn0IRAttr:]]
define spir_func void @foo() #0 {
entry:
ret void
}
; CHECK-LLVM: attributes #[[#Fcn0IRAttr]] = { nounwind }
attributes #0 = { nounwind }
