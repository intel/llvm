; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-preserve-auxdata -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-preserve-auxdata
; RUN: llvm-spirv -r --spirv-preserve-auxdata %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Extension "SPV_KHR_non_semantic_info"
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
