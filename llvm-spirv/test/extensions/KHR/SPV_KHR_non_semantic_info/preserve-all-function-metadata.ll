; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-preserve-auxdata -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-preserve-auxdata
; RUN: llvm-spirv -r -emit-opaque-pointers --spirv-preserve-auxdata %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r -emit-opaque-pointers %t.spv -o %t.rev.without.bc
; RUN: llvm-dis %t.rev.without.bc -o - | FileCheck %s --implicit-check-not="{{foo|bar|baz}}"

; CHECK-SPIRV: Extension "SPV_KHR_non_semantic_info"
; CHECK-SPIRV: ExtInstImport [[#Import:]] "NonSemantic.AuxData"

; CHECK-SPIRV: String [[#MD0Name:]] "foo"
; CHECK-SPIRV: String [[#MD1Name:]] "bar"
; CHECK-SPIRV: String [[#MD1Value:]] "baz"

; CHECK-SPIRV: Name [[#Fcn0:]] "test_val"
; CHECK-SPIRV: Name [[#Fcn1:]] "test_string"

; CHECK-SPIRV: TypeInt [[#Int32T:]] 32 0
; CHECK-SPIRV: Constant [[#Int32T]] [[#MD0Value:]] 5

; CHECK-SPIRV: TypeVoid [[#VoidT:]]

; CHECK-SPIRV: ExtInst [[#VoidT]] [[#ValInst:]] [[#Import]] NonSemanticAuxDataFunctionMetadata [[#Fcn0]] [[#MD0Name]] [[#MD0Value]] {{$}}
; CHECK-SPIRV: ExtInst [[#VoidT]] [[#StrInst:]] [[#Import]] NonSemanticAuxDataFunctionMetadata [[#Fcn1]] [[#MD1Name]] [[#MD1Value]] {{$}}

target triple = "spir64-unknown-unknown"

; CHECK-LLVM: define spir_func void @test_val() {{.*}} !foo ![[#LLVMVal:]]
define spir_func void @test_val() #1 !foo !1 {
ret void
}

; CHECK-LLVM: define spir_func void @test_string() {{.*}} !bar ![[#LLVMStr:]]
define spir_func void @test_string() #1 !bar !2 {
ret void
}

; CHECK-LLVM: ![[#LLVMVal]] = !{i32 5}
!1 = !{i32 5}
; CHECK-LLVM ![[#LLVMSTR]] = !{!"baz"}
!2 = !{!"baz"}
