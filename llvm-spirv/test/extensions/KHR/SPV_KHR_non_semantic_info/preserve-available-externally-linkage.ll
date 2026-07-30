; RUN: llvm-spirv %s -spirv-text --spirv-preserve-auxdata --spirv-max-version=1.5 -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-EXT
; RUN: llvm-spirv %s -o %t.spv --spirv-preserve-auxdata --spirv-max-version=1.5
; RUN: llvm-spirv -r --spirv-preserve-auxdata %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %s -spirv-text --spirv-preserve-auxdata -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NOEXT
; RUN: llvm-spirv %s -o %t.spv --spirv-preserve-auxdata
; RUN: llvm-spirv -r --spirv-preserve-auxdata %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; Check SPIR-V versions in a format magic number + version
; CHECK-SPIRV-EXT: 119734787 65536
; CHECK-SPIRV-EXT: Extension "SPV_KHR_non_semantic_info"
; CHECK-SPIRV-NOEXT: 119734787 67072

; CHECK-SPIRV: ExtInstImport [[#Import:]] "NonSemantic.AuxData"
; CHECK-SPIRV: Name [[#GV:]] "extern_gv"
; CHECK-SPIRV: Name [[#Fn:]] "inlinable"
; CHECK-SPIRV: TypeInt [[#I32T:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#I32T]] [[#LinkageVal:]] 0
; CHECK-SPIRV: TypeVoid [[#VoidT:]]

; CHECK-SPIRV-DAG: ExtInst [[#VoidT]] [[#]] [[#Import]] NonSemanticAuxDataLinkage [[#GV]] [[#LinkageVal]] {{$}}
; CHECK-SPIRV-DAG: ExtInst [[#VoidT]] [[#]] [[#Import]] NonSemanticAuxDataLinkage [[#Fn]] [[#LinkageVal]] {{$}}

target triple = "spir64-unknown-unknown"

; CHECK-LLVM: @extern_gv = available_externally addrspace(1) global i32 42
@extern_gv = available_externally addrspace(1) global i32 42

; CHECK-LLVM: define available_externally spir_func i32 @inlinable(i32 %x)
define available_externally spir_func i32 @inlinable(i32 %x) {
  %r = add i32 %x, 1
  ret i32 %r
}
