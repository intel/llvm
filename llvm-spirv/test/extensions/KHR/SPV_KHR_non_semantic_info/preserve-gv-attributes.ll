; RUN: llvm-as < %s -o %t.bc
; RUN: not llvm-spirv %t.bc -spirv-text --spirv-preserve-auxdata --spirv-max-version=1.5 --spirv-ext=-SPV_KHR_non_semantic_info,+SPV_INTEL_global_variable_decorations -o - 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-EXT-DISABLED
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-preserve-auxdata --spirv-max-version=1.5 --spirv-ext=+SPV_INTEL_global_variable_decorations
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-EXT
; RUN: llvm-spirv -r --spirv-preserve-auxdata %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r %t.spv -o %t.rev.without.bc
; RUN: llvm-dis %t.rev.without.bc -o - | FileCheck %s --implicit-check-not="{{foo|bar|baz}}"

; RUN: llvm-spirv %t.bc -spirv-text --spirv-preserve-auxdata --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_INTEL_global_variable_decorations -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NOEXT
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-preserve-auxdata --spirv-ext=+SPV_INTEL_global_variable_decorations
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

; CHECK-SPIRV: String [[#Attr0LHS:]] "sycl-device-global-size"
; CHECK-SPIRV: String [[#Attr0RHS:]] "32"
; CHECK-SPIRV: String [[#Attr1:]] "sycl-device-image-scope"
; CHECK-SPIRV: String [[#Attr2LHS:]] "sycl-host-access"
; CHECK-SPIRV: String [[#Attr2RHS:]] "0"
; CHECK-SPIRV: String [[#Attr3LHS:]] "sycl-unique-id"
; CHECK-SPIRV: String [[#Attr3RHS:]] "_Z20__AsanKernelMetadata"

; CHECK-SPIRV: Name [[#GVName:]] "__AsanKernelMetadata"

; CHECK-SPIRV: TypeVoid [[#VoidT:]]

; CHECK-SPIRV: ExtInst [[#VoidT]] [[#Attr0Inst:]] [[#Import]] NonSemanticAuxDataGlobalVariableAttribute [[#GVName]] [[#Attr0LHS]] [[#Attr0RHS]] {{$}}
; CHECK-SPIRV: ExtInst [[#VoidT]] [[#Attr1Inst:]] [[#Import]] NonSemanticAuxDataGlobalVariableAttribute [[#GVName]] [[#Attr1]] {{$}}
; CHECK-SPIRV: ExtInst [[#VoidT]] [[#Attr1Inst:]] [[#Import]] NonSemanticAuxDataGlobalVariableAttribute [[#GVName]] [[#Attr2LHS]] [[#Attr2RHS]] {{$}}
; CHECK-SPIRV: ExtInst [[#VoidT]] [[#Attr1Inst:]] [[#Import]] NonSemanticAuxDataGlobalVariableAttribute [[#GVName]] [[#Attr3LHS]] [[#Attr3RHS]] {{$}}

target triple = "spir64-unknown-unknown"

; CHECK-LLVM: @__AsanKernelMetadata = addrspace(1) global [1 x %structtype] [%structtype { i64 0, i64 92 }] #[[#GVIRAttr:]]
%structtype = type { i64, i64 }

@__AsanKernelMetadata = addrspace(1) global [1 x %structtype] [%structtype { i64 ptrtoint (ptr addrspace(2) null to i64), i64 92 }], !spirv.Decorations !0 #0

; CHECK-LLVM: attributes #[[#GVIRAttr]] = { "sycl-device-global-size"="32" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="_Z20__AsanKernelMetadata" }
attributes #0 = { "sycl-device-global-size"="32" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="_Z20__AsanKernelMetadata" }

!0 = !{!1}
!1 = !{i32 6147, i32 0, !"_Z20__AsanKernelMetadata"}

; CHECK-SPIRV-EXT-DISABLED: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-SPIRV-EXT-DISABLED-NEXT: SPV_KHR_non_semantic_info
