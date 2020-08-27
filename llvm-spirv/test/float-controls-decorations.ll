; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_vector_compute,+SPV_INTEL_float_controls2
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.bc -r 
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck < %t.ll %s --check-prefix=CHECK-LLVM
target triple = "spir64"


define dso_local <4 x i32> @foo(<4 x i32> %a, <4 x i32> %b) #0 {
entry:
  ret <4 x i32> %a
}

define dso_local <4 x i32> @bar(<4 x i32> %a, <4 x i32> %b) #1 {
entry:
  ret <4 x i32> %b
}


; CHECK-LLVM: "VCFloatControl"="0"
; CHECK-LLVM: "VCFloatControl"="48"
; CHECK-SPIRV: 3 Name [[#FOO_ID:]] "foo"
; CHECK-SPIRV: 3 Name [[#BAR_ID:]] "bar"
; CHECK-SPIRV: 5 Decorate [[#DEC_DENORM_GROUP_ID:]] FunctionDenormModeINTEL 16 1
; CHECK-SPIRV: 5 Decorate [[#DEC_DENORM_GROUP_ID+1]] FunctionDenormModeINTEL 32 1
; CHECK-SPIRV: 5 Decorate [[#DEC_DENORM_GROUP_ID+2]] FunctionDenormModeINTEL 64 1
; CHECK-SPIRV: 5 Decorate [[#DEC_FLT_GROUP_ID:]] FunctionFloatingPointModeINTEL 16 0
; CHECK-SPIRV: 5 Decorate [[#DEC_FLT_GROUP_ID+1]] FunctionFloatingPointModeINTEL 32 0
; CHECK-SPIRV: 5 Decorate [[#DEC_FLT_GROUP_ID+2]] FunctionFloatingPointModeINTEL 64 0
; CHECK-SPIRV: 5 Decorate [[#FOO_ID]] FunctionRoundingModeINTEL 16 0
; CHECK-SPIRV-NEXT: 5 Decorate [[#BAR_ID]] FunctionRoundingModeINTEL 16 1
; CHECK-SPIRV-NEXT: 5 Decorate [[#FOO_ID]] FunctionRoundingModeINTEL 32 0
; CHECK-SPIRV-NEXT: 5 Decorate [[#BAR_ID]] FunctionRoundingModeINTEL 32 1
; CHECK-SPIRV-NEXT: 5 Decorate [[#FOO_ID]] FunctionRoundingModeINTEL 64 0
; CHECK-SPIRV-NEXT: 5 Decorate [[#BAR_ID]] FunctionRoundingModeINTEL 64 1
; CHECK-SPIRV: 4 GroupDecorate [[#DEC_DENORM_GROUP_ID]] [[#FOO_ID]] [[#BAR_ID]]
; CHECK-SPIRV-NEXT: 4 GroupDecorate [[#DEC_DENORM_GROUP_ID+1]] [[#FOO_ID]] [[#BAR_ID]]
; CHECK-SPIRV-NEXT: 4 GroupDecorate [[#DEC_DENORM_GROUP_ID+2]] [[#FOO_ID]] [[#BAR_ID]]
; CHECK-SPIRV-NEXT: 4 GroupDecorate [[#DEC_FLT_GROUP_ID]] [[#FOO_ID]] [[#BAR_ID]]
; CHECK-SPIRV-NEXT: 4 GroupDecorate [[#DEC_FLT_GROUP_ID+1]] [[#FOO_ID]] [[#BAR_ID]]
; CHECK-SPIRV-NEXT: 4 GroupDecorate [[#DEC_FLT_GROUP_ID+2]] [[#FOO_ID]] [[#BAR_ID]]

attributes #0 = { "VCFloatControl"="0" "VCFunction"  }
attributes #1 = { "VCFloatControl"="48" "VCFunction" }


