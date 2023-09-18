; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_fast_composite
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.bc -r
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck < %t.ll %s --check-prefix=CHECK-LLVM

target triple = "spir64-unknown-unknown"


; CHECK-SPIRV: {{[0-9]+}} EntryPoint {{[0-9]+}} [[#FOO_ID:]] "foo"
; CHECK-SPIRV: {{[0-9]+}} EntryPoint {{[0-9]+}} [[#BAR_ID:]] "bar"
; CHECK-SPIRV: 3 ExecutionMode [[#FOO_ID]] 6088
; CHECK-SPIRV-NOT: 3 ExecutionMode [[#BAR_ID]] 6088

; CHECK-LLVM: define spir_kernel void @foo
; CHECK-LLVM-SAME: #[[#FOO_ATTR_ID:]]
; CHECK-LLVM: define spir_kernel void @bar
; CHECK-LLVM-SAME: #[[#BAR_ATTR_ID:]]

; CHECK-LLVM: attributes #[[#FOO_ATTR_ID]]
; CHECK-LLVM-SAME: "VCFCEntry"
; CHECK-LLVM: attributes #[[#BAR_ATTR_ID]]
; CHECK-LLVM-NOT: "VCFCEntry"


define spir_kernel void @foo(<4 x i32> %a, <4 x i32> %b) #0 {
entry:
  ret void
}

define spir_kernel void @bar(<4 x i32> %a, <4 x i32> %b) #1 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "VCFCEntry" "VCFloatControl"="0" "VCFunction" }
attributes #1 = { noinline nounwind "VCFloatControl"="48" "VCFunction" }

