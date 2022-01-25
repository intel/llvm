; RUN: llvm-as < %s > %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_unstructured_loop_controls --spirv-ext=+SPV_INTEL_fpga_loop_controls -o - -spirv-text | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_unstructured_loop_controls --spirv-ext=+SPV_INTEL_fpga_loop_controls -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: 2 Capability UnstructuredLoopControlsINTEL
; CHECK-SPIRV: 2 Capability FPGALoopControlsINTEL
; CHECK-SPIRV: 9 Extension "SPV_INTEL_fpga_loop_controls"
; CHECK-SPIRV: 11 Extension "SPV_INTEL_unstructured_loop_controls"
; CHECK-SPIRV: 3 Name [[FOO:[0-9]+]] "foo"
; CHECK-SPIRV: 4 Name [[ENTRY_1:[0-9]+]] "entry"
; CHECK-SPIRV: 5 Name [[FOR:[0-9]+]] "for.cond"
; CHECK-SPIRV: 3 Name [[BOO:[0-9]+]] "boo"
; CHECK-SPIRV: 4 Name [[ENTRY_2:[0-9]+]] "entry"
; CHECK-SPIRV: 5 Name [[WHILE:[0-9]+]] "while.body"

; CHECK-SPIRV: 5 Function 2 [[FOO]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: 2 Label [[ENTRY_1]]
; CHECK-SPIRV: 2 Branch [[FOR]]
; CHECK-SPIRV: 2 Label [[FOR]]
; Per SPIR-V spec extension INTEL/SPV_INTEL_fpga_loop_controls,
; LoopControlMaxConcurrencyINTELMask = 0x20000 (131072)
; CHECK-SPIRV: 3 LoopControlINTEL 131072 2
; CHECK-SPIRV-NEXT: 2 Branch [[FOR]]

; CHECK-SPIRV: 5 Function 2 [[BOO]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: 2 Label [[ENTRY_2]]
; CHECK-SPIRV: 2 Branch [[WHILE]]
; CHECK-SPIRV: 2 Label [[WHILE]]
; Per SPIR-V spec extension INTEL/SPV_INTEL_fpga_loop_controls,
; LoopControlInitiationIntervalINTELMask = 0x10000 (65536)
; CHECK-SPIRV: 3 LoopControlINTEL 65536 2
; CHECK-SPIRV-NEXT: 2 Branch [[WHILE]]

; ModuleID = 'infinite.cl'
source_filename = "infinite.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: norecurse noreturn nounwind readnone
define spir_kernel void @foo() local_unnamed_addr #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  br label %for.cond, !llvm.loop !3
; CHECK-LLVM: define spir_kernel void @foo()
; CHECK-LLVM: br label %for.cond, !llvm.loop ![[MD_1:[0-9]+]]
}

; Function Attrs: norecurse noreturn nounwind readnone
define spir_kernel void @boo() local_unnamed_addr #0 {
entry:
  br label %while.body

while.body:                                       ; preds = %entry, %while.body
  br label %while.body, !llvm.loop !5
; CHECK-LLVM: define spir_kernel void @boo()
; CHECK-LLVM: br label %while.body, !llvm.loop ![[MD_2:[0-9]+]]
}

attributes #0 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{!"clang version 9.0.0"}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.max_concurrency.count", i32 2}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.ii.count", i32 2}

; CHECK-LLVM: ![[MD_1]] = distinct !{![[MD_1]], ![[LOOP_MD_1:[0-9]+]]}
; CHECK-LLVM: ![[LOOP_MD_1]] = !{!"llvm.loop.max_concurrency.count", i32 2}
; CHECK-LLVM: ![[MD_2]] = distinct !{![[MD_2]], ![[LOOP_MD_2:[0-9]+]]}
; CHECK-LLVM: ![[LOOP_MD_2]] = !{!"llvm.loop.ii.count", i32 2}
