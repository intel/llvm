; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_attributes -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r -emit-opaque-pointers %t.spv --spirv-target-env=SPV-IR -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-SPV-IR

; RUN: llvm-spirv -r -emit-opaque-pointers %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; Expected to fail - the decorations require enabled extension to be translated.
; RUN: not llvm-spirv %t.bc -o %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@char_var = addrspace(1) global i8 0, !spirv.Decorations !0

; CHECK-SPIRV: Capability FPGAMemoryAttributesINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_fpga_memory_attributes"

; CHECK-SPIRV: Decorate [[#VAR_ID:]] RegisterINTEL 
; CHECK-SPIRV: Decorate [[#VAR_ID]] MemoryINTEL "DEFAULT" 
; CHECK-SPIRV: Decorate [[#VAR_ID]] NumbanksINTEL 4 
; CHECK-SPIRV: Decorate [[#VAR_ID]] MaxPrivateCopiesINTEL 3 
; CHECK-SPIRV: Decorate [[#VAR_ID]] SinglepumpINTEL 
; CHECK-SPIRV: Decorate [[#VAR_ID]] DoublepumpINTEL 
; CHECK-SPIRV: Decorate [[#VAR_ID]] MaxReplicatesINTEL 5 
; CHECK-SPIRV: Decorate [[#VAR_ID]] SimpleDualPortINTEL 
; CHECK-SPIRV: Decorate [[#VAR_ID]] ForcePow2DepthINTEL 0 

; 5 is a global storage
; CHECK-SPIRV: Variable [[#]] [[#VAR_ID]] 5

!0 = !{!1, !2, !3, !4, !5, !6, !7, !8, !9}
!1 = !{i32 5825}
!2 = !{i32 5826, [8 x i8] c"DEFAULT\00"}
!3 = !{i32 5827, i32 4}
!4 = !{i32 5829, i32 3}
!5 = !{i32 5830}
!6 = !{i32 5831}
!7 = !{i32 5832, i32 5}
!8 = !{i32 5833}
!9 = !{i32 5836, i1 false}
; !10 = !{i32 5883, i32 2}
; !11 = !{i32 5884, i32 8}
; !12 = !{i32 5885}

; CHECK-SPV-IR: @char_var = addrspace(1) global i8 0, !spirv.Decorations ![[#VAR_DEC:]]

; CHECK-SPV-IR: ![[#VAR_DEC]] = !{![[#]], ![[#RegisterINTEL:]], ![[#MemoryINTEL:]], ![[#NumbanksINTEL:]], ![[#MaxPrivateCopiesINTEL:]], ![[#SinglepumpINTEL:]], ![[#DoublepumpINTEL:]], ![[#MaxReplicatesINTEL:]], ![[#SimpleDualPortINTEL:]], ![[#ForcePow2DepthINTEL:]]}
; CHECK-SPV-IR: ![[#RegisterINTEL]] = !{i32 5825}
; CHECK-SPV-IR: ![[#MemoryINTEL]] = !{i32 5826, !"DEFAULT"}
; CHECK-SPV-IR: ![[#NumbanksINTEL]] = !{i32 5827, i32 4}
; CHECK-SPV-IR: ![[#MaxPrivateCopiesINTEL]] = !{i32 5829, i32 3}
; CHECK-SPV-IR: ![[#SinglepumpINTEL]] = !{i32 5830}
; CHECK-SPV-IR: ![[#DoublepumpINTEL]] = !{i32 5831}
; CHECK-SPV-IR: ![[#MaxReplicatesINTEL]] = !{i32 5832, i32 5}
; CHECK-SPV-IR: ![[#SimpleDualPortINTEL]] = !{i32 5833}
; CHECK-SPV-IR: ![[#ForcePow2DepthINTEL]] = !{i32 5836, i32 0}

; CHECK-LLVM-NOT: @char_var = {{.*}}, !spirv.Decorations ![[#]]
