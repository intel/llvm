; RUN: llc --debugify-and-strip-all-safe=0 -mtriple=arm64-- -O0 -debug-pass=Structure < %s -o /dev/null 2>&1 | \
; RUN:     grep -v "Verify generated machine code" | FileCheck %s

; REQUIRES: asserts

; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: Target Pass Configuration
; CHECK-NEXT: Machine Module Information
; CHECK-NEXT: Target Transform Information
; CHECK-NEXT: Create Garbage Collector Module Metadata
; CHECK-NEXT: Profile summary info
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-NEXT: Machine Branch Probability Analysis
; CHECK-NEXT:   ModulePass Manager
; CHECK-NEXT:     Pre-ISel Intrinsic Lowering
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Expand large div/rem
; CHECK-NEXT:       Expand fp
; CHECK-NEXT:       Expand Atomic instructions
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:       Lower Garbage Collection Instructions
; CHECK-NEXT:       Shadow Stack GC Lowering
; CHECK-NEXT:       Remove unreachable blocks from the CFG
; CHECK-NEXT:       Instrument function entry/exit with calls to e.g. mcount() (post inlining)
; CHECK-NEXT:       Scalarize Masked Memory Intrinsics
; CHECK-NEXT:       Expand reduction intrinsics
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Optimization Remark Emitter
; CHECK-NEXT:       AArch64 Stack Tagging
; CHECK-NEXT:       SME ABI Pass
; CHECK-NEXT:       FPBuiltin Function Selection
; CHECK-NEXT:       Exception handling preparation
; CHECK-NEXT:       Prepare callbr
; CHECK-NEXT:       Safe Stack instrumentation pass
; CHECK-NEXT:       Insert stack protectors
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:       Analysis containing CSE Info
; CHECK-NEXT:       IRTranslator
; CHECK-NEXT:       Analysis for ComputingKnownBits
; CHECK-NEXT:       AArch64O0PreLegalizerCombiner
; CHECK-NEXT:       Localizer
; CHECK-NEXT:       Analysis containing CSE Info
; CHECK-NEXT:       Analysis for ComputingKnownBits
; CHECK-NEXT:       Legalizer
; CHECK-NEXT:       AArch64PostLegalizerLowering
; CHECK-NEXT:       RegBankSelect
; CHECK-NEXT:       Analysis for ComputingKnownBits
; CHECK-NEXT:       InstructionSelect
; CHECK-NEXT:       ResetMachineFunction
; CHECK-NEXT:       Assignment Tracking Analysis
; CHECK-NEXT:       AArch64 Instruction Selection
; CHECK-NEXT:       Finalize ISel and expand pseudo-instructions
; CHECK-NEXT:       Local Stack Slot Allocation
; CHECK-NEXT:       Eliminate PHI nodes for register allocation
; CHECK-NEXT:       Two-Address instruction pass
; CHECK-NEXT:       Fast Register Allocator
; CHECK-NEXT:       Remove Redundant DEBUG_VALUE analysis
; CHECK-NEXT:       Fixup Statepoint Caller Saved
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       Prologue/Epilogue Insertion & Frame Finalization
; CHECK-NEXT:       Post-RA pseudo instruction expansion pass
; CHECK-NEXT:       AArch64 pseudo instruction expansion pass
; CHECK-NEXT:       Insert KCFI indirect call checks
; CHECK-NEXT:       AArch64 speculation hardening pass
; CHECK-NEXT:       Analyze Machine Code For Garbage Collection
; CHECK-NEXT:       Insert fentry calls
; CHECK-NEXT:       Insert XRay ops
; CHECK-NEXT:       Implement the 'patchable-function' attribute
; CHECK-NEXT:       Workaround A53 erratum 835769 pass
; CHECK-NEXT:       Contiguously Lay Out Funclets
; CHECK-NEXT:       Remove Loads Into Fake Uses
; CHECK-NEXT:       StackMap Liveness Analysis
; CHECK-NEXT:       Live DEBUG_VALUE analysis
; CHECK-NEXT:       Machine Sanitizer Binary Metadata
; CHECK-NEXT:       AArch64 sls hardening pass
; CHECK-NEXT:       AArch64 Pointer Authentication
; CHECK-NEXT:       AArch64 Branch Targets
; CHECK-NEXT:       Branch relaxation pass
; CHECK-NEXT:       Insert CFI remember/restore state instructions
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       Stack Frame Layout Analysis
; CHECK-NEXT:       Unpack machine instruction bundles
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       AArch64 Assembly Printer
; CHECK-NEXT:       Free MachineFunction

define void @f() {
  ret void
}
