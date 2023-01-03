// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL2.0 -O0 -debug-info-kind=standalone -gno-column-info -emit-llvm %s -o %t.ll
// RUN: llvm-as %t.ll -o %t.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.bc -spirv-text -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.bc -o %t.spv
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers -r -emit-opaque-pointers %t.spv -o %t.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-dis -opaque-pointers < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

// Test that no debug info instruction is inserted
// between LoopMerge and Branch/BranchConditional instructions.
// Otherwise, debug info interferes with SPIRVToLLVM translation
// of structured flow control

kernel
void sample() {
  int arr[10];
  #pragma clang loop unroll(enable)
  for (int i = 0; i < 10; i++)
    arr[i] = 0;
  int j = 0;
  #pragma clang loop unroll(enable)
  do {
    arr[j] = 0;
  } while (j++ < 10);
}

// Check that all Line items are retained
// FIXME: Due to shift in lines, DILocation line moved 18 -> 30
// CHECK-SPIRV: Line [[File:[0-9]+]] 30 0
// Control flow
// CHECK-SPIRV: {{[0-9]+}} LoopMerge [[MergeBlock:[0-9]+]] [[ContinueTarget:[0-9]+]] 1
// CHECK-SPIRV-NEXT: BranchConditional

// Check that all Line items are retained
// FIXME: Due to shift in lines, DILocation line moved 23 -> 35 and 23 -> 36
// CHECK-SPIRV: Line [[File]] 35 0
// CHECK-SPIRV: Line [[File]] 36 0
// Control flow
// CHECK-SPIRV: {{[0-9]+}} LoopMerge [[MergeBlock:[0-9]+]] [[ContinueTarget:[0-9]+]] 1
// CHECK-SPIRV-NEXT: Branch

// CHECK-LLVM: br label %{{.*}}, !dbg !{{[0-9]+}}, !llvm.loop ![[MD:[0-9]+]]
// CHECK-LLVM: ![[MD]] = distinct !{![[MD]], ![[MD_unroll:[0-9]+]]}
// CHECK-LLVM: ![[MD_unroll]] = !{!"llvm.loop.unroll.enable"}
