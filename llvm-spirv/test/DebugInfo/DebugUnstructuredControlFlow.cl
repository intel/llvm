// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL2.0 -O0 -debug-info-kind=standalone -gno-column-info -emit-llvm-bc %s -o %t.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.bc --spirv-ext=+SPV_INTEL_unstructured_loop_controls -o %t.spv
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.spv --to-text -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers -r -emit-opaque-pointers %t.spv -o %t.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-dis -opaque-pointers < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

// Test that no debug info instruction is inserted between LoopControlINTEL and
// Branch instructions. Otherwise, debug info interferes with SPIRVToLLVM
// translation of structured flow control

kernel
void sample() {
  #pragma clang loop unroll(full)
  for(;;);
}

// Check that all Line items are retained
// CHECK-SPIRV: Line [[File:[0-9]+]] 15 0
// Loop control
// CHECK-SPIRV: LoopControlINTEL 257 1
// CHECK-SPIRV-NEXT: Branch

// CHECK-LLVM: br label %{{.*}}, !dbg !{{[0-9]+}}, !llvm.loop ![[MD:[0-9]+]]
// CHECK-LLVM: ![[MD]] = distinct !{![[MD]], ![[MD_unroll:[0-9]+]]}
// CHECK-LLVM: ![[MD_unroll]] = !{!"llvm.loop.unroll.full"}
