// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL2.0 -O0 -debug-info-kind=standalone -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_unstructured_loop_controls -o %t.spv
// RUN: llvm-spirv %t.spv --to-text -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -r %t.spv -o %t.bc
// RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

// Test that no debug info instruction is inserted between LoopControlINTEL and
// Branch instructions. Otherwise, debug info interferes with SPIRVToLLVM
// translation of structured flow control. Yet, Line DebugInfo instruction is
// still presenting between LoopControlINTEL and Branch instructions.

kernel
void sample() {
  #pragma clang loop unroll(full)
  for(;;);
}

// CHECK-SPIRV: 2 LoopControlINTEL 1
// CHECK-SPIRV-NOT: ExtInst
// CHECK-SPIRV: {{[0-9]+}} Line {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
// CHECK-SPIRV: Branch
// CHECK-LLVM: br label %{{.*}}, !dbg !{{[0-9]+}}, !llvm.loop ![[MD:[0-9]+]]
// CHECK-LLVM: ![[MD]] = distinct !{![[MD]], ![[MD_unroll:[0-9]+]]}
// CHECK-LLVM: ![[MD_unroll]] = !{!"llvm.loop.unroll.enable"}
