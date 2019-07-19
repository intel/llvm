// RUN: %clang_cc1 -triple spir-unknown-unknown -finclude-default-header -O0 -cl-std=CL2.0 -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.spv.txt
// RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

kernel void test_enqueue_marker(global int *out) {
  queue_t queue = get_default_queue();

  clk_event_t waitlist, evt;

  // CHECK-SPIRV: EnqueueMarker
  // CHECK-LLVM: _Z14enqueue_marker9ocl_queuejPK12ocl_clkeventPS0_
  *out = enqueue_marker(queue, 1, &waitlist, &evt);
}
