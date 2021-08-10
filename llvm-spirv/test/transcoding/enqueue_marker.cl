// RUN: %clang_cc1 -triple spir-unknown-unknown -fdeclare-opencl-builtins -finclude-default-header -O0 -cl-std=CL2.0 -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.spv.txt
// RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: llvm-spirv -r -spirv-target-env="SPV-IR" %t.spv -o %t.rev.bc
// RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-SPV-IR

// Check that SPIR-V friendly IR is correctly recognized
// RUN: llvm-spirv %t.rev.bc -spirv-text -o %t.spv.txt
// RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV

kernel void test_enqueue_marker(global int *out) {
  queue_t queue = get_default_queue();

  clk_event_t waitlist, evt;

  // CHECK-SPIRV: EnqueueMarker
  // CHECK-LLVM: _Z14enqueue_marker9ocl_queuejPU3AS4K12ocl_clkeventPU3AS4S0_
  // CHECK-SPV-IR: call spir_func %spirv.Queue* @_Z23__spirv_GetDefaultQueuev()
  // CHECK-SPV-IR: call spir_func i32 @_Z21__spirv_EnqueueMarkerP13__spirv_QueueiPU3AS4P19__spirv_DeviceEventPU3AS4P19__spirv_DeviceEvent(%spirv.Queue* %0, i32 1, %spirv.DeviceEvent* addrspace(4)* %waitlist.ascast, %spirv.DeviceEvent* addrspace(4)* %evt.ascast)
  *out = enqueue_marker(queue, 1, &waitlist, &evt);
}
