__kernel void test_fn( const __global char *src)
{
	wait_group_events(0, NULL);
}
// RUN: %clang_cc1 -triple spir64 -x cl -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -O0 -emit-llvm-bc %s -o %t.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.bc -spirv-text -o %t.spt
// RUN: FileCheck < %t.spt %s
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.bc -o %t.spv
// RUN: spirv-val %t.spv

// CHECK-NOT:Capability Groups
// CHECK:GroupWaitEvents
