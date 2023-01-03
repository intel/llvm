// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -r --spirv-target-env=CL2.0 %t.spv -o %t.rev.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-dis -opaque-pointers < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

kernel void testAtomicFlag(global int *res) {
  atomic_flag f;

  *res = atomic_flag_test_and_set(&f);
  *res += atomic_flag_test_and_set_explicit(&f, memory_order_seq_cst);
  *res += atomic_flag_test_and_set_explicit(&f, memory_order_seq_cst, memory_scope_work_group);

  atomic_flag_clear(&f);
  atomic_flag_clear_explicit(&f, memory_order_seq_cst);
  atomic_flag_clear_explicit(&f, memory_order_seq_cst, memory_scope_work_group);
}

// CHECK-SPIRV: AtomicFlagTestAndSet
// CHECK-SPIRV: AtomicFlagTestAndSet
// CHECK-SPIRV: AtomicFlagTestAndSet
// CHECK-SPIRV: AtomicFlagClear
// CHECK-SPIRV: AtomicFlagClear
// CHECK-SPIRV: AtomicFlagClear

// CHECK-LLVM-LABEL: @testAtomicFlag
// CHECK-LLVM: call spir_func i1 @_Z33atomic_flag_test_and_set_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(
// CHECK-LLVM: call spir_func i1 @_Z33atomic_flag_test_and_set_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(
// CHECK-LLVM: call spir_func i1 @_Z33atomic_flag_test_and_set_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(
// CHECK-LLVM: call spir_func void @_Z26atomic_flag_clear_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(
// CHECK-LLVM: call spir_func void @_Z26atomic_flag_clear_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(
// CHECK-LLVM: call spir_func void @_Z26atomic_flag_clear_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(
