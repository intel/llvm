// REQUIRES: system-linux, x86-registered-target, spirv-to-ir-wrapper, sycl-post-link

// Test for default llvm-spirv options

// RUN: %clang_cc1 -triple spir64-unknown-unknown -emit-llvm-bc -o %t.device.bc %s
// RUN: llvm-offload-binary -o %t.fat \
// RUN:   --image=file=%t.device.bc,kind=sycl,triple=spir64-unknown-unknown
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.fat %s
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/ld \
// RUN:   --dry-run -- -o /dev/null %t.o 2>&1 | FileCheck %s

// CHECK: llvm-spirv{{.*}}-spirv-debug-info-version=nonsemantic-shader-200
// CHECK-SAME:-spirv-ext=-all
// CHECK-SAME:,+SPV_EXT_shader_atomic_float_add
// CHECK-SAME:,+SPV_EXT_shader_atomic_float_min_max
// CHECK-SAME:,+SPV_KHR_no_integer_wrap_decoration,+SPV_KHR_float_controls
// CHECK-SAME:,+SPV_KHR_expect_assume,+SPV_KHR_linkonce_odr
// CHECK-SAME:,+SPV_INTEL_subgroups
// CHECK-SAME:,+SPV_INTEL_unstructured_loop_controls
// CHECK-SAME:,+SPV_INTEL_function_pointers,+SPV_INTEL_kernel_attributes
// CHECK-SAME:,+SPV_INTEL_inline_assembly
// CHECK-SAME:,+SPV_INTEL_arbitrary_precision_integers
// CHECK-SAME:,+SPV_INTEL_float_controls2,+SPV_INTEL_vector_compute
// CHECK-SAME:,+SPV_INTEL_variable_length_array,+SPV_INTEL_fp_fast_math_mode
// CHECK-SAME:,+SPV_INTEL_long_composites
// CHECK-SAME:,+SPV_EXT_arithmetic_fence
// CHECK-SAME:,+SPV_INTEL_global_variable_decorations
// CHECK-SAME:,+SPV_INTEL_cache_controls
// CHECK-SAME:,+SPV_KHR_shader_clock
// CHECK-SAME:,+SPV_INTEL_bindless_images
// CHECK-SAME:,+SPV_INTEL_bfloat16_conversion
// CHECK-SAME:,+SPV_INTEL_joint_matrix
// CHECK-SAME:,+SPV_KHR_uniform_group_instructions
// CHECK-SAME:,+SPV_INTEL_masked_gather_scatter
// CHECK-SAME:,+SPV_INTEL_tensor_float32_conversion
// CHECK-SAME:,+SPV_INTEL_optnone
// CHECK-SAME:,+SPV_KHR_non_semantic_info
// CHECK-SAME:,+SPV_KHR_cooperative_matrix
// CHECK-SAME:,+SPV_EXT_shader_atomic_float16_add
// CHECK-SAME:,+SPV_INTEL_fp_max_error
// CHECK-SAME:,+SPV_INTEL_memory_access_aliasing
// CHECK-SAME:,+SPV_INTEL_maximum_registers
// CHECK-NOT: ocl-100
