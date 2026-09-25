// REQUIRES: system-linux, x86-registered-target

// Test for default llvm-spirv options

// RUN: touch %t.o
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/ld \
// RUN:   -- -o /dev/null %t.o --dry-run 2>&1 | FileCheck %s

// CHECK: llvm-spirv{{.*}}-spirv-debug-info-version=nonsemantic-shader-200
// CHECK-SAME:-spirv-ext=-all
// CHECK-SAME:,+SPV_EXT_shader_atomic_float_add
// CHECK-SAME:,+SPV_EXT_shader_atomic_float_min_max
// CHECK-SAME:,+SPV_KHR_no_integer_wrap_decoration,+SPV_KHR_float_controls
// CHECK-SAME:,+SPV_KHR_expect_assume,+SPV_KHR_linkonce_odr
// CHECK-SAME:,+SPV_INTEL_subgroups,+SPV_INTEL_media_block_io
// CHECK-SAME:,+SPV_INTEL_device_side_avc_motion_estimation
// CHECK-SAME:,+SPV_INTEL_fpga_loop_controls
// CHECK-SAME:,+SPV_INTEL_unstructured_loop_controls,+SPV_INTEL_fpga_reg
// CHECK-SAME:,+SPV_INTEL_blocking_pipes,+SPV_INTEL_function_pointers
// CHECK-SAME:,+SPV_INTEL_kernel_attributes,+SPV_INTEL_io_pipes
// CHECK-SAME:,+SPV_INTEL_inline_assembly,+SPV_INTEL_arbitrary_precision_integers
// CHECK-SAME:,+SPV_INTEL_float_controls2
// CHECK-SAME:,+SPV_INTEL_vector_compute
// CHECK-SAME:,+SPV_INTEL_arbitrary_precision_fixed_point
// CHECK-SAME:,+SPV_INTEL_arbitrary_precision_floating_point
// CHECK-SAME:,+SPV_INTEL_variable_length_array,+SPV_INTEL_fp_fast_math_mode
// CHECK-SAME:,+SPV_INTEL_long_composites
// CHECK-SAME:,+SPV_INTEL_arithmetic_fence
// CHECK-SAME:,+SPV_INTEL_cache_controls
// CHECK-SAME:,+SPV_INTEL_fpga_buffer_location
// CHECK-SAME:,+SPV_INTEL_fpga_argument_interfaces
// CHECK-SAME:,+SPV_INTEL_fpga_invocation_pipelining_attributes
// CHECK-SAME:,+SPV_INTEL_fpga_latency_control
// CHECK-SAME:,+SPV_KHR_shader_clock
// CHECK-SAME:,+SPV_INTEL_bindless_images
// CHECK-SAME:,+SPV_INTEL_task_sequence
// CHECK-SAME:,+SPV_INTEL_bfloat16_conversion
// CHECK-SAME:,+SPV_INTEL_joint_matrix
// CHECK-SAME:,+SPV_INTEL_hw_thread_queries
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
