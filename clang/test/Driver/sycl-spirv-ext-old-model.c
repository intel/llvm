/// Check llvm-spirv extensions that are set

// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64_gen-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64_gen-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64_x86_64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-CPU

// CHECK-DEFAULT: llvm-spirv{{.*}}"-spirv-ext=-all
// CHECK-DEFAULT-SAME:,+SPV_EXT_shader_atomic_float_add
// CHECK-DEFAULT-SAME:,+SPV_EXT_shader_atomic_float_min_max
// CHECK-DEFAULT-SAME:,+SPV_KHR_no_integer_wrap_decoration,+SPV_KHR_float_controls
// CHECK-DEFAULT-SAME:,+SPV_KHR_expect_assume,+SPV_KHR_linkonce_odr
// CHECK-DEFAULT-SAME:,+SPV_INTEL_subgroups,+SPV_INTEL_media_block_io
// CHECK-DEFAULT-SAME:,+SPV_INTEL_device_side_avc_motion_estimation
// CHECK-DEFAULT-SAME:,+SPV_INTEL_fpga_loop_controls
// CHECK-DEFAULT-SAME:,+SPV_INTEL_unstructured_loop_controls,+SPV_INTEL_fpga_reg
// CHECK-DEFAULT-SAME:,+SPV_INTEL_blocking_pipes,+SPV_INTEL_function_pointers
// CHECK-DEFAULT-SAME:,+SPV_INTEL_kernel_attributes,+SPV_INTEL_io_pipes
// CHECK-DEFAULT-SAME:,+SPV_INTEL_inline_assembly,+SPV_INTEL_arbitrary_precision_integers
// CHECK-DEFAULT-SAME:,+SPV_INTEL_float_controls2
// CHECK-DEFAULT-SAME:,+SPV_INTEL_vector_compute,+SPV_INTEL_fast_composite
// CHECK-DEFAULT-SAME:,+SPV_INTEL_arbitrary_precision_fixed_point
// CHECK-DEFAULT-SAME:,+SPV_INTEL_arbitrary_precision_floating_point
// CHECK-DEFAULT-SAME:,+SPV_INTEL_variable_length_array,+SPV_INTEL_fp_fast_math_mode
// CHECK-DEFAULT-SAME:,+SPV_INTEL_long_composites
// CHECK-DEFAULT-SAME:,+SPV_INTEL_arithmetic_fence
// CHECK-DEFAULT-SAME:,+SPV_INTEL_cache_controls
// CHECK-DEFAULT-SAME:,+SPV_INTEL_fpga_buffer_location
// CHECK-DEFAULT-SAME:,+SPV_INTEL_fpga_argument_interfaces
// CHECK-DEFAULT-SAME:,+SPV_INTEL_fpga_invocation_pipelining_attributes
// CHECK-DEFAULT-SAME:,+SPV_INTEL_fpga_latency_control
// CHECK-DEFAULT-SAME:,+SPV_KHR_shader_clock
// CHECK-DEFAULT-SAME:,+SPV_INTEL_bindless_images
// CHECK-DEFAULT-SAME:,+SPV_INTEL_task_sequence
// CHECK-DEFAULT-SAME:,+SPV_INTEL_bfloat16_conversion
// CHECK-DEFAULT-SAME:,+SPV_INTEL_joint_matrix
// CHECK-DEFAULT-SAME:,+SPV_INTEL_hw_thread_queries
// CHECK-DEFAULT-SAME:,+SPV_KHR_uniform_group_instructions
// CHECK-DEFAULT-SAME:,+SPV_INTEL_masked_gather_scatter
// CHECK-DEFAULT-SAME:,+SPV_INTEL_tensor_float32_conversion
// CHECK-DEFAULT-SAME:,+SPV_INTEL_optnone
// CHECK-DEFAULT-SAME:,+SPV_KHR_non_semantic_info
// CHECK-DEFAULT-SAME:,+SPV_KHR_cooperative_matrix
// CHECK-DEFAULT-SAME:,+SPV_EXT_shader_atomic_float16_add
// CHECK-DEFAULT-SAME:,+SPV_INTEL_fp_max_error
// CHECK-CPU: llvm-spirv{{.*}}"-spirv-allow-unknown-intrinsics=llvm.genx.,llvm.fpbuiltin"
// CHECK-CPU-SAME: {{.*}}"-spirv-ext=-all
// CHECK-CPU-SAME:,+SPV_EXT_shader_atomic_float_add
// CHECK-CPU-SAME:,+SPV_EXT_shader_atomic_float_min_max
// CHECK-CPU-SAME:,+SPV_KHR_no_integer_wrap_decoration,+SPV_KHR_float_controls
// CHECK-CPU-SAME:,+SPV_KHR_expect_assume,+SPV_KHR_linkonce_odr
// CHECK-CPU-SAME:,+SPV_INTEL_subgroups,+SPV_INTEL_media_block_io
// CHECK-CPU-SAME:,+SPV_INTEL_device_side_avc_motion_estimation
// CHECK-CPU-SAME:,+SPV_INTEL_fpga_loop_controls
// CHECK-CPU-SAME:,+SPV_INTEL_unstructured_loop_controls,+SPV_INTEL_fpga_reg
// CHECK-CPU-SAME:,+SPV_INTEL_blocking_pipes,+SPV_INTEL_function_pointers
// CHECK-CPU-SAME:,+SPV_INTEL_kernel_attributes,+SPV_INTEL_io_pipes
// CHECK-CPU-SAME:,+SPV_INTEL_inline_assembly,+SPV_INTEL_arbitrary_precision_integers
// CHECK-CPU-SAME:,+SPV_INTEL_float_controls2
// CHECK-CPU-SAME:,+SPV_INTEL_vector_compute,+SPV_INTEL_fast_composite
// CHECK-CPU-SAME:,+SPV_INTEL_arbitrary_precision_fixed_point
// CHECK-CPU-SAME:,+SPV_INTEL_arbitrary_precision_floating_point
// CHECK-CPU-SAME:,+SPV_INTEL_variable_length_array,+SPV_INTEL_fp_fast_math_mode
// CHECK-CPU-SAME:,+SPV_INTEL_long_composites
// CHECK-CPU-SAME:,+SPV_INTEL_arithmetic_fence
// CHECK-CPU-SAME:,+SPV_INTEL_cache_controls
// CHECK-CPU-SAME:,+SPV_INTEL_fpga_buffer_location
// CHECK-CPU-SAME:,+SPV_INTEL_fpga_argument_interfaces
// CHECK-CPU-SAME:,+SPV_INTEL_fpga_invocation_pipelining_attributes
// CHECK-CPU-SAME:,+SPV_INTEL_fpga_latency_control
// CHECK-CPU-SAME:,+SPV_INTEL_task_sequence
// CHECK-CPU-SAME:,+SPV_INTEL_bfloat16_conversion
// CHECK-CPU-SAME:,+SPV_INTEL_joint_matrix
// CHECK-CPU-SAME:,+SPV_INTEL_hw_thread_queries
// CHECK-CPU-SAME:,+SPV_KHR_uniform_group_instructions
// CHECK-CPU-SAME:,+SPV_INTEL_masked_gather_scatter
// CHECK-CPU-SAME:,+SPV_INTEL_tensor_float32_conversion
// CHECK-CPU-SAME:,+SPV_INTEL_optnone
// CHECK-CPU-SAME:,+SPV_KHR_non_semantic_info
// CHECK-CPU-SAME:,+SPV_KHR_cooperative_matrix
// CHECK-CPU-SAME:,+SPV_INTEL_fp_max_error
