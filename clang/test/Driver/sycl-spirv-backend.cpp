///
/// Tests for using SPIR-V backend for SYCL offloading
///
// RUN: %clangxx -fsycl -fsycl-use-spirv-backend-for-spirv-gen -### %s 2>&1 | FileCheck %s

// CHECK: llc{{.*}} "-filetype=obj" "-mtriple=spirv64{{[^-]*}}-unknown-unknown" "--avoid-spirv-capabilities=Shader" "--translator-compatibility-mode" "-spirv-ext=
// CHECK-SAME: +SPV_EXT_relaxed_printf_string_address_space
// CHECK-SAME:,+SPV_EXT_shader_atomic_float16_add
// CHECK-SAME:,+SPV_EXT_shader_atomic_float_add
// CHECK-SAME:,+SPV_EXT_shader_atomic_float_min_max
// CHECK-SAME:,+SPV_INTEL_2d_block_io
// CHECK-SAME:,+SPV_INTEL_arbitrary_precision_integers
// CHECK-SAME:,+SPV_INTEL_bfloat16_conversion
// CHECK-SAME:,+SPV_INTEL_bindless_images
// CHECK-SAME:,+SPV_INTEL_cache_controls
// CHECK-SAME:,+SPV_INTEL_float_controls2
// CHECK-SAME:,+SPV_INTEL_fp_max_error
// CHECK-SAME:,+SPV_INTEL_function_pointers
// CHECK-SAME:,+SPV_INTEL_inline_assembly
// CHECK-SAME:,+SPV_INTEL_joint_matrix
// CHECK-SAME:,+SPV_INTEL_long_composites
// CHECK-SAME:,+SPV_INTEL_subgroups
// CHECK-SAME:,+SPV_INTEL_tensor_float32_conversion
// CHECK-SAME:,+SPV_INTEL_variable_length_array
// CHECK-SAME:,+SPV_KHR_16bit_storage
// CHECK-SAME:,+SPV_KHR_cooperative_matrix
// CHECK-SAME:,+SPV_KHR_expect_assume
// CHECK-SAME:,+SPV_KHR_float_controls
// CHECK-SAME:,+SPV_KHR_linkonce_odr
// CHECK-SAME:,+SPV_KHR_no_integer_wrap_decoration
// CHECK-SAME:,+SPV_KHR_non_semantic_info
// CHECK-SAME:,+SPV_KHR_shader_clock
// CHECK-SAME:,+SPV_KHR_uniform_group_instructions"
