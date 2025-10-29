///
/// Tests for using SPIR-V backend for SYCL offloading
///
// RUN: %clangxx -fsycl -fsycl-use-spirv-backend-for-spirv-gen -### %s 2>&1 | FileCheck %s

// CHECK: llc{{.*}} "-filetype=obj" "-mtriple=spirv64{{[^-]*}}-unknown-unknown" "--avoid-spirv-capabilities=Shader" "--translator-compatibility-mode" "-spirv-ext=+SPV_EXT_shader_atomic_float_add,+SPV_EXT_shader_atomic_float_min_max,+SPV_KHR_no_integer_wrap_decoration,+SPV_KHR_float_controls,+SPV_KHR_expect_assume,+SPV_KHR_linkonce_odr,+SPV_INTEL_subgroups,+SPV_INTEL_media_block_io,+SPV_INTEL_function_pointers,+SPV_INTEL_inline_assembly,+SPV_INTEL_arbitrary_precision_integers,+SPV_INTEL_float_controls2,+SPV_INTEL_variable_length_array,+SPV_INTEL_long_composites,+SPV_INTEL_cache_controls,+SPV_KHR_shader_clock,+SPV_INTEL_bindless_images,+SPV_INTEL_bfloat16_conversion,+SPV_INTEL_joint_matrix,+SPV_KHR_uniform_group_instructions,+SPV_INTEL_tensor_float32_conversion,+SPV_INTEL_optnone,+SPV_KHR_non_semantic_info,+SPV_KHR_cooperative_matrix,+SPV_EXT_shader_atomic_float16_add,+SPV_INTEL_fp_max_error"
