// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown -Xssimulation %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -Xssimulation %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown -Xsemulator %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -Xsemulator %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT

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
// CHECK-DEFAULT-SAME:,+SPV_INTEL_long_constant_composite
// CHECK-DEFAULT-SAME:,+SPV_INTEL_arithmetic_fence
// CHECK-DEFAULT-SAME:,+SPV_INTEL_token_type
// CHECK-DEFAULT-SAME:,+SPV_INTEL_bfloat16_conversion
// CHECK-DEFAULT-SAME:,+SPV_INTEL_joint_matrix
// CHECK-DEFAULT-SAME:,+SPV_INTEL_hw_thread_queries"
// CHECK-FPGA-HW: llvm-spirv{{.*}}"-spirv-ext=-all
// CHECK-FPGA-HW-SAME:,+SPV_EXT_shader_atomic_float_add
// CHECK-FPGA-HW-SAME:,+SPV_EXT_shader_atomic_float_min_max
// CHECK-FPGA-HW-SAME:,+SPV_KHR_no_integer_wrap_decoration,+SPV_KHR_float_controls
// CHECK-FPGA-HW-SAME:,+SPV_KHR_expect_assume
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_subgroups,+SPV_INTEL_media_block_io
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_device_side_avc_motion_estimation
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_fpga_loop_controls
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_unstructured_loop_controls,+SPV_INTEL_fpga_reg
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_blocking_pipes,+SPV_INTEL_function_pointers
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_kernel_attributes,+SPV_INTEL_io_pipes
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_inline_assembly,+SPV_INTEL_arbitrary_precision_integers
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_float_controls2
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_vector_compute,+SPV_INTEL_fast_composite
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_arbitrary_precision_fixed_point
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_arbitrary_precision_floating_point
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_variable_length_array,+SPV_INTEL_fp_fast_math_mode
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_long_constant_composite
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_arithmetic_fence
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_usm_storage_classes
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_runtime_aligned
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_fpga_cluster_attributes,+SPV_INTEL_loop_fuse
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_fpga_buffer_location
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_fpga_invocation_pipelining_attributes
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_fpga_dsp_control
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_fpga_memory_accesses
// CHECK-FPGA-HW-SAME:,+SPV_INTEL_fpga_memory_attributes"
