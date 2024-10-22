// This test is intended to ensure that we have no trackers marked as XFAIL
// without a tracker information added to a test.
//
// The format we check is:
// XFAIL: lit,features
// XFAIL-TRACKER: [GitHub issue URL|Internal tracker ID]
//
// GitHub issue URL format:
//     https://github.com/owner/repo/issues/12345
//
// Internal tracker ID format:
//     PROJECT-123456
//
// REQUIRES: linux
//
// Explanation of the command:
// - search for all "XFAIL" occurrences, display line with match and the next one
//   -I, --include to drop binary files and other unrelated files
// - in the result, search for "XFAIL" again, but invert the result - this
//   allows us to get the line *after* XFAIL
// - in those lines, check that XFAIL-TRACKER is present and correct. Once
//   again, invert the search to get all "bad" lines and save the test names in
//   the temp file
// - make a final count of how many ill-formatted directives there are and
//   verify that against the reference
// - ...and check if the list of improperly "xfailed" tests needs to be updated.
//
// RUN: grep -rI "XFAIL:" %S -A 1 --include=*.c --include=*.cpp \
// RUN:      --exclude=no-xfail-without-tracker.cpp --no-group-separator | \
// RUN: grep -v "XFAIL:" | \
// RUN: grep -Pv "XFAIL-TRACKER:\s+(?:https://github.com/[\w\d-]+/[\w\d-]+/issues/[\d]+)|(?:[\w]+-[\d]+)" > %t | \
// RUN: cat %t | wc -l | FileCheck %s --check-prefix NUMBER-OF-XFAIL-WITHOUT-TRACKER
// RUN: cat %t | FileCheck %s
//
// The number below is a number of tests which are *improperly* XFAIL-ed, i.e.
// we either don't have a tracker associated with a failure listed in those
// tests, or it is listed in a wrong format.
// Note: strictly speaking, that is not amount of files, but amount of XFAIL
// directives. If a test contains several XFAIL directives, some of them may be
// valid and other may not.
//
// That number *must not* increase. Any PR which causes this number to grow
// should be rejected and it should be updated to either keep the number as-is
// or have it reduced (preferrably, down to zero).
//
// If you see this test failed for your patch, it means that you either
// introduced XFAIL directive to a test improperly, or broke the format of an
// existing XFAIL-ed tests.
// Another possibility (and that is a good option) is that you updated some
// tests to match the required format and in that case you should just update
// (i.e. reduce) the number below.
//
// NUMBER-OF-XFAIL-WITHOUT-TRACKER: 160
//
// List of improperly "xfailed" tests.
// Remove the CHECK once the test has been propely "xfailed".
//
// CHECK-DAG: AOT/fpga-aoc-archive-split-per-kernel.cpp
// CHECK-DAG: AddressSanitizer/nullpointer/private_nullptr.cpp
// CHECK-DAG: Basic/accessor/accessor.cpp
// CHECK-DAG: Basic/aspects.cpp
// CHECK-DAG: Basic/buffer/reinterpret.cpp
// CHECK-DAG: Basic/built-ins.cpp
// CHECK-DAG: Basic/device_event.cpp
// CHECK-DAG: Basic/diagnostics/handler.cpp
// CHECK-DAG: Basic/fpga_tests/fpga_pipes_mixed_usage.cpp
// CHECK-DAG: Basic/image/srgba-read.cpp
// CHECK-DAG: Basic/max_linear_work_group_size_props.cpp
// CHECK-DAG: Basic/max_work_group_size_props.cpp
// CHECK-DAG: Basic/partition_supported.cpp
// CHECK-DAG: Basic/queue/queue.cpp
// CHECK-DAG: Basic/queue/release.cpp
// CHECK-DAG: Basic/span.cpp
// CHECK-DAG: Basic/stream/auto_flush.cpp
// CHECK-DAG: DeprecatedFeatures/queue_old_interop.cpp
// CHECK-DAG: DeprecatedFeatures/set_arg_interop.cpp
// CHECK-DAG: DeviceArchitecture/device_architecture_comparison_on_device_aot.cpp
// CHECK-DAG: DeviceCodeSplit/split-per-kernel.cpp
// CHECK-DAG: DeviceCodeSplit/split-per-source-main.cpp
// CHECK-DAG: DeviceLib/assert-windows.cpp
// CHECK-DAG: ESIMD/assert.cpp
// CHECK-DAG: ESIMD/hardware_dispatch.cpp
// CHECK-DAG: GroupAlgorithm/root_group.cpp
// CHECK-DAG: GroupLocalMemory/group_local_memory.cpp
// CHECK-DAG: GroupLocalMemory/no_early_opt.cpp
// CHECK-DAG: InlineAsm/asm_multiple_instructions.cpp
// CHECK-DAG: InvokeSimd/Feature/ImplicitSubgroup/invoke_simd_struct.cpp
// CHECK-DAG: InvokeSimd/Feature/invoke_simd_struct.cpp
// CHECK-DAG: InvokeSimd/Spec/ImplicitSubgroup/tuple.cpp
// CHECK-DAG: InvokeSimd/Spec/ImplicitSubgroup/tuple_return.cpp
// CHECK-DAG: InvokeSimd/Spec/ImplicitSubgroup/tuple_vadd.cpp
// CHECK-DAG: InvokeSimd/Spec/tuple.cpp
// CHECK-DAG: InvokeSimd/Spec/tuple_return.cpp
// CHECK-DAG: InvokeSimd/Spec/tuple_vadd.cpp
// CHECK-DAG: KernelAndProgram/kernel-bundle-merge-options.cpp
// CHECK-DAG: LLVMIntrinsicLowering/sub_byte_bitreverse.cpp
// CHECK-DAG: Matrix/joint_matrix_bf16_fill_k_cache_arg_dim.cpp
// CHECK-DAG: Matrix/joint_matrix_bf16_fill_k_cache_runtime_dim.cpp
// CHECK-DAG: Matrix/joint_matrix_bfloat16_colmajorA_colmajorB.cpp
// CHECK-DAG: Matrix/joint_matrix_colA_rowB_colC.cpp
// CHECK-DAG: Matrix/joint_matrix_int8_colmajorA_colmajorB.cpp
// CHECK-DAG: Matrix/SG32/joint_matrix_annotated_ptr.cpp
// CHECK-DAG: Matrix/SG32/joint_matrix_bfloat16_colmajorA_colmajorB.cpp
// CHECK-DAG: Matrix/SG32/joint_matrix_bfloat16_packedB.cpp
// CHECK-DAG: Matrix/SG32/joint_matrix_colA_rowB_colC.cpp
// CHECK-DAG: Matrix/SG32/joint_matrix_int8_colmajorA_colmajorB.cpp
// CHECK-DAG: Matrix/SG32/joint_matrix_int8_rowmajorA_rowmajorB.cpp
// CHECK-DAG: Matrix/SG32/joint_matrix_out_bounds.cpp
// CHECK-DAG: Matrix/SG32/joint_matrix_prefetch.cpp
// CHECK-DAG: Matrix/SG32/joint_matrix_rowmajorA_rowmajorB.cpp
// CHECK-DAG: Matrix/SG32/joint_matrix_unaligned_k.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/element_wise_abc.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/element_wise_all_ops.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/element_wise_all_ops_half.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/element_wise_all_ops_int8.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/element_wise_all_ops_int8_packed.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/element_wise_all_ops_scalar.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/element_wise_all_sizes.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/element_wise_ops.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/get_coord_float_matC.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/get_coord_int8_matA.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/get_coord_int8_matB.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_all_sizes.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_annotated_ptr.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_apply_bf16.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_apply_two_matrices.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_bf16_fill_k_cache.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_bf16_fill_k_cache_unroll.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_bfloat16_array.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_bfloat16_colmajorA_colmajorB.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_bfloat16.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_bfloat16_packedB.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_colA_rowB_colC.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_down_convert.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_half.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_int8_colmajorA_colmajorB.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_rowmajorA_rowmajorB.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_ss_int8.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_su_int8.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_us_int8.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/joint_matrix_uu_int8.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/element_wise_abc.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/element_wise_all_ops.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/element_wise_all_ops_half.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/element_wise_all_ops_int8.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/element_wise_all_ops_int8_packed.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/element_wise_all_sizes.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/element_wise_ops.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/get_coord_float_matC.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/get_coord_int8_matA.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/get_coord_int8_matB.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_all_sizes.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_annotated_ptr.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_annotated_ptr.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_apply_bf16.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_apply_two_matrices.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_bf16_fill_k_cache.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_bf16_fill_k_cache_unroll.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_bfloat16_array.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_bfloat16_colmajorA_colmajorB.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_bfloat16.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_bfloat16_packedB.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_colA_rowB_colC.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_down_convert.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_half.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_int8_colmajorA_colmajorB.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_int8_rowmajorA_rowmajorB.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_out_bounds.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_prefetch.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_rowmajorA_rowmajorB.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_rowmajorA_rowmajorB.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_ss_int8.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_su_int8.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_unaligned_k.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_us_int8.cpp
// CHECK-DAG: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_uu_int8.cpp
// CHECK-DAG: NewOffloadDriver/multisource.cpp
// CHECK-DAG: NewOffloadDriver/split-per-source-main.cpp
// CHECK-DAG: NewOffloadDriver/sycl-external-with-optional-features.cpp
// CHECK-DAG: OptionalKernelFeatures/throw-exception-for-out-of-registers-on-kernel-launch.cpp
// CHECK-DAG: PerformanceTests/Reduction/reduce_over_sub_group.cpp
// CHECK-DAG: Plugin/interop-cuda-experimental.cpp
// CHECK-DAG: Plugin/interop-experimental-single-TU-SYCL-CUDA-compilation.cpp
// CHECK-DAG: Plugin/level_zero_device_free_mem.cpp
// CHECK-DAG: Printf/int.cpp
// CHECK-DAG: Printf/mixed-address-space.cpp
// CHECK-DAG: Printf/percent-symbol.cpp
// CHECK-DAG: Reduction/reduction_big_data.cpp
// CHECK-DAG: Reduction/reduction_nd_conditional.cpp
// CHECK-DAG: Reduction/reduction_nd_dw.cpp
// CHECK-DAG: Reduction/reduction_nd_ext_double.cpp
// CHECK-DAG: Reduction/reduction_nd_ext_half.cpp
// CHECK-DAG: Reduction/reduction_nd_N_queue_shortcut.cpp
// CHECK-DAG: Reduction/reduction_nd_queue_shortcut.cpp
// CHECK-DAG: Reduction/reduction_nd_reducer_skip.cpp
// CHECK-DAG: Reduction/reduction_nd_rw.cpp
// CHECK-DAG: Reduction/reduction_range_queue_shortcut.cpp
// CHECK-DAG: Reduction/reduction_range_usm_dw.cpp
// CHECK-DAG: Reduction/reduction_reducer_op_eq.cpp
// CHECK-DAG: Reduction/reduction_span_pack.cpp
// CHECK-DAG: Reduction/reduction_usm.cpp
// CHECK-DAG: Reduction/reduction_usm_dw.cpp
// CHECK-DAG: Regression/build_log.cpp
// CHECK-DAG: Regression/complex_global_object.cpp
// CHECK-DAG: Regression/context_is_destroyed_after_exception.cpp
// CHECK-DAG: Regression/kernel_bundle_ignore_sycl_external.cpp
// CHECK-DAG: Regression/multiple-targets.cpp
// CHECK-DAG: Regression/reduction_resource_leak_dw.cpp
// CHECK-DAG: Scheduler/InOrderQueueDeps.cpp
// CHECK-DAG: Scheduler/MemObjRemapping.cpp
// CHECK-DAG: Scheduler/MultipleDevices.cpp
// CHECK-DAG: Scheduler/ReleaseResourcesTest.cpp
// CHECK-DAG: syclcompat/launch/launch_policy_lmem.cpp
// CHECK-DAG: Tracing/buffer_printers.cpp
// CHECK-DAG: VirtualFunctions/multiple-translation-units/separate-call.cpp
// CHECK-DAG: VirtualFunctions/multiple-translation-units/separate-vf-defs-and-call.cpp
// CHECK-DAG: VirtualFunctions/multiple-translation-units/separate-vf-defs.cpp
