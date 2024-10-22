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
//   improperly-xfailed-tests.txt
// - make a final count of how many ill-formatted directives there are and
//   verify that against the reference
// - ...and check if the list of improperly "xfailed" tests needs to be updated.
//
// RUN: grep -rI "XFAIL:" %S -A 1 --exclude=%s --include=*.c --include=*.cpp \
// RUN:     --exclude=no-xfail-without-tracker.cpp --no-group-separator | \
// RUN: grep -v "XFAIL:" | \
// RUN: grep -Pv "XFAIL-TRACKER:\s+(?:https://github.com/[\w\d-]+/[\w\d-]+/issues/[\d]+)|(?:[\w]+-[\d]+)" > improperly-xfailed-tests.txt | \
// RUN: cat improperly-xfailed-tests.txt | wc -l | FileCheck %s --check-prefix NUMBER-OF-XFAIL-WITHOUT-TRACKER
// RUN: cat improperly-xfailed-tests.txt | FileCheck %s --check-prefix CHECK-TEST-NAME
// RUN: rm improperly-xfailed-tests.txt
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
// NUMBER-OF-XFAIL-WITHOUT-TRACKER: 162
//
// List of improperly "xfailed" tests.
// Remove the CHECK-TEST-NAME once the test has been propely "xfailed".
//
// CHECK-TEST-NAME: Basic/stream/auto_flush.cpp
// CHECK-TEST-NAME: Basic/accessor/accessor.cpp
// CHECK-TEST-NAME: Basic/max_linear_work_group_size_props.cpp
// CHECK-TEST-NAME: Basic/buffer/reinterpret.cpp
// CHECK-TEST-NAME: Basic/span.cpp
// CHECK-TEST-NAME: Basic/built-ins.cpp
// CHECK-TEST-NAME: Basic/fpga_tests/fpga_pipes_mixed_usage.cpp
// CHECK-TEST-NAME: Basic/max_work_group_size_props.cpp
// CHECK-TEST-NAME: Basic/image/srgba-read.cpp
// CHECK-TEST-NAME: Basic/queue/queue.cpp
// CHECK-TEST-NAME: Basic/queue/release.cpp
// CHECK-TEST-NAME: Basic/diagnostics/handler.cpp
// CHECK-TEST-NAME: Basic/device_event.cpp
// CHECK-TEST-NAME: Basic/aspects.cpp
// CHECK-TEST-NAME: Basic/partition_supported.cpp
// CHECK-TEST-NAME: Reduction/reduction_nd_ext_double.cpp
// CHECK-TEST-NAME: Reduction/reduction_range_usm_dw.cpp
// CHECK-TEST-NAME: Reduction/reduction_range_queue_shortcut.cpp
// CHECK-TEST-NAME: Reduction/reduction_span_pack.cpp
// CHECK-TEST-NAME: Reduction/reduction_reducer_op_eq.cpp
// CHECK-TEST-NAME: Reduction/reduction_nd_ext_half.cpp
// CHECK-TEST-NAME: Reduction/reduction_nd_rw.cpp
// CHECK-TEST-NAME: Reduction/reduction_nd_dw.cpp
// CHECK-TEST-NAME: Reduction/reduction_nd_N_queue_shortcut.cpp
// CHECK-TEST-NAME: Reduction/reduction_nd_queue_shortcut.cpp
// CHECK-TEST-NAME: Reduction/reduction_usm_dw.cpp
// CHECK-TEST-NAME: Reduction/reduction_nd_conditional.cpp
// CHECK-TEST-NAME: Reduction/reduction_usm.cpp
// CHECK-TEST-NAME: Reduction/reduction_big_data.cpp
// CHECK-TEST-NAME: Reduction/reduction_nd_reducer_skip.cpp
// CHECK-TEST-NAME: PerformanceTests/Reduction/reduce_over_sub_group.cpp
// CHECK-TEST-NAME: ESIMD/assert.cpp
// CHECK-TEST-NAME: ESIMD/hardware_dispatch.cpp
// CHECK-TEST-NAME: DeviceCodeSplit/split-per-kernel.cpp
// CHECK-TEST-NAME: DeviceCodeSplit/split-per-source-main.cpp
// CHECK-TEST-NAME: InlineAsm/asm_multiple_instructions.cpp
// CHECK-TEST-NAME: GroupLocalMemory/group_local_memory.cpp
// CHECK-TEST-NAME: GroupLocalMemory/no_early_opt.cpp
// CHECK-TEST-NAME: Plugin/interop-experimental-single-TU-SYCL-CUDA-compilation.cpp
// CHECK-TEST-NAME: Plugin/level_zero_device_free_mem.cpp
// CHECK-TEST-NAME: Plugin/interop-cuda-experimental.cpp
// CHECK-TEST-NAME: Printf/percent-symbol.cpp
// CHECK-TEST-NAME: Printf/int.cpp
// CHECK-TEST-NAME: Printf/mixed-address-space.cpp
// CHECK-TEST-NAME: AOT/fpga-aoc-archive-split-per-kernel.cpp
// CHECK-TEST-NAME: OptionalKernelFeatures/throw-exception-for-out-of-registers-on-kernel-launch.cpp
// CHECK-TEST-NAME: Matrix/joint_matrix_bf16_fill_k_cache_arg_dim.cpp
// CHECK-TEST-NAME: Matrix/SG32/joint_matrix_rowmajorA_rowmajorB.cpp
// CHECK-TEST-NAME: Matrix/SG32/joint_matrix_bfloat16_packedB.cpp
// CHECK-TEST-NAME: Matrix/SG32/joint_matrix_colA_rowB_colC.cpp
// CHECK-TEST-NAME: Matrix/SG32/joint_matrix_bfloat16_colmajorA_colmajorB.cpp
// CHECK-TEST-NAME: Matrix/SG32/joint_matrix_unaligned_k.cpp
// CHECK-TEST-NAME: Matrix/SG32/joint_matrix_int8_colmajorA_colmajorB.cpp
// CHECK-TEST-NAME: Matrix/SG32/joint_matrix_annotated_ptr.cpp
// CHECK-TEST-NAME: Matrix/SG32/joint_matrix_int8_rowmajorA_rowmajorB.cpp
// CHECK-TEST-NAME: Matrix/SG32/joint_matrix_prefetch.cpp
// CHECK-TEST-NAME: Matrix/SG32/joint_matrix_out_bounds.cpp
// CHECK-TEST-NAME: Matrix/joint_matrix_bf16_fill_k_cache_runtime_dim.cpp
// CHECK-TEST-NAME: Matrix/joint_matrix_colA_rowB_colC.cpp
// CHECK-TEST-NAME: Matrix/joint_matrix_bfloat16_colmajorA_colmajorB.cpp
// CHECK-TEST-NAME: Matrix/joint_matrix_int8_colmajorA_colmajorB.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/get_coord_int8_matA.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_rowmajorA_rowmajorB.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_down_convert.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_apply_bf16.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_uu_int8.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/element_wise_all_ops_half.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/element_wise_all_sizes.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_bfloat16_packedB.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_all_sizes.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_bfloat16.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/element_wise_all_ops_int8.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/get_coord_int8_matB.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/get_coord_int8_matA.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_rowmajorA_rowmajorB.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_rowmajorA_rowmajorB.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_down_convert.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_apply_bf16.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_uu_int8.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/element_wise_all_ops_half.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/element_wise_all_sizes.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_bfloat16_packedB.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_all_sizes.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_bfloat16.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/element_wise_all_ops_int8.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/get_coord_int8_matB.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/element_wise_all_ops.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_su_int8.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/element_wise_abc.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/element_wise_ops.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_colA_rowB_colC.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/get_coord_float_matC.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_apply_two_matrices.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_bfloat16_colmajorA_colmajorB.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_ss_int8.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_half.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_unaligned_k.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_bfloat16_array.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_int8_colmajorA_colmajorB.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_bf16_fill_k_cache_unroll.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_annotated_ptr.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_annotated_ptr.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_bf16_fill_k_cache.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_int8_rowmajorA_rowmajorB.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_prefetch.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_out_bounds.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/element_wise_all_ops_int8_packed.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/SG32/joint_matrix_us_int8.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/element_wise_all_ops.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_su_int8.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/element_wise_abc.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/element_wise_ops.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_colA_rowB_colC.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/get_coord_float_matC.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_apply_two_matrices.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_bfloat16_colmajorA_colmajorB.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_ss_int8.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_half.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_bfloat16_array.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_int8_colmajorA_colmajorB.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_bf16_fill_k_cache_unroll.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/element_wise_all_ops_scalar.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_annotated_ptr.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_bf16_fill_k_cache.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/element_wise_all_ops_int8_packed.cpp
// CHECK-TEST-NAME: Matrix/SPVCooperativeMatrix/joint_matrix_us_int8.cpp
// CHECK-TEST-NAME: LLVMIntrinsicLowering/sub_byte_bitreverse.cpp
// CHECK-TEST-NAME: NewOffloadDriver/sycl-external-with-optional-features.cpp
// CHECK-TEST-NAME: NewOffloadDriver/split-per-source-main.cpp
// CHECK-TEST-NAME: NewOffloadDriver/multisource.cpp
// CHECK-TEST-NAME: DeviceArchitecture/device_architecture_comparison_on_device_aot.cpp
// CHECK-TEST-NAME: Tracing/buffer_printers.cpp
// CHECK-TEST-NAME: KernelAndProgram/kernel-bundle-merge-options.cpp
// CHECK-TEST-NAME: syclcompat/launch/launch_policy_lmem.cpp
// CHECK-TEST-NAME: Scheduler/MultipleDevices.cpp
// CHECK-TEST-NAME: Scheduler/InOrderQueueDeps.cpp
// CHECK-TEST-NAME: Scheduler/ReleaseResourcesTest.cpp
// CHECK-TEST-NAME: Scheduler/MemObjRemapping.cpp
// CHECK-TEST-NAME: VirtualFunctions/multiple-translation-units/separate-call.cpp
// CHECK-TEST-NAME: VirtualFunctions/multiple-translation-units/separate-vf-defs-and-call.cpp
// CHECK-TEST-NAME: VirtualFunctions/multiple-translation-units/separate-vf-defs.cpp
// CHECK-TEST-NAME: AddressSanitizer/nullpointer/private_nullptr.cpp
// CHECK-TEST-NAME: Regression/reduction_resource_leak_dw.cpp
// CHECK-TEST-NAME: Regression/build_log.cpp
// CHECK-TEST-NAME: Regression/context_is_destroyed_after_exception.cpp
// CHECK-TEST-NAME: Regression/multiple-targets.cpp
// CHECK-TEST-NAME: Regression/kernel_bundle_ignore_sycl_external.cpp
// CHECK-TEST-NAME: Regression/complex_global_object.cpp
// CHECK-TEST-NAME: DeprecatedFeatures/queue_old_interop.cpp
// CHECK-TEST-NAME: DeprecatedFeatures/set_arg_interop.cpp
// CHECK-TEST-NAME: GroupAlgorithm/root_group.cpp
// CHECK-TEST-NAME: InvokeSimd/Feature/invoke_simd_struct.cpp
// CHECK-TEST-NAME: InvokeSimd/Feature/ImplicitSubgroup/invoke_simd_struct.cpp
// CHECK-TEST-NAME: InvokeSimd/Spec/ImplicitSubgroup/tuple.cpp
// CHECK-TEST-NAME: InvokeSimd/Spec/ImplicitSubgroup/tuple_return.cpp
// CHECK-TEST-NAME: InvokeSimd/Spec/ImplicitSubgroup/tuple_vadd.cpp
// CHECK-TEST-NAME: InvokeSimd/Spec/tuple.cpp
// CHECK-TEST-NAME: InvokeSimd/Spec/tuple_return.cpp
// CHECK-TEST-NAME: InvokeSimd/Spec/tuple_vadd.cpp
// CHECK-TEST-NAME: DeviceLib/assert-windows.cpp
