// This test is intended to ensure that we have no tests marked as
// UNSUPPORTED without an information added to a test.
// For more info see: sycl/test-e2e/README.md
//
// The format we check is:
// UNSUPPORTED: lit,features
// UNSUPPORTED-TRACKER: [GitHub issue URL|Internal tracker ID]
// *OR*
// UNSUPPORTED: lit,features
// UNSUPPORTED-INTENDED: explanation why the test isn't intended to be run with this feature
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
// - search for all "UNSUPPORTED" occurrences, display line with match and the next one
//   -I, --include to drop binary files and other unrelated files
// - in the result, search for "UNSUPPORTED" again, but invert the result - this
//   allows us to get the line *after* UNSUPPORTED
// - in those lines, check that UNSUPPORTED-TRACKER or UNSUPPORTED-INTENDED is
//   present and correct. Once again, invert the search to get all "bad" lines
//   and save the test names in the temp file
// - make a final count of how many ill-formatted directives there are and
//   verify that against the reference
// - ...and check if the list of improperly UNSUPPORTED tests needs to be updated.
//
// RUN: grep -rI "UNSUPPORTED:" %S/../../test-e2e \
// RUN: -A 1 --include=*.cpp --no-group-separator | \
// RUN: grep -v "UNSUPPORTED:" | \
// RUN: grep -Pv "(?:UNSUPPORTED-TRACKER:\s+(?:(?:https:\/\/github.com\/[\w\d-]+\/[\w\d-]+\/issues\/[\d]+)|(?:[\w]+-[\d]+)))|(?:UNSUPPORTED-INTENDED:\s*.+)" > %t
// RUN: cat %t | wc -l | FileCheck %s --check-prefix NUMBER-OF-UNSUPPORTED-WITHOUT-INFO
// RUN: cat %t | sed 's/\.cpp.*/.cpp/' | sort | FileCheck %s
//
// The number below is a number of tests which are *improperly* UNSUPPORTED, i.e.
// we either don't have a tracker associated with a failure listed in those
// tests, or it is listed in a wrong format.
// Note: strictly speaking, that is not amount of files, but amount of UNSUPPORTED
// directives. If a test contains several UNSUPPORTED directives, some of them may be
// valid and other may not.
//
// That number *must not* increase. Any PR which causes this number to grow
// should be rejected and it should be updated to either keep the number as-is
// or have it reduced (preferably, down to zero).
//
// If you see this test failed for your patch, it means that you either
// introduced UNSUPPORTED directive to a test improperly, or broke the format of an
// existing UNSUPPORTED tests.
// Another possibility (and that is a good option) is that you updated some
// tests to match the required format and in that case you should just update
// (i.e. reduce) the number and the list below.
//
// NUMBER-OF-UNSUPPORTED-WITHOUT-INFO: 178
//
// List of improperly UNSUPPORTED tests.
// Remove the CHECK once the test has been properly UNSUPPORTED.
//
// CHECK: Adapters/enqueue-arg-order-image.cpp
// CHECK-NEXT: Adapters/level_zero/batch_event_status.cpp
// CHECK-NEXT: Adapters/level_zero/batch_test.cpp
// CHECK-NEXT: Adapters/level_zero/batch_test_copy_with_compute.cpp
// CHECK-NEXT: Adapters/level_zero/device_scope_events.cpp
// CHECK-NEXT: Adapters/level_zero/dynamic_batch_test.cpp
// CHECK-NEXT: Adapters/level_zero/imm_cmdlist_per_thread.cpp
// CHECK-NEXT: Adapters/level_zero/interop-buffer-ownership.cpp
// CHECK-NEXT: Adapters/level_zero/interop-buffer.cpp
// CHECK-NEXT: Adapters/level_zero/interop-direct.cpp
// CHECK-NEXT: Adapters/level_zero/interop-image-get-native-mem.cpp
// CHECK-NEXT: Adapters/level_zero/interop-image-ownership.cpp
// CHECK-NEXT: Adapters/level_zero/interop-image-ownership.cpp
// CHECK-NEXT: Adapters/level_zero/interop-image.cpp
// CHECK-NEXT: Adapters/level_zero/interop.cpp
// CHECK-NEXT: Adapters/level_zero/queue_profiling.cpp
// CHECK-NEXT: Adapters/level_zero/usm_device_read_only.cpp
// CHECK-NEXT: Adapters/max_malloc.cpp
// CHECK-NEXT: AddressCast/dynamic_address_cast.cpp
// CHECK-NEXT: AmdNvidiaJIT/kernel_and_bundle.cpp
// CHECK-NEXT: Assert/assert_in_simultaneous_kernels.cpp
// CHECK-NEXT: Assert/assert_in_simultaneously_multiple_tus.cpp
// CHECK-NEXT: Assert/check_resource_leak.cpp
// CHECK-NEXT: Assert/check_resource_leak.cpp
// CHECK-NEXT: Basic/buffer/buffer_create.cpp
// CHECK-NEXT: Basic/build_log.cpp
// CHECK-NEXT: Basic/gpu_max_wgs_error.cpp
// CHECK-NEXT: Basic/host-task-dependency.cpp
// CHECK-NEXT: Basic/image/image_accessor_range.cpp
// CHECK-NEXT: Basic/kernel_info_attr.cpp
// CHECK-NEXT: Basic/submit_time.cpp
// CHECK-NEXT: DeprecatedFeatures/DiscardEvents/discard_events_using_assert.cpp
// CHECK-NEXT: DeviceLib/built-ins/printf.cpp
// CHECK-NEXT: DeviceLib/cmath-aot.cpp
// CHECK-NEXT: DeviceLib/separate_compile_test.cpp
// CHECK-NEXT: ESIMD/PerformanceTests/BitonicSortK.cpp
// CHECK-NEXT: ESIMD/PerformanceTests/BitonicSortKv2.cpp
// CHECK-NEXT: ESIMD/PerformanceTests/Stencil.cpp
// CHECK-NEXT: ESIMD/PerformanceTests/matrix_transpose.cpp
// CHECK-NEXT: ESIMD/PerformanceTests/stencil2.cpp
// CHECK-NEXT: ESIMD/api/bin_and_cmp_ops_heavy.cpp
// CHECK-NEXT: ESIMD/api/replicate_smoke.cpp
// CHECK-NEXT: ESIMD/api/simd_copy_to_from.cpp
// CHECK-NEXT: ESIMD/api/simd_copy_to_from_stateful.cpp
// CHECK-NEXT: ESIMD/api/simd_subscript_operator.cpp
// CHECK-NEXT: ESIMD/api/simd_view_subscript_operator.cpp
// CHECK-NEXT: ESIMD/api/svm_gather_scatter.cpp
// CHECK-NEXT: ESIMD/api/svm_gather_scatter_scalar_off.cpp
// CHECK-NEXT: ESIMD/api/unary_ops_heavy.cpp
// CHECK-NEXT: ESIMD/assert.cpp
// CHECK-NEXT: ESIMD/ext_math.cpp
// CHECK-NEXT: ESIMD/ext_math_fast.cpp
// CHECK-NEXT: ESIMD/ext_math_saturate.cpp
// CHECK-NEXT: ESIMD/fp_in_phi.cpp
// CHECK-NEXT: ESIMD/lsc/lsc_store_2d_u16.cpp
// CHECK-NEXT: ESIMD/lsc/lsc_usm_store_u8_u16.cpp
// CHECK-NEXT: ESIMD/lsc/lsc_usm_store_u8_u16_64.cpp
// CHECK-NEXT: ESIMD/matrix_transpose2.cpp
// CHECK-NEXT: ESIMD/private_memory/private_memory.cpp
// CHECK-NEXT: ESIMD/regression/bitreverse.cpp
// CHECK-NEXT: ESIMD/regression/copyto_char_test.cpp
// CHECK-NEXT: ESIMD/regression/variable_gather_mask.cpp
// CHECK-NEXT: ESIMD/slm_init_no_inline.cpp
// CHECK-NEXT: Graph/Explicit/buffer_copy_host2target.cpp
// CHECK-NEXT: Graph/Explicit/buffer_copy_host2target_2d.cpp
// CHECK-NEXT: Graph/Explicit/buffer_copy_host2target_offset.cpp
// CHECK-NEXT: Graph/Explicit/buffer_copy_target2host.cpp
// CHECK-NEXT: Graph/Explicit/buffer_copy_target2host_2d.cpp
// CHECK-NEXT: Graph/Explicit/buffer_copy_target2host_offset.cpp
// CHECK-NEXT: Graph/Explicit/host_task2_multiple_roots.cpp
// CHECK-NEXT: Graph/Explicit/host_task_multiple_roots.cpp
// CHECK-NEXT: Graph/Explicit/interop-level-zero-launch-kernel.cpp
// CHECK-NEXT: Graph/Explicit/memadvise.cpp
// CHECK-NEXT: Graph/Explicit/prefetch.cpp
// CHECK-NEXT: Graph/Explicit/spec_constants_kernel_bundle_api.cpp
// CHECK-NEXT: Graph/Explicit/work_group_size_prop.cpp
// CHECK-NEXT: Graph/RecordReplay/buffer_copy_host2target.cpp
// CHECK-NEXT: Graph/RecordReplay/buffer_copy_host2target_2d.cpp
// CHECK-NEXT: Graph/RecordReplay/buffer_copy_host2target_offset.cpp
// CHECK-NEXT: Graph/RecordReplay/buffer_copy_target2host.cpp
// CHECK-NEXT: Graph/RecordReplay/buffer_copy_target2host_2d.cpp
// CHECK-NEXT: Graph/RecordReplay/buffer_copy_target2host_offset.cpp
// CHECK-NEXT: Graph/RecordReplay/host_task2_multiple_roots.cpp
// CHECK-NEXT: Graph/RecordReplay/host_task_multiple_roots.cpp
// CHECK-NEXT: Graph/RecordReplay/interop-level-zero-launch-kernel.cpp
// CHECK-NEXT: Graph/RecordReplay/memadvise.cpp
// CHECK-NEXT: Graph/RecordReplay/prefetch.cpp
// CHECK-NEXT: Graph/RecordReplay/spec_constants_kernel_bundle_api.cpp
// CHECK-NEXT: Graph/RecordReplay/work_group_size_prop.cpp
// CHECK-NEXT: Graph/UnsupportedDevice/device_query.cpp
// CHECK-NEXT: GroupAlgorithm/root_group.cpp
// CHECK-NEXT: HierPar/hier_par_wgscope.cpp
// CHECK-NEXT: InvokeSimd/Feature/ImplicitSubgroup/SPMD_invoke_ESIMD_external.cpp
// CHECK-NEXT: InvokeSimd/Feature/ImplicitSubgroup/popcnt.cpp
// CHECK-NEXT: InvokeSimd/Feature/popcnt.cpp
// CHECK-NEXT: InvokeSimd/Regression/ImplicitSubgroup/call_vadd_1d_spill.cpp
// CHECK-NEXT: InvokeSimd/Regression/call_vadd_1d_spill.cpp
// CHECK-NEXT: KernelAndProgram/cache-build-result.cpp
// CHECK-NEXT: KernelAndProgram/kernel-bundle-merge-options-env.cpp
// CHECK-NEXT: KernelAndProgram/kernel-bundle-merge-options.cpp
// CHECK-NEXT: KernelAndProgram/level-zero-static-link-flow.cpp
// CHECK-NEXT: KernelAndProgram/multiple-kernel-linking.cpp
// CHECK-NEXT: KernelAndProgram/spec_constants_after_link.cpp
// CHECK-NEXT: KernelAndProgram/spec_constants_after_link.cpp
// CHECK-NEXT: Matrix/SG32/element_wise_abc.cpp
// CHECK-NEXT: Matrix/SG32/element_wise_all_ops.cpp
// CHECK-NEXT: Matrix/SG32/element_wise_all_ops_half.cpp
// CHECK-NEXT: Matrix/SG32/element_wise_all_ops_int8.cpp
// CHECK-NEXT: Matrix/SG32/element_wise_all_ops_int8_packed.cpp
// CHECK-NEXT: Matrix/SG32/element_wise_all_sizes.cpp
// CHECK-NEXT: Matrix/SG32/element_wise_ops.cpp
// CHECK-NEXT: Matrix/SG32/get_coordinate_ops.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_all_sizes.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_apply_bf16.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_apply_two_matrices.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_bf16_fill_k_cache.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_bf16_fill_k_cache_SLM.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_bf16_fill_k_cache_init.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_bf16_fill_k_cache_unroll.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_bf16_fill_k_cache_unroll_init.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_bfloat16_array.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_down_convert.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_int8_rowmajorA_rowmajorB.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_prefetch.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_rowmajorA_rowmajorB.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_ss_int8.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_su_int8.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_transposeC.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_us_int8.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_uu_int8.cpp
// CHECK-NEXT: Matrix/joint_matrix_annotated_ptr.cpp
// CHECK-NEXT: Matrix/joint_matrix_bf16_fill_k_cache_OOB.cpp
// CHECK-NEXT: Matrix/joint_matrix_down_convert.cpp
// CHECK-NEXT: Matrix/joint_matrix_rowmajorA_rowmajorB.cpp
// CHECK-NEXT: OptionalKernelFeatures/large-reqd-work-group-size.cpp
// CHECK-NEXT: OptionalKernelFeatures/no-fp64-optimization-declared-aspects.cpp
// CHECK-NEXT: Printf/char.cpp
// CHECK-NEXT: Printf/double.cpp
// CHECK-NEXT: Printf/float.cpp
// CHECK-NEXT: Printf/int.cpp
// CHECK-NEXT: Printf/mixed-address-space.cpp
// CHECK-NEXT: Printf/percent-symbol.cpp
// CHECK-NEXT: ProfilingTag/in_order_profiling_queue.cpp
// CHECK-NEXT: ProfilingTag/profiling_queue.cpp
// CHECK-NEXT: Regression/barrier_waitlist_with_interop_event.cpp
// CHECK-NEXT: Regression/complex_global_object.cpp
// CHECK-NEXT: Regression/event_destruction.cpp
// CHECK-NEXT: Regression/invalid_reqd_wg_size_correct_exception.cpp
// CHECK-NEXT: Regression/kernel_bundle_ignore_sycl_external.cpp
// CHECK-NEXT: Regression/no-split-reqd-wg-size-2.cpp
// CHECK-NEXT: Regression/no-split-reqd-wg-size.cpp
// CHECK-NEXT: Regression/reduction_resource_leak_usm.cpp
// CHECK-NEXT: Regression/static-buffer-dtor.cpp
// CHECK-NEXT: Scheduler/InOrderQueueDeps.cpp
// CHECK-NEXT: SpecConstants/2020/kernel-bundle-api.cpp
// CHECK-NEXT: SpecConstants/2020/non_native/gpu.cpp
// CHECK-NEXT: SubGroup/generic_reduce.cpp
// CHECK-NEXT: Tracing/code_location_queue_copy.cpp
// CHECK-NEXT: Tracing/code_location_queue_parallel_for.cpp
// CHECK-NEXT: Tracing/code_location_queue_submit.cpp
// CHECK-NEXT: Tracing/task_execution.cpp
// CHECK-NEXT: Tracing/task_execution_handler.cpp
// CHECK-NEXT: Tracing/usm/queue_copy_released_pointer.cpp
// CHECK-NEXT: Tracing/usm/queue_single_task_nullptr.cpp
// CHECK-NEXT: Tracing/usm/queue_single_task_released_pointer.cpp
// CHECK-NEXT: USM/badmalloc.cpp
// CHECK-NEXT: USM/memops2d/copy2d_device_to_host.cpp
// CHECK-NEXT: USM/memops2d/copy2d_host_to_device.cpp
// CHECK-NEXT: USM/memops2d/memcpy2d_device_to_host.cpp
// CHECK-NEXT: USM/memops2d/memcpy2d_host_to_device.cpp
// CHECK-NEXT: USM/pointer_query_descendent_device.cpp
// CHECK-NEXT: syclcompat/atomic/atomic_arith.cpp
// CHECK-NEXT: syclcompat/atomic/atomic_bitwise.cpp
// CHECK-NEXT: syclcompat/atomic/atomic_class.cpp
// CHECK-NEXT: syclcompat/atomic/atomic_comp_exchange.cpp
// CHECK-NEXT: syclcompat/atomic/atomic_memory_acq_rel.cpp
// CHECK-NEXT: syclcompat/atomic/atomic_minmax.cpp
// CHECK-NEXT: syclcompat/kernel/kernel_lin.cpp
