// This test is intended to ensure that we have no tests marked as XFAIL
// without a tracker information added to a test.
// For more info see: sycl/test-e2e/README.md
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
// - ...and check if the list of improperly XFAIL-ed tests needs to be updated.
//
// RUN: grep -rI "XFAIL:" %S/../../test-e2e \
// RUN: -A 1 --include=*.cpp --no-group-separator | \
// RUN: grep -v "XFAIL:" | \
// RUN: grep -Pv "XFAIL-TRACKER:\s+(?:https://github.com/[\w\d-]+/[\w\d-]+/issues/[\d]+)|(?:[\w]+-[\d]+)" > %t
// RUN: cat %t | wc -l | FileCheck %s --check-prefix NUMBER-OF-XFAIL-WITHOUT-TRACKER
// RUN: cat %t | sed 's/\.cpp.*/.cpp/' | sort | FileCheck %s
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
// or have it reduced (preferably, down to zero).
//
// If you see this test failed for your patch, it means that you either
// introduced XFAIL directive to a test improperly, or broke the format of an
// existing XFAIL-ed tests.
// Another possibility (and that is a good option) is that you updated some
// tests to match the required format and in that case you should just update
// (i.e. reduce) the number and the list below.
//
// NUMBER-OF-XFAIL-WITHOUT-TRACKER: 41
//
// List of improperly XFAIL-ed tests.
// Remove the CHECK once the test has been properly XFAIL-ed.
//
// CHECK: AddressSanitizer/nullpointer/private_nullptr.cpp
// CHECK-NEXT: Basic/aspects.cpp
// CHECK-NEXT: Basic/device_event.cpp
// CHECK-NEXT: Basic/diagnostics/handler.cpp
// CHECK-NEXT: Basic/max_linear_work_group_size_props.cpp
// CHECK-NEXT: Basic/max_work_group_size_props.cpp
// CHECK-NEXT: Basic/partition_supported.cpp
// CHECK-NEXT: DeviceLib/assert-windows.cpp
// CHECK-NEXT: ESIMD/hardware_dispatch.cpp
// CHECK-NEXT: InlineAsm/asm_multiple_instructions.cpp
// CHECK-NEXT: KernelAndProgram/kernel-bundle-merge-options.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_annotated_ptr.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_bfloat16_colmajorA_colmajorB.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_bfloat16_packedB.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_colA_rowB_colC.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_int8_colmajorA_colmajorB.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_int8_rowmajorA_rowmajorB.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_out_bounds.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_prefetch.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_rowmajorA_rowmajorB.cpp
// CHECK-NEXT: Matrix/SG32/joint_matrix_unaligned_k.cpp
// CHECK-NEXT: Matrix/joint_matrix_bfloat16_colmajorA_colmajorB.cpp
// CHECK-NEXT: Matrix/joint_matrix_colA_rowB_colC.cpp
// CHECK-NEXT: Matrix/joint_matrix_int8_colmajorA_colmajorB.cpp
// CHECK-NEXT: NewOffloadDriver/multisource.cpp
// CHECK-NEXT: NewOffloadDriver/split-per-source-main.cpp
// CHECK-NEXT: NewOffloadDriver/sycl-external-with-optional-features.cpp
// CHECK-NEXT: OptionalKernelFeatures/throw-exception-for-out-of-registers-on-kernel-launch.cpp
// CHECK-NEXT: PerformanceTests/Reduction/reduce_over_sub_group.cpp
// CHECK-NEXT: Printf/int.cpp
// CHECK-NEXT: Printf/mixed-address-space.cpp
// CHECK-NEXT: Printf/percent-symbol.cpp
// CHECK-NEXT: Reduction/reduction_big_data.cpp
// CHECK-NEXT: Reduction/reduction_nd_reducer_skip.cpp
// CHECK-NEXT: Reduction/reduction_reducer_op_eq.cpp
// CHECK-NEXT: Regression/build_log.cpp
// CHECK-NEXT: Regression/complex_global_object.cpp
// CHECK-NEXT: Regression/context_is_destroyed_after_exception.cpp
// CHECK-NEXT: Regression/kernel_bundle_ignore_sycl_external.cpp
// CHECK-NEXT: Regression/multiple-targets.cpp
// CHECK-NEXT: Regression/reduction_resource_leak_dw.cpp
