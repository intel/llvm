// REQUIRES: x86-registered-target
// UNSUPPORTED: target={{.*}}-darwin{{.*}}, target={{.*}}-aix{{.*}}, system-windows

// Check working of bundler before and after standardization
// RUN: clang-offload-bundler -type=o -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown -input=%S/Inputs/bundles/bundle_bef_standardization_of_target_triple.o -output=test-host-x86_64-unknown-linux-gnu.o -output=test-sycl-spir64-unknown-unknown.o -unbundle 2>&1 | FileCheck %s -check-prefix=CHECK-STD-OLD --allow-empty
// CHECK-STD-OLD-NOT: error: Can't find bundles for
// RUN: clang-offload-bundler -type=o -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown -input=%S/Inputs/bundles/bundle_aft_standardization_of_target_triple.o -output=test-host-x86_64-unknown-linux-gnu.o -output=test-sycl-spir64-unknown-unknown.o -unbundle 2>&1 | FileCheck %s -check-prefix=CHECK-STD-NEW --allow-empty
// CHECK-STD-NEW-NOT: error: Can't find bundles for


