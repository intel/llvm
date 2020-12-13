///
/// tests specific to -fintelfpga -fsycl w/ static libs
///
// REQUIRES: clang-driver

// make dummy archive
// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fintelfpga -fsycl %t1.cpp -c -o %t1_bundle.o
// RUN: llvm-ar cr %t.a %t1_bundle.o

/// Check phases with static lib
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga %t.a -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_PHASES %s
// CHECK_PHASES: 0: input, "[[INPUT:.+\.a]]", object, (host-sycl)
// CHECK_PHASES: 1: linker, {0}, image, (host-sycl)
// CHECK_PHASES: 2: input, "[[INPUT]]", archive
// CHECK_PHASES: 3: clang-offload-unbundler, {2}, archive
// CHECK_PHASES: 4: linker, {3}, ir, (device-sycl)
// CHECK_PHASES: 5: sycl-post-link, {4}, ir, (device-sycl)
// CHECK_PHASES: 6: llvm-spirv, {5}, spirv, (device-sycl)
// CHECK_PHASES: 7: input, "[[INPUT]]", archive
// CHECK_PHASES: 8: clang-offload-unbundler, {7}, fpga_dependencies_list
// CHECK_PHASES: 9: backend-compiler, {6, 8}, fpga_aocx, (device-sycl)
// CHECK_PHASES: 10: clang-offload-wrapper, {9}, object, (device-sycl)
// CHECK_PHASES: 11: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {10}, image

/// Check for unbundle and use of deps in static lib
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga %t.a -### 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_UNBUNDLE %s
// CHECK_UNBUNDLE: clang-offload-bundler" "-type=aoo" "-targets=sycl-fpga_dep" "-inputs={{.*}}" "-outputs=[[DEPFILES:.+\.txt]]" "-unbundle"
// CHECK_UNBUNDLE: aoc{{.*}} "-dep-files=@[[DEPFILES]]"
