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
// CHECK_PHASES: 3: partial-link, {2}, object
// CHECK_PHASES: 4: clang-offload-unbundler, {3}, object
// CHECK_PHASES: 5: linker, {4}, ir, (device-sycl)
// CHECK_PHASES: 6: sycl-post-link, {5}, ir, (device-sycl)
// CHECK_PHASES: 7: llvm-spirv, {6}, spirv, (device-sycl)
// CHECK_PHASES: 8: input, "[[INPUT]]", archive
// CHECK_PHASES: 9: clang-offload-unbundler, {8}, fpga_dependencies_list
// CHECK_PHASES: 10: backend-compiler, {7, 9}, fpga_aocx, (device-sycl)
// CHECK_PHASES: 11: clang-offload-wrapper, {10}, object, (device-sycl)
// CHECK_PHASES: 12: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {11}, image

/// Check for unbundle and use of deps in static lib
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga %t.a -### 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_UNBUNDLE %s
// CHECK_UNBUNDLE: clang-offload-bundler" "-type=aoo" "-targets=sycl-fpga_dep" "-inputs={{.*}}" "-outputs=[[DEPFILES:.+\.txt]]" "-unbundle"
// CHECK_UNBUNDLE: aoc{{.*}} "-dep-files=@[[DEPFILES]]"
