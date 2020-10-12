///
/// tests specific to -fintelfpga -fsycl w/ static libs
///
// REQUIRES: clang-driver
// REQUIRES: system-windows

// make dummy archive
// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl -fintelfpga %t1.cpp -c -o %t1_bundle.obj
// RUN: lib -out:%t.lib %t1_bundle.obj

/// Check phases with static lib
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl -fno-sycl-device-lib=all -fintelfpga %t.lib -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_PHASES %s
// CHECK_PHASES: 0: input, "[[INPUT:.+\.lib]]", object, (host-sycl)
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
// CHECK_PHASES: 11: offload, "host-sycl (x86_64-pc-windows-msvc)" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {10}, image

/// Check for unbundle and use of deps in static lib
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl -fno-sycl-device-lib=all -fintelfpga %t.lib -### 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_UNBUNDLE %s
// CHECK_UNBUNDLE: clang-offload-bundler" "-type=aoo" "-targets=sycl-fpga_dep" "-inputs={{.*}}" "-outputs=[[DEPFILES:.+\.txt]]" "-unbundle"
// CHECK_UNBUNDLE: aoc{{.*}} "-dep-files=@[[DEPFILES]]"
