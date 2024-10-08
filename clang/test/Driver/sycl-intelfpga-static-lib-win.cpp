///
/// tests specific to -fintelfpga -fsycl w/ static libs
///
// REQUIRES: system-windows

// make dummy archive
// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fintelfpga %t1.cpp -c -o %t1_bundle.obj
// RUN: lib -out:%t.lib %t1_bundle.obj

/// Check phases with static lib
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga %t.lib -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_PHASES %s
// CHECK_PHASES: 0: input, "[[INPUT:.+\.lib]]", object, (host-sycl)
// CHECK_PHASES: 1: linker, {0}, host_dep_image, (host-sycl)
// CHECK_PHASES: 2: clang-offload-deps, {1}, ir, (host-sycl)
// CHECK_PHASES: 3: input, "[[INPUT]]", archive
// CHECK_PHASES: 4: clang-offload-unbundler, {3}, tempfilelist
// CHECK_PHASES: 5: spirv-to-ir-wrapper, {4}, tempfilelist, (device-sycl)
// CHECK_PHASES: 6: linker, {2, 5}, ir, (device-sycl)
// CHECK_PHASES: 7: sycl-post-link, {6}, tempfiletable, (device-sycl)
// CHECK_PHASES: 8: file-table-tform, {7}, tempfilelist, (device-sycl)
// CHECK_PHASES: 9: llvm-spirv, {8}, tempfilelist, (device-sycl)
// CHECK_PHASES: 10: input, "[[INPUT]]", archive
// CHECK_PHASES: 11: clang-offload-unbundler, {10}, fpga_dep_list
// CHECK_PHASES: 12: backend-compiler, {9, 11}, fpga_aocx, (device-sycl)
// CHECK_PHASES: 13: file-table-tform, {7, 12}, tempfiletable, (device-sycl)
// CHECK_PHASES: 14: clang-offload-wrapper, {13}, object, (device-sycl)
// CHECK_PHASES: 15: offload, "device-sycl (spir64_fpga-unknown-unknown)" {14}, object
// CHECK_PHASES: 16: linker, {0, 15}, image, (host-sycl)

/// Check for unbundle and use of deps in static lib
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fno-sycl-device-lib=all -fintelfpga -Xshardware %t.lib -### 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_UNBUNDLE %s
// CHECK_UNBUNDLE: clang-offload-bundler" "-type=aoo" "-targets=sycl-fpga_dep" "-input={{.*}}" "-output=[[DEPFILES:.+\.txt]]" "-unbundle"
// CHECK_UNBUNDLE: aoc{{.*}} "-dep-files=@[[DEPFILES]]"

/// Check for no unbundle and use of deps in static lib when using triple
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl -fno-sycl-device-lib=all -Xshardware -fsycl-targets=spir64_fpga-unknown-unknown %t.lib -### 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_NO_UNBUNDLE %s
// CHECK_NO_UNBUNDLE-NOT: clang-offload-bundler" "-type=aoo" "-targets=sycl-fpga_dep"
// CHECK_NO_UNBUNDLE-NOT: aoc{{.*}} "-dep-files={{.*}}"
