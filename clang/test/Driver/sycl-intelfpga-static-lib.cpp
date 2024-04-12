///
/// tests specific to -fintelfpga -fsycl w/ static libs
///

// make dummy archive
// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clangxx -fno-sycl-use-footer --target=x86_64-unknown-linux-gnu -fintelfpga %t1.cpp -c -o %t1_bundle.o
// RUN: llvm-ar cr %t.a %t1_bundle.o

/// Check phases with static lib
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fno-sycl-dead-args-optimization -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga %t.a -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_PHASES %s
// CHECK_PHASES: 0: input, "[[INPUT:.+\.a]]", object, (host-sycl)
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
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fno-sycl-device-lib=all -fintelfpga -Xshardware %t.a -### 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_UNBUNDLE %s
// CHECK_UNBUNDLE: clang-offload-bundler" "-type=aoo" "-targets=sycl-fpga_dep" "-input={{.*}}" "-output=[[DEPFILES:.+\.txt]]" "-unbundle"
// CHECK_UNBUNDLE: aoc{{.*}} "-dep-files=@[[DEPFILES]]"

/// Check for no unbundle and use of deps in static lib when using triple
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown %t.a -### 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_NO_UNBUNDLE %s
// CHECK_NO_UNBUNDLE-NOT: clang-offload-bundler" "-type=aoo" "-targets=sycl-fpga_dep"
// CHECK_NO_UNBUNDLE-NOT: aoc{{.*}} "-dep-files={{.*}}"
