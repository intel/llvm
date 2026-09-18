/// test behaviors of passing a fat static lib with -fno-sycl-rdc on Windows
// REQUIRES: system-windows

// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clangxx -target x86_64-pc-windows-msvc -fsycl %t1.cpp -c -o %t1_bundle.o
// RUN: llvm-ar cr %t_lib.a %t1_bundle.o
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=off --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=auto --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=per_kernel --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=per_source --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=off /clang:--sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=auto /clang:--sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=per_kernel /clang:--sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=per_source /clang:--sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// CHECK: 0: input, "[[INPUT:.+\.a]]", object, (host-sycl)
// CHECK: 1: input, "[[INPUT]]", archive
// CHECK: 2: clang-offload-unbundler, {1}, tempfilelist
// CHECK: 3: spirv-to-ir-wrapper, {2}, tempfilelist, (device-sycl)
// CHECK: 4: input, "{{.*}}libsycl-crt{{.*}}", ir, (device-sycl)
// CHECK: 12: linker, {4, {{.*}}}, ir, (device-sycl)
// CHECK: 13: linker, {3, 12}, ir, (device-sycl)
// CHECK: 14: foreach, {3, 13}, ir, (device-sycl)
// CHECK: 15: file-table-tform, {3, 14}, tempfilelist, (device-sycl)
// CHECK: 16: sycl-post-link, {15}, tempfiletable, (device-sycl)
// CHECK: 17: foreach, {15, 16}, tempfiletable, (device-sycl)
// CHECK: 18: file-table-tform, {17}, tempfilelist, (device-sycl)
// CHECK: 19: file-table-tform, {17}, tempfilelist, (device-sycl)
// CHECK: 20: foreach, {15, 19}, tempfilelist, (device-sycl)
// CHECK: 21: file-table-tform, {20}, tempfilelist, (device-sycl)
// CHECK: 22: llvm-spirv, {21}, tempfilelist, (device-sycl)
// CHECK: 23: file-table-tform, {18, 22}, tempfiletable, (device-sycl)
// CHECK: 24: clang-offload-wrapper, {23}, object, (device-sycl)
// CHECK: 25: offload, "device-sycl (spir64-unknown-unknown)" {24}, object
// CHECK: 26: linker, {0, 25}, image, (host-sycl)
