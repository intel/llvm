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
// CHECK: 15: linker, {4, {{.*}}}, ir, (device-sycl)
// CHECK: 16: linker, {3, 15}, ir, (device-sycl)
// CHECK: 17: foreach, {3, 16}, ir, (device-sycl)
// CHECK: 18: file-table-tform, {3, 17}, tempfilelist, (device-sycl)
// CHECK: 19: sycl-post-link, {18}, tempfiletable, (device-sycl)
// CHECK: 20: foreach, {18, 19}, tempfiletable, (device-sycl)
// CHECK: 21: file-table-tform, {20}, tempfilelist, (device-sycl)
// CHECK: 22: file-table-tform, {20}, tempfilelist, (device-sycl)
// CHECK: 23: foreach, {18, 22}, tempfilelist, (device-sycl)
// CHECK: 24: file-table-tform, {23}, tempfilelist, (device-sycl)
// CHECK: 25: llvm-spirv, {24}, tempfilelist, (device-sycl)
// CHECK: 26: file-table-tform, {21, 25}, tempfiletable, (device-sycl)
// CHECK: 27: clang-offload-wrapper, {26}, object, (device-sycl)
// CHECK: 28: offload, "device-sycl (spir64-unknown-unknown)" {27}, object
// CHECK: 29: linker, {0, 28}, image, (host-sycl)
