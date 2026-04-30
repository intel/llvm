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
// CHECK: 17: linker, {4, {{.*}}}, ir, (device-sycl)
// CHECK: 18: linker, {3, 17}, ir, (device-sycl)
// CHECK: 19: foreach, {3, 18}, ir, (device-sycl)
// CHECK: 20: file-table-tform, {3, 19}, tempfilelist, (device-sycl)
// CHECK: 21: sycl-post-link, {20}, tempfiletable, (device-sycl)
// CHECK: 22: foreach, {20, 21}, tempfiletable, (device-sycl)
// CHECK: 23: file-table-tform, {22}, tempfilelist, (device-sycl)
// CHECK: 24: file-table-tform, {22}, tempfilelist, (device-sycl)
// CHECK: 25: foreach, {20, 24}, tempfilelist, (device-sycl)
// CHECK: 26: file-table-tform, {25}, tempfilelist, (device-sycl)
// CHECK: 27: llvm-spirv, {26}, tempfilelist, (device-sycl)
// CHECK: 28: file-table-tform, {23, 27}, tempfiletable, (device-sycl)
// CHECK: 29: clang-offload-wrapper, {28}, object, (device-sycl)
// CHECK: 30: offload, "device-sycl (spir64-unknown-unknown)" {29}, object
// CHECK: 31: linker, {0, 30}, image, (host-sycl)
