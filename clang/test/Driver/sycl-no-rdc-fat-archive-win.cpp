/// test behaviors of passing a fat static lib with -fno-sycl-rdc on Windows
// REQUIRES: system-windows

// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clangxx -target x86_64-pc-windows-msvc -fsycl %t1.cpp -c -o %t1_bundle.o
// RUN: llvm-ar cr %t_lib.a %t1_bundle.o
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=off --sysroot=%S/Inputs/SYCL-windows %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=auto --sysroot=%S/Inputs/SYCL-windows %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=per_kernel --sysroot=%S/Inputs/SYCL-windows %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=per_source --sysroot=%S/Inputs/SYCL-windows %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=off /clang:--sysroot=%S/Inputs/SYCL-windows %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=auto /clang:--sysroot=%S/Inputs/SYCL-windows %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=per_kernel /clang:--sysroot=%S/Inputs/SYCL-windows %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=per_source /clang:--sysroot=%S/Inputs/SYCL-windows %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// CHECK: 2: input, "{{.*}}_lib.a", archive
// CHECK: 3: clang-offload-unbundler, {2}, tempfilelist
// CHECK: 4: spirv-to-ir-wrapper, {3}, tempfilelist, (device-sycl)
// CHECK: 5: input, "{{.*}}libsycl-crt{{.*}}", object
// CHECK: 6: clang-offload-unbundler, {5}, object
// CHECK: 7: offload, " (spir64-unknown-unknown)" {6}, object
// CHECK: 68: linker, {7, {{.*}}}, ir, (device-sycl)
// CHECK: 69: linker, {4, 68}, ir, (device-sycl)
// CHECK: 70: foreach, {4, 69}, ir, (device-sycl)
// CHECK: 71: file-table-tform, {4, 70}, tempfilelist, (device-sycl)
// CHECK: 72: sycl-post-link, {71}, tempfiletable, (device-sycl)
// CHECK: 73: foreach, {71, 72}, tempfiletable, (device-sycl)
// CHECK: 74: file-table-tform, {73}, tempfilelist, (device-sycl)
// CHECK: 75: file-table-tform, {73}, tempfilelist, (device-sycl)
// CHECK: 76: foreach, {71, 75}, tempfilelist, (device-sycl)
// CHECK: 77: file-table-tform, {76}, tempfilelist, (device-sycl)
// CHECK: 78: llvm-spirv, {77}, tempfilelist, (device-sycl)
// CHECK: 79: file-table-tform, {74, 78}, tempfiletable, (device-sycl)
// CHECK: 80: clang-offload-wrapper, {79}, object, (device-sycl)
// CHECK: 81: offload, "host-sycl (x86_64-pc-windows-msvc)" {1}, "device-sycl (spir64-unknown-unknown)" {80}, image
