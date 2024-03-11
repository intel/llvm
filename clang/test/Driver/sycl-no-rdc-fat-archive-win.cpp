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
// CHECK: 0: input, "[[INPUT:.+\.a]]", object, (host-sycl)
// CHECK: 1: input, "[[INPUT]]", archive
// CHECK: 2: clang-offload-unbundler, {1}, tempfilelist
// CHECK: 3: spirv-to-ir-wrapper, {2}, tempfilelist, (device-sycl)
// CHECK: 4: input, "{{.*}}libsycl-crt{{.*}}", object
// CHECK: 5: clang-offload-unbundler, {4}, object
// CHECK: 6: offload, " (spir64-unknown-unknown)" {5}, object
// CHECK: 67: linker, {6, {{.*}}}, ir, (device-sycl)
// CHECK: 68: linker, {3, 67}, ir, (device-sycl)
// CHECK: 69: foreach, {3, 68}, ir, (device-sycl)
// CHECK: 70: file-table-tform, {3, 69}, tempfilelist, (device-sycl)
// CHECK: 71: sycl-post-link, {70}, tempfiletable, (device-sycl)
// CHECK: 72: foreach, {70, 71}, tempfiletable, (device-sycl)
// CHECK: 73: file-table-tform, {72}, tempfilelist, (device-sycl)
// CHECK: 74: file-table-tform, {72}, tempfilelist, (device-sycl)
// CHECK: 75: foreach, {70, 74}, tempfilelist, (device-sycl)
// CHECK: 76: file-table-tform, {75}, tempfilelist, (device-sycl)
// CHECK: 77: llvm-spirv, {76}, tempfilelist, (device-sycl)
// CHECK: 78: file-table-tform, {73, 77}, tempfiletable, (device-sycl)
// CHECK: 79: clang-offload-wrapper, {78}, object, (device-sycl)
// CHECK: 80: offload, "device-sycl (spir64-unknown-unknown)" {79}, object
// CHECK: 81: linker, {0, 80}, image, (host-sycl)
