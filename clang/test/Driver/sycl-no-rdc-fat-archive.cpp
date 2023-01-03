/// test behaviors of passing a fat static lib with -fno-sycl-rdc
// UNSUPPORTED: system-windows

// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl %t1.cpp -c -o %t1_bundle.o
// RUN: llvm-ar cr %t_lib.a %t1_bundle.o
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=none --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=auto --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=per_kernel --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=per_source --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// CHECK: 2: input, "{{.*}}_lib.a", archive
// CHECK: 3: clang-offload-unbundler, {2}, tempfilelist
// CHECK: 4: spirv-to-ir-wrapper, {3}, tempfilelist, (device-sycl)
// CHECK: 5: input, "{{.*}}libsycl-crt{{.*}}", object
// CHECK: 6: clang-offload-unbundler, {5}, object
// CHECK: 7: offload, " (spir64-unknown-unknown)" {6}, object
// CHECK: 65: linker, {7, {{.*}}}, ir, (device-sycl)
// CHECK: 66: linker, {4, 65}, ir, (device-sycl)
// CHECK: 67: foreach, {4, 66}, ir, (device-sycl)
// CHECK: 68: file-table-tform, {4, 67}, tempfilelist, (device-sycl)
// CHECK: 69: sycl-post-link, {68}, tempfiletable, (device-sycl)
// CHECK: 70: foreach, {68, 69}, tempfiletable, (device-sycl)
// CHECK: 71: file-table-tform, {70}, tempfilelist, (device-sycl)
// CHECK: 72: file-table-tform, {70}, tempfilelist, (device-sycl)
// CHECK: 73: foreach, {68, 72}, tempfilelist, (device-sycl)
// CHECK: 74: file-table-tform, {73}, tempfilelist, (device-sycl)
// CHECK: 75: llvm-spirv, {74}, tempfilelist, (device-sycl)
// CHECK: 76: file-table-tform, {71, 75}, tempfiletable, (device-sycl)
// CHECK: 77: clang-offload-wrapper, {76}, object, (device-sycl)
// CHECK: 78: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown)" {77}, image
