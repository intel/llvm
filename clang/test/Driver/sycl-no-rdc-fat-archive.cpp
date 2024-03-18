/// test behaviors of passing a fat static lib with -fno-sycl-rdc
// UNSUPPORTED: system-windows

// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl %t1.cpp -c -o %t1_bundle.o
// RUN: llvm-ar cr %t_lib.a %t1_bundle.o
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=off --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=auto --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=per_kernel --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-device-code-split=per_source --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// CHECK: 1: input, "{{.*}}_lib.a", archive
// CHECK: 2: clang-offload-unbundler, {1}, tempfilelist
// CHECK: 3: spirv-to-ir-wrapper, {2}, tempfilelist, (device-sycl)
// CHECK: 4: input, "{{.*}}libsycl-crt{{.*}}", object
// CHECK: 5: clang-offload-unbundler, {4}, object
// CHECK: 6: offload, " (spir64-unknown-unknown)" {5}, object
// CHECK: 64: linker, {6, {{.*}}}, ir, (device-sycl)
// CHECK: 65: linker, {3, 64}, ir, (device-sycl)
// CHECK: 66: foreach, {3, 65}, ir, (device-sycl)
// CHECK: 67: file-table-tform, {3, 66}, tempfilelist, (device-sycl)
// CHECK: 68: sycl-post-link, {67}, tempfiletable, (device-sycl)
// CHECK: 69: foreach, {67, 68}, tempfiletable, (device-sycl)
// CHECK: 70: file-table-tform, {69}, tempfilelist, (device-sycl)
// CHECK: 71: file-table-tform, {69}, tempfilelist, (device-sycl)
// CHECK: 72: foreach, {67, 71}, tempfilelist, (device-sycl)
// CHECK: 73: file-table-tform, {72}, tempfilelist, (device-sycl)
// CHECK: 74: llvm-spirv, {73}, tempfilelist, (device-sycl)
// CHECK: 75: file-table-tform, {70, 74}, tempfiletable, (device-sycl)
// CHECK: 76: clang-offload-wrapper, {75}, object, (device-sycl)
// CHECK: 77: offload, "device-sycl (spir64-unknown-unknown)" {76}, object
// CHECK: 78: linker, {0, 77}, image, (host-sycl)
