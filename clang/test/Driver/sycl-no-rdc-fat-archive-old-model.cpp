/// test behaviors of passing a fat static lib with -fno-sycl-rdc
// UNSUPPORTED: system-windows

// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver %t1.cpp -c -o %t1_bundle.o
// RUN: llvm-ar cr %t_lib.a %t1_bundle.o
// RUN: %clang -### -fsycl --no-offload-new-driver -fno-sycl-rdc -fsycl-device-code-split=off --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl --no-offload-new-driver -fno-sycl-rdc -fsycl-device-code-split=auto --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl --no-offload-new-driver -fno-sycl-rdc -fsycl-device-code-split=per_kernel --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl --no-offload-new-driver -fno-sycl-rdc -fsycl-device-code-split=per_source --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// CHECK: 1: input, "{{.*}}_lib.a", archive
// CHECK: 2: clang-offload-unbundler, {1}, tempfilelist
// CHECK: 3: spirv-to-ir-wrapper, {2}, tempfilelist, (device-sycl)
// CHECK: 4: input, "{{.*}}libsycl-crt.bc", ir, (device-sycl)
// CHECK: 24: linker, {4, {{.*}}}, ir, (device-sycl)
// CHECK: 25: linker, {3, 24}, ir, (device-sycl)
// CHECK: 26: foreach, {3, 25}, ir, (device-sycl)
// CHECK: 27: file-table-tform, {3, 26}, tempfilelist, (device-sycl)
// CHECK: 28: sycl-post-link, {27}, tempfiletable, (device-sycl)
// CHECK: 29: foreach, {27, 28}, tempfiletable, (device-sycl)
// CHECK: 30: file-table-tform, {29}, tempfilelist, (device-sycl)
// CHECK: 31: file-table-tform, {29}, tempfilelist, (device-sycl)
// CHECK: 32: foreach, {27, 31}, tempfilelist, (device-sycl)
// CHECK: 33: file-table-tform, {32}, tempfilelist, (device-sycl)
// CHECK: 34: llvm-spirv, {33}, tempfilelist, (device-sycl)
// CHECK: 35: file-table-tform, {30, 34}, tempfiletable, (device-sycl)
// CHECK: 36: clang-offload-wrapper, {35}, object, (device-sycl)
// CHECK: 37: offload, "device-sycl (spir64-unknown-unknown)" {36}, object
// CHECK: 38: linker, {0, 37}, image, (host-sycl)
