/// test behaviors of passing a fat static lib with -fno-sycl-rdc
// UNSUPPORTED: system-windows

// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver %t1.cpp -c -o %t1_bundle.o
// RUN: llvm-ar cr %t_lib.a %t1_bundle.o
// RUN: %clang -### -fsycl --no-offload-new-driver -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=off --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl --no-offload-new-driver -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=auto --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl --no-offload-new-driver -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=per_kernel --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl --no-offload-new-driver -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=per_source --sysroot=%S/Inputs/SYCL %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// CHECK: 1: input, "{{.*}}_lib.a", archive
// CHECK: 2: clang-offload-unbundler, {1}, tempfilelist
// CHECK: 3: spirv-to-ir-wrapper, {2}, tempfilelist, (device-sycl)
// CHECK: 4: input, "{{.*}}libsycl-crt.bc", ir, (device-sycl)
// CHECK: 10: linker, {4, {{.*}}}, ir, (device-sycl)
// CHECK: 11: linker, {3, 10}, ir, (device-sycl)
// CHECK: 12: foreach, {3, 11}, ir, (device-sycl)
// CHECK: 13: file-table-tform, {3, 12}, tempfilelist, (device-sycl)
// CHECK: 14: sycl-post-link, {13}, tempfiletable, (device-sycl)
// CHECK: 15: foreach, {13, 14}, tempfiletable, (device-sycl)
// CHECK: 16: file-table-tform, {15}, tempfilelist, (device-sycl)
// CHECK: 17: file-table-tform, {15}, tempfilelist, (device-sycl)
// CHECK: 18: foreach, {13, 17}, tempfilelist, (device-sycl)
// CHECK: 19: file-table-tform, {18}, tempfilelist, (device-sycl)
// CHECK: 20: llvm-spirv, {19}, tempfilelist, (device-sycl)
// CHECK: 21: file-table-tform, {16, 20}, tempfiletable, (device-sycl)
// CHECK: 22: clang-offload-wrapper, {21}, object, (device-sycl)
// CHECK: 23: offload, "device-sycl (spir64-unknown-unknown)" {22}, object
// CHECK: 24: linker, {0, 23}, image, (host-sycl)
