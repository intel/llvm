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
// CHECK: 16: linker, {4, {{.*}}}, ir, (device-sycl)
// CHECK: 17: linker, {3, 16}, ir, (device-sycl)
// CHECK: 18: foreach, {3, 17}, ir, (device-sycl)
// CHECK: 19: file-table-tform, {3, 18}, tempfilelist, (device-sycl)
// CHECK: 20: sycl-post-link, {19}, tempfiletable, (device-sycl)
// CHECK: 21: foreach, {19, 20}, tempfiletable, (device-sycl)
// CHECK: 22: file-table-tform, {21}, tempfilelist, (device-sycl)
// CHECK: 23: file-table-tform, {21}, tempfilelist, (device-sycl)
// CHECK: 24: foreach, {19, 23}, tempfilelist, (device-sycl)
// CHECK: 25: file-table-tform, {24}, tempfilelist, (device-sycl)
// CHECK: 26: llvm-spirv, {25}, tempfilelist, (device-sycl)
// CHECK: 27: file-table-tform, {22, 26}, tempfiletable, (device-sycl)
// CHECK: 28: clang-offload-wrapper, {27}, object, (device-sycl)
// CHECK: 29: offload, "device-sycl (spir64-unknown-unknown)" {28}, object
// CHECK: 30: linker, {0, 29}, image, (host-sycl)
