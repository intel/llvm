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
// CHECK: 23: linker, {4, {{.*}}}, ir, (device-sycl)
// CHECK: 24: linker, {3, 23}, ir, (device-sycl)
// CHECK: 25: foreach, {3, 24}, ir, (device-sycl)
// CHECK: 26: file-table-tform, {3, 25}, tempfilelist, (device-sycl)
// CHECK: 27: sycl-post-link, {26}, tempfiletable, (device-sycl)
// CHECK: 28: foreach, {26, 27}, tempfiletable, (device-sycl)
// CHECK: 29: file-table-tform, {28}, tempfilelist, (device-sycl)
// CHECK: 30: file-table-tform, {28}, tempfilelist, (device-sycl)
// CHECK: 31: foreach, {26, 30}, tempfilelist, (device-sycl)
// CHECK: 32: file-table-tform, {31}, tempfilelist, (device-sycl)
// CHECK: 33: llvm-spirv, {32}, tempfilelist, (device-sycl)
// CHECK: 34: file-table-tform, {29, 33}, tempfiletable, (device-sycl)
// CHECK: 35: clang-offload-wrapper, {34}, object, (device-sycl)
// CHECK: 36: offload, "device-sycl (spir64-unknown-unknown)" {35}, object
// CHECK: 37: linker, {0, 36}, image, (host-sycl)
