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
// CHECK: 4: input, "{{.*}}libsycl-devicelib.bc", ir, (device-sycl)
// CHECK: 11: linker, {4, {{.*}}}, ir, (device-sycl)
// CHECK: 12: linker, {3, 11}, ir, (device-sycl)
// CHECK: 13: foreach, {3, 12}, ir, (device-sycl)
// CHECK: 14: file-table-tform, {3, 13}, tempfilelist, (device-sycl)
// CHECK: 15: sycl-post-link, {14}, tempfiletable, (device-sycl)
// CHECK: 16: foreach, {14, 15}, tempfiletable, (device-sycl)
// CHECK: 17: file-table-tform, {16}, tempfilelist, (device-sycl)
// CHECK: 18: file-table-tform, {16}, tempfilelist, (device-sycl)
// CHECK: 19: foreach, {14, 18}, tempfilelist, (device-sycl)
// CHECK: 20: file-table-tform, {19}, tempfilelist, (device-sycl)
// CHECK: 21: llvm-spirv, {20}, tempfilelist, (device-sycl)
// CHECK: 22: file-table-tform, {17, 21}, tempfiletable, (device-sycl)
// CHECK: 23: clang-offload-wrapper, {22}, object, (device-sycl)
// CHECK: 24: offload, "device-sycl (spir64-unknown-unknown)" {23}, object
// CHECK: 25: linker, {0, 24}, image, (host-sycl)
