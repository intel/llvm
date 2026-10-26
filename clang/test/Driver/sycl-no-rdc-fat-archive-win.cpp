/// test behaviors of passing a fat static lib with -fno-sycl-rdc on Windows
// REQUIRES: system-windows

/// Directory holding the compiler resource files, including the SYCL
/// compiler-rt builtins library (libclang_rt.builtins.bc).
// DEFINE: %{resource_dir} = %/S/Inputs/SYCL/lib/clang/resource_dir

// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clangxx -target x86_64-pc-windows-msvc -fsycl %t1.cpp -c -o %t1_bundle.o
// RUN: llvm-ar cr %t_lib.a %t1_bundle.o
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=off --sysroot=%S/Inputs/SYCL -resource-dir=%{resource_dir} %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=auto --sysroot=%S/Inputs/SYCL -resource-dir=%{resource_dir} %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=per_kernel --sysroot=%S/Inputs/SYCL -resource-dir=%{resource_dir} %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=per_source --sysroot=%S/Inputs/SYCL -resource-dir=%{resource_dir} %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=off /clang:--sysroot=%S/Inputs/SYCL -resource-dir=%{resource_dir} %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=auto /clang:--sysroot=%S/Inputs/SYCL -resource-dir=%{resource_dir} %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=per_kernel /clang:--sysroot=%S/Inputs/SYCL -resource-dir=%{resource_dir} %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -fsycl-device-code-split=per_source /clang:--sysroot=%S/Inputs/SYCL -resource-dir=%{resource_dir} %t_lib.a 2>&1 -ccc-print-phases | FileCheck %s
// CHECK: 0: input, "[[INPUT:.+\.a]]", object, (host-sycl)
// CHECK: 1: input, "[[INPUT]]", archive
// CHECK: 2: clang-offload-unbundler, {1}, tempfilelist
// CHECK: 3: spirv-to-ir-wrapper, {2}, tempfilelist, (device-sycl)
// CHECK: 4: input, "{{.*}}libsycl-crt.bc", ir, (device-sycl)
// CHECK: 11: input, "{{.*}}libc.bc", ir, (device-sycl)
// CHECK: 12: input, "{{.*}}libclang_rt.builtins.bc", ir, (device-sycl)
// CHECK: 13: linker, {4, {{.*}}, 12}, ir, (device-sycl)
// CHECK: 14: linker, {3, 13}, ir, (device-sycl)
// CHECK: 15: foreach, {3, 14}, ir, (device-sycl)
// CHECK: 16: file-table-tform, {3, 15}, tempfilelist, (device-sycl)
// CHECK: 17: sycl-post-link, {16}, tempfiletable, (device-sycl)
// CHECK: 18: foreach, {16, 17}, tempfiletable, (device-sycl)
// CHECK: 19: file-table-tform, {18}, tempfilelist, (device-sycl)
// CHECK: 20: file-table-tform, {18}, tempfilelist, (device-sycl)
// CHECK: 21: foreach, {16, 20}, tempfilelist, (device-sycl)
// CHECK: 22: file-table-tform, {21}, tempfilelist, (device-sycl)
// CHECK: 23: llvm-spirv, {22}, tempfilelist, (device-sycl)
// CHECK: 24: file-table-tform, {19, 23}, tempfiletable, (device-sycl)
// CHECK: 25: clang-offload-wrapper, {24}, object, (device-sycl)
// CHECK: 26: offload, "device-sycl (spir64-unknown-unknown)" {25}, object
// CHECK: 27: linker, {0, 26}, image, (host-sycl)
