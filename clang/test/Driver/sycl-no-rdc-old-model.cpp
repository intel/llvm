/// Tests for -fno-sycl-rdc
// UNSUPPORTED: system-windows

// RUN: touch %t1.cpp
// RUN: touch %t2.cpp
// RUN: %clang -### -fsycl --no-offload-new-driver -fno-sycl-rdc -fsycl-instrument-device-code --sysroot=%S/Inputs/SYCL %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s

// CHECK: 2: input, "{{.*}}1.cpp", c++, (device-sycl)
// CHECK: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHECK: 4: compiler, {3}, ir, (device-sycl)
// CHECK: 11: input, "{{.*}}2.cpp", c++, (device-sycl)
// CHECK: 12: preprocessor, {11}, c++-cpp-output, (device-sycl)
// CHECK: 13: compiler, {12}, ir, (device-sycl)
// CHECK: 18: input, "{{.*}}libsycl-crt.bc", ir, (device-sycl)
// CHECK: 24: linker, {18, {{.*}}}, ir, (device-sycl)
// CHECK: 25: linker, {4, 24}, ir, (device-sycl)
// CHECK: 26: sycl-post-link, {25}, tempfiletable, (device-sycl)
// CHECK: 27: file-table-tform, {26}, tempfilelist, (device-sycl)
// CHECK: 28: llvm-spirv, {27}, tempfilelist, (device-sycl)
// CHECK: 29: file-table-tform, {26, 28}, tempfiletable, (device-sycl)
// CHECK: 30: clang-offload-wrapper, {29}, object, (device-sycl)
// CHECK: 31: offload, "device-sycl (spir64-unknown-unknown)" {30}, object
// CHECK: 32: linker, {13, 24}, ir, (device-sycl)
// CHECK: 33: sycl-post-link, {32}, tempfiletable, (device-sycl)
// CHECK: 34: file-table-tform, {33}, tempfilelist, (device-sycl)
// CHECK: 35: llvm-spirv, {34}, tempfilelist, (device-sycl)
// CHECK: 36: file-table-tform, {33, 35}, tempfiletable, (device-sycl)
// CHECK: 37: clang-offload-wrapper, {36}, object, (device-sycl)
// CHECK: 38: offload, "device-sycl (spir64-unknown-unknown)" {37}, object
// CHECK: 39: linker, {8, 17, 31, 38}, image, (host-sycl)
