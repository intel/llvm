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
// CHECK: 18: input, "{{.*}}libsycl-devicelib.bc", ir, (device-sycl)
// CHECK: 25: linker, {18, {{.*}}}, ir, (device-sycl)
// CHECK: 26: linker, {4, 25}, ir, (device-sycl)
// CHECK: 27: sycl-post-link, {26}, tempfiletable, (device-sycl)
// CHECK: 28: file-table-tform, {27}, tempfilelist, (device-sycl)
// CHECK: 29: llvm-spirv, {28}, tempfilelist, (device-sycl)
// CHECK: 30: file-table-tform, {27, 29}, tempfiletable, (device-sycl)
// CHECK: 31: clang-offload-wrapper, {30}, object, (device-sycl)
// CHECK: 32: offload, "device-sycl (spir64-unknown-unknown)" {31}, object
// CHECK: 33: linker, {13, 25}, ir, (device-sycl)
// CHECK: 34: sycl-post-link, {33}, tempfiletable, (device-sycl)
// CHECK: 35: file-table-tform, {34}, tempfilelist, (device-sycl)
// CHECK: 36: llvm-spirv, {35}, tempfilelist, (device-sycl)
// CHECK: 37: file-table-tform, {34, 36}, tempfiletable, (device-sycl)
// CHECK: 38: clang-offload-wrapper, {37}, object, (device-sycl)
// CHECK: 39: offload, "device-sycl (spir64-unknown-unknown)" {38}, object
// CHECK: 40: linker, {8, 17, 32, 39}, image, (host-sycl)
