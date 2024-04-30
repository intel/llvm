/// Tests for -fno-sycl-rdc
// UNSUPPORTED: system-windows

// RUN: touch %t1.cpp
// RUN: touch %t2.cpp
// RUN: %clang -### -fsycl -fno-sycl-rdc --sysroot=%S/Inputs/SYCL %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s

// CHECK: 3: input, "{{.*}}1.cpp", c++, (device-sycl)
// CHECK: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHECK: 5: compiler, {4}, ir, (device-sycl)
// CHECK: 13: input, "{{.*}}2.cpp", c++, (device-sycl)
// CHECK: 14: preprocessor, {13}, c++-cpp-output, (device-sycl)
// CHECK: 15: compiler, {14}, ir, (device-sycl)
// CHECK: 20: input, {{.*}}libsycl-crt{{.*}}, object
// CHECK: 21: clang-offload-unbundler, {20}, object
// CHECK: 22: offload, " (spir64-unknown-unknown)" {21}, object
// CHECK: 80: linker, {22, {{.*}}}, ir, (device-sycl)
// CHECK: 81: linker, {5, 80}, ir, (device-sycl)
// CHECK: 82: sycl-post-link, {81}, tempfiletable, (device-sycl)
// CHECK: 83: file-table-tform, {82}, tempfilelist, (device-sycl)
// CHECK: 84: llvm-spirv, {83}, tempfilelist, (device-sycl)
// CHECK: 85: file-table-tform, {82, 84}, tempfiletable, (device-sycl)
// CHECK: 86: clang-offload-wrapper, {85}, object, (device-sycl)
// CHECK: 87: offload, "device-sycl (spir64-unknown-unknown)" {86}, object
// CHECK: 88: linker, {15, 80}, ir, (device-sycl)
// CHECK: 89: sycl-post-link, {88}, tempfiletable, (device-sycl)
// CHECK: 90: file-table-tform, {89}, tempfilelist, (device-sycl)
// CHECK: 91: llvm-spirv, {90}, tempfilelist, (device-sycl)
// CHECK: 92: file-table-tform, {89, 91}, tempfiletable, (device-sycl)
// CHECK: 93: clang-offload-wrapper, {92}, object, (device-sycl)
// CHECK: 94: offload, "device-sycl (spir64-unknown-unknown)" {93}, object
// CHECK: 95: linker, {9, 19, 87, 94}, image, (host-sycl)
