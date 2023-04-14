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

// CHECK: 21: input, {{.*}}libsycl-crt{{.*}}, object
// CHECK: 22: clang-offload-unbundler, {21}, object
// CHECK: 23: offload, " (spir64-unknown-unknown)" {22}, object
// CHECK: 81: linker, {23, {{.*}}}, ir, (device-sycl)
// CHECK: 82: linker, {5, 81}, ir, (device-sycl)
// CHECK: 83: sycl-post-link, {82}, tempfiletable, (device-sycl)
// CHECK: 84: file-table-tform, {83}, tempfilelist, (device-sycl)
// CHECK: 85: llvm-spirv, {84}, tempfilelist, (device-sycl)
// CHECK: 86: file-table-tform, {83, 85}, tempfiletable, (device-sycl)
// CHECK: 87: clang-offload-wrapper, {86}, object, (device-sycl)

// CHECK: 88: linker, {15, 81}, ir, (device-sycl)
// CHECK: 89: sycl-post-link, {88}, tempfiletable, (device-sycl)
// CHECK: 90: file-table-tform, {89}, tempfilelist, (device-sycl)
// CHECK: 91: llvm-spirv, {90}, tempfilelist, (device-sycl)
// CHECK: 92: file-table-tform, {89, 91}, tempfiletable, (device-sycl)
// CHECK: 93: clang-offload-wrapper, {92}, object, (device-sycl)

// CHECK: 94: offload, "host-sycl (x86_64-unknown-linux-gnu)" {{{.*}}}, "device-sycl (spir64-unknown-unknown)" {87}, "device-sycl (spir64-unknown-unknown)" {93}, image
