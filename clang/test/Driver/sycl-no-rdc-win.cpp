/// Tests for -fno-sycl-rdc on Windows
// REQUIRES: system-windows

// RUN: touch %t1.cpp
// RUN: touch %t2.cpp
// RUN: %clang -### -fsycl -fno-sycl-rdc --sysroot=%S/Inputs/SYCL-windows %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc /clang:--sysroot=%S/Inputs/SYCL-windows  %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s

// CHECK: 3: input, "{{.*}}1.cpp", c++, (device-sycl)
// CHECK: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHECK: 5: compiler, {4}, ir, (device-sycl)
// CHECK: 13: input, "{{.*}}2.cpp", c++, (device-sycl)
// CHECK: 14: preprocessor, {13}, c++-cpp-output, (device-sycl)
// CHECK: 15: compiler, {14}, ir, (device-sycl)

// CHECK: 21: input, {{.*}}libsycl-crt{{.*}}, object
// CHECK: 22: clang-offload-unbundler, {21}, object
// CHECK: 23: offload, " (spir64-unknown-unknown)" {22}, object
// CHECK: 84: linker, {23, {{.*}}}, ir, (device-sycl)
// CHECK: 85: linker, {5, 84}, ir, (device-sycl)
// CHECK: 86: sycl-post-link, {85}, tempfiletable, (device-sycl)
// CHECK: 87: file-table-tform, {86}, tempfilelist, (device-sycl)
// CHECK: 88: llvm-spirv, {87}, tempfilelist, (device-sycl)
// CHECK: 89: file-table-tform, {86, 88}, tempfiletable, (device-sycl)
// CHECK: 90: clang-offload-wrapper, {89}, object, (device-sycl)

// CHECK: 91: linker, {15, 84}, ir, (device-sycl)
// CHECK: 92: sycl-post-link, {91}, tempfiletable, (device-sycl)
// CHECK: 93: file-table-tform, {92}, tempfilelist, (device-sycl)
// CHECK: 94: llvm-spirv, {93}, tempfilelist, (device-sycl)
// CHECK: 95: file-table-tform, {92, 94}, tempfiletable, (device-sycl)
// CHECK: 96: clang-offload-wrapper, {95}, object, (device-sycl)

// CHECK: 97: offload, "host-sycl (x86_64-pc-windows-msvc)" {{{.*}}}, "device-sycl (spir64-unknown-unknown)" {90}, "device-sycl (spir64-unknown-unknown)" {96}, image
