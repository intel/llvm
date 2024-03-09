/// Tests for -fno-sycl-rdc on Windows
// REQUIRES: system-windows

// RUN: touch %t1.cpp
// RUN: touch %t2.cpp
// RUN: %clang -### -fsycl -fno-sycl-rdc --sysroot=%S/Inputs/SYCL-windows %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc /clang:--sysroot=%S/Inputs/SYCL-windows  %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s
// CHECK: 3: input, "{{.*}}1.cpp", c++, (device-sycl)
// CHECK: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHECK: 5: compiler, {4}, ir, (device-sycl)
// CHECK: 6: offload, "host-sycl (x86_64-pc-windows-msvc)" {2}, "device-sycl (spir64-unknown-unknown)" {5}, c++-cpp-output
// CHECK: 13: input, "{{.*}}2.cpp", c++, (device-sycl)
// CHECK: 14: preprocessor, {13}, c++-cpp-output, (device-sycl)
// CHECK: 15: compiler, {14}, ir, (device-sycl)
// CHECK: 16: offload, "host-sycl (x86_64-pc-windows-msvc)" {12}, "device-sycl (spir64-unknown-unknown)" {15}, c++-cpp-output
// CHECK: 20: input, "{{.*}}libsycl-crt{{.*}}", object
// CHECK: 21: clang-offload-unbundler, {20}, object
// CHECK: 22: offload, " (spir64-unknown-unknown)" {21}, object
// CHECK: 83: linker, {22, {{.*}}}, ir, (device-sycl)
// CHECK: 84: linker, {5, 83}, ir, (device-sycl)
// CHECK: 85: sycl-post-link, {84}, tempfiletable, (device-sycl)
// CHECK: 86: file-table-tform, {85}, tempfilelist, (device-sycl)
// CHECK: 87: llvm-spirv, {86}, tempfilelist, (device-sycl)
// CHECK: 88: file-table-tform, {85, 87}, tempfiletable, (device-sycl)
// CHECK: 89: clang-offload-wrapper, {88}, object, (device-sycl)
// CHECK: 90: offload, "device-sycl (spir64-unknown-unknown)" {89}, object
// CHECK: 91: linker, {15, 83}, ir, (device-sycl)
// CHECK: 92: sycl-post-link, {91}, tempfiletable, (device-sycl)
// CHECK: 93: file-table-tform, {92}, tempfilelist, (device-sycl)
// CHECK: 94: llvm-spirv, {93}, tempfilelist, (device-sycl)
// CHECK: 95: file-table-tform, {92, 94}, tempfiletable, (device-sycl)
// CHECK: 96: clang-offload-wrapper, {95}, object, (device-sycl)
// CHECK: 97: offload, "device-sycl (spir64-unknown-unknown)" {96}, object
// CHECK: 98: linker, {9, 19, 90, 97}, image, (host-sycl)

// RUN: %clang -### -fsycl -fno-sycl-rdc -c -fsycl-targets=spir64_gen --sysroot=%S/Inputs/SYCL-windows %t1.cpp 2>&1 | FileCheck -check-prefix=CHECK-EARLY %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -c -fsycl-targets=spir64_gen /clang:--sysroot=%S/Inputs/SYCL-windows %t1.cpp 2>&1 | FileCheck -check-prefix=CHECK-EARLY %s
// CHECK-EARLY: llvm-link{{.*}}
// CHECK-EARLY-NOT: -only-needed
// CHECK-EARLY: llvm-link{{.*}}-only-needed