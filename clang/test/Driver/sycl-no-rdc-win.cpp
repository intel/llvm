/// Tests for -fno-sycl-rdc on Windows
// REQUIRES: system-windows

// RUN: touch %t1.cpp
// RUN: touch %t2.cpp
// RUN: %clang -### -fsycl -fno-sycl-rdc --sysroot=%S/Inputs/SYCL %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc /clang:--sysroot=%S/Inputs/SYCL  %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s

// CHECK: 2: input, "{{.*}}1.cpp", c++, (device-sycl)
// CHECK: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHECK: 4: compiler, {3}, ir, (device-sycl)
// CHECK: 5: offload, "host-sycl (x86_64-pc-windows-msvc)" {1}, "device-sycl (spir64-unknown-unknown)" {4}, c++-cpp-output
// CHECK: 11: input, "{{.*}}2.cpp", c++, (device-sycl)
// CHECK: 12: preprocessor, {11}, c++-cpp-output, (device-sycl)
// CHECK: 13: compiler, {12}, ir, (device-sycl)
// CHECK: 14: offload, "host-sycl (x86_64-pc-windows-msvc)" {10}, "device-sycl (spir64-unknown-unknown)" {13}, c++-cpp-output
// CHECK: 18: input, "{{.*}}libsycl-crt{{.*}}", ir, (device-sycl)
// CHECK: 39: linker, {18, {{.*}}}, ir, (device-sycl)
// CHECK: 40: linker, {4, 39}, ir, (device-sycl)
// CHECK: 41: sycl-post-link, {40}, tempfiletable, (device-sycl)
// CHECK: 42: file-table-tform, {41}, tempfilelist, (device-sycl)
// CHECK: 43: llvm-spirv, {42}, tempfilelist, (device-sycl)
// CHECK: 44: file-table-tform, {41, 43}, tempfiletable, (device-sycl)
// CHECK: 45: clang-offload-wrapper, {44}, object, (device-sycl)
// CHECK: 46: offload, "device-sycl (spir64-unknown-unknown)" {45}, object
// CHECK: 47: linker, {13, 39}, ir, (device-sycl)
// CHECK: 48: sycl-post-link, {47}, tempfiletable, (device-sycl)
// CHECK: 49: file-table-tform, {48}, tempfilelist, (device-sycl)
// CHECK: 50: llvm-spirv, {49}, tempfilelist, (device-sycl)
// CHECK: 51: file-table-tform, {48, 50}, tempfiletable, (device-sycl)
// CHECK: 52: clang-offload-wrapper, {51}, object, (device-sycl)
// CHECK: 53: offload, "device-sycl (spir64-unknown-unknown)" {52}, object
// CHECK: 54: linker, {8, 17, 46, 53}, image, (host-sycl)

// RUN: %clang -### -fsycl -fno-sycl-rdc -c -fsycl-targets=spir64_gen --sysroot=%S/Inputs/SYCL %t1.cpp 2>&1 | FileCheck -check-prefix=CHECK-EARLY %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -c -fsycl-targets=spir64_gen /clang:--sysroot=%S/Inputs/SYCL %t1.cpp 2>&1 | FileCheck -check-prefix=CHECK-EARLY %s
// CHECK-EARLY: llvm-link{{.*}}
// CHECK-EARLY-NOT: -only-needed
// CHECK-EARLY: llvm-link{{.*}}-only-needed
