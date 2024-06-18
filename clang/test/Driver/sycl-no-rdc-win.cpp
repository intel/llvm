/// Tests for -fno-sycl-rdc on Windows
// REQUIRES: system-windows

// RUN: touch %t1.cpp
// RUN: touch %t2.cpp
// RUN: %clang -### -fsycl -fno-sycl-rdc --sysroot=%S/Inputs/SYCL %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc /clang:--sysroot=%S/Inputs/SYCL  %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s

// CHECK: 3: input, "{{.*}}1.cpp", c++, (device-sycl)
// CHECK: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHECK: 5: compiler, {4}, ir, (device-sycl)
// CHECK: 6: offload, "host-sycl (x86_64-pc-windows-msvc)" {2}, "device-sycl (spir64-unknown-unknown)" {5}, c++-cpp-output
// CHECK: 13: input, "{{.*}}2.cpp", c++, (device-sycl)
// CHECK: 14: preprocessor, {13}, c++-cpp-output, (device-sycl)
// CHECK: 15: compiler, {14}, ir, (device-sycl)
// CHECK: 16: offload, "host-sycl (x86_64-pc-windows-msvc)" {12}, "device-sycl (spir64-unknown-unknown)" {15}, c++-cpp-output
// CHECK: 20: input, "{{.*}}libsycl-crt{{.*}}", ir, (device-sycl)
// CHECK: 41: linker, {20, {{.*}}}, ir, (device-sycl)
// CHECK: 42: linker, {5, 41}, ir, (device-sycl)
// CHECK: 43: sycl-post-link, {42}, tempfiletable, (device-sycl)
// CHECK: 44: file-table-tform, {43}, tempfilelist, (device-sycl)
// CHECK: 45: llvm-spirv, {44}, tempfilelist, (device-sycl)
// CHECK: 46: file-table-tform, {43, 45}, tempfiletable, (device-sycl)
// CHECK: 47: clang-offload-wrapper, {46}, object, (device-sycl)
// CHECK: 48: offload, "device-sycl (spir64-unknown-unknown)" {47}, object
// CHECK: 49: linker, {15, 41}, ir, (device-sycl)
// CHECK: 50: sycl-post-link, {49}, tempfiletable, (device-sycl)
// CHECK: 51: file-table-tform, {50}, tempfilelist, (device-sycl)
// CHECK: 52: llvm-spirv, {51}, tempfilelist, (device-sycl)
// CHECK: 53: file-table-tform, {50, 52}, tempfiletable, (device-sycl)
// CHECK: 54: clang-offload-wrapper, {53}, object, (device-sycl)
// CHECK: 55: offload, "device-sycl (spir64-unknown-unknown)" {54}, object
// CHECK: 56: linker, {9, 19, 48, 55}, image, (host-sycl)

// RUN: %clang -### -fsycl -fno-sycl-rdc -c -fsycl-targets=spir64_gen --sysroot=%S/Inputs/SYCL %t1.cpp 2>&1 | FileCheck -check-prefix=CHECK-EARLY %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -c -fsycl-targets=spir64_gen /clang:--sysroot=%S/Inputs/SYCL %t1.cpp 2>&1 | FileCheck -check-prefix=CHECK-EARLY %s
// CHECK-EARLY: llvm-link{{.*}}
// CHECK-EARLY-NOT: -only-needed
// CHECK-EARLY: llvm-link{{.*}}-only-needed
