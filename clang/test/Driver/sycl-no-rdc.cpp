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
// CHECK: 20: input, "{{.*}}libsycl-crt.bc", ir, (device-sycl)
// CHECK: 40: linker, {20, {{.*}}}, ir, (device-sycl)
// CHECK: 41: linker, {5, 40}, ir, (device-sycl)
// CHECK: 42: sycl-post-link, {41}, tempfiletable, (device-sycl)
// CHECK: 43: file-table-tform, {42}, tempfilelist, (device-sycl)
// CHECK: 44: llvm-spirv, {43}, tempfilelist, (device-sycl)
// CHECK: 45: file-table-tform, {42, 44}, tempfiletable, (device-sycl)
// CHECK: 46: clang-offload-wrapper, {45}, object, (device-sycl)
// CHECK: 47: offload, "device-sycl (spir64-unknown-unknown)" {46}, object
// CHECK: 48: linker, {15, 40}, ir, (device-sycl)
// CHECK: 49: sycl-post-link, {48}, tempfiletable, (device-sycl)
// CHECK: 50: file-table-tform, {49}, tempfilelist, (device-sycl)
// CHECK: 51: llvm-spirv, {50}, tempfilelist, (device-sycl)
// CHECK: 52: file-table-tform, {49, 51}, tempfiletable, (device-sycl)
// CHECK: 53: clang-offload-wrapper, {52}, object, (device-sycl)
// CHECK: 54: offload, "device-sycl (spir64-unknown-unknown)" {53}, object
// CHECK: 55: linker, {9, 19, 47, 54}, image, (host-sycl)
