/// Tests for -fno-sycl-rdc
// UNSUPPORTED: system-windows

// RUN: touch %t1.cpp
// RUN: touch %t2.cpp
// RUN: %clang -### -fsycl --no-offload-new-driver -fno-sycl-rdc --sysroot=%S/Inputs/SYCL %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s

// CHECK: 2: input, "{{.*}}1.cpp", c++, (device-sycl)
// CHECK: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHECK: 4: compiler, {3}, ir, (device-sycl)
// CHECK: 11: input, "{{.*}}2.cpp", c++, (device-sycl)
// CHECK: 12: preprocessor, {11}, c++-cpp-output, (device-sycl)
// CHECK: 13: compiler, {12}, ir, (device-sycl)
// CHECK: 18: input, "{{.*}}libsycl-crt.bc", ir, (device-sycl)
// CHECK: 38: linker, {18, {{.*}}}, ir, (device-sycl)
// CHECK: 39: linker, {4, 38}, ir, (device-sycl)
// CHECK: 40: sycl-post-link, {39}, tempfiletable, (device-sycl)
// CHECK: 41: file-table-tform, {40}, tempfilelist, (device-sycl)
// CHECK: 42: llvm-spirv, {41}, tempfilelist, (device-sycl)
// CHECK: 43: file-table-tform, {40, 42}, tempfiletable, (device-sycl)
// CHECK: 44: clang-offload-wrapper, {43}, object, (device-sycl)
// CHECK: 45: offload, "device-sycl (spir64-unknown-unknown)" {44}, object
// CHECK: 46: linker, {13, 38}, ir, (device-sycl)
// CHECK: 47: sycl-post-link, {46}, tempfiletable, (device-sycl)
// CHECK: 48: file-table-tform, {47}, tempfilelist, (device-sycl)
// CHECK: 49: llvm-spirv, {48}, tempfilelist, (device-sycl)
// CHECK: 50: file-table-tform, {47, 49}, tempfiletable, (device-sycl)
// CHECK: 51: clang-offload-wrapper, {50}, object, (device-sycl)
// CHECK: 52: offload, "device-sycl (spir64-unknown-unknown)" {51}, object
// CHECK: 53: linker, {8, 17, 45, 52}, image, (host-sycl)
