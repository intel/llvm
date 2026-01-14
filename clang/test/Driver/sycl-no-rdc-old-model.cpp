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
// CHECK: 37: linker, {18, {{.*}}}, ir, (device-sycl)
// CHECK: 38: linker, {4, 37}, ir, (device-sycl)
// CHECK: 39: sycl-post-link, {38}, tempfiletable, (device-sycl)
// CHECK: 40: file-table-tform, {39}, tempfilelist, (device-sycl)
// CHECK: 41: llvm-spirv, {40}, tempfilelist, (device-sycl)
// CHECK: 42: file-table-tform, {39, 41}, tempfiletable, (device-sycl)
// CHECK: 43: clang-offload-wrapper, {42}, object, (device-sycl)
// CHECK: 44: offload, "device-sycl (spir64-unknown-unknown)" {43}, object
// CHECK: 45: linker, {13, 37}, ir, (device-sycl)
// CHECK: 46: sycl-post-link, {45}, tempfiletable, (device-sycl)
// CHECK: 47: file-table-tform, {46}, tempfilelist, (device-sycl)
// CHECK: 48: llvm-spirv, {47}, tempfilelist, (device-sycl)
// CHECK: 49: file-table-tform, {46, 48}, tempfiletable, (device-sycl)
// CHECK: 50: clang-offload-wrapper, {49}, object, (device-sycl)
// CHECK: 51: offload, "device-sycl (spir64-unknown-unknown)" {50}, object
// CHECK: 52: linker, {8, 17, 44, 51}, image, (host-sycl)
