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
// CHECK: 30: linker, {18, {{.*}}}, ir, (device-sycl)
// CHECK: 31: linker, {4, 30}, ir, (device-sycl)
// CHECK: 32: sycl-post-link, {31}, tempfiletable, (device-sycl)
// CHECK: 33: file-table-tform, {32}, tempfilelist, (device-sycl)
// CHECK: 34: llvm-spirv, {33}, tempfilelist, (device-sycl)
// CHECK: 35: file-table-tform, {32, 34}, tempfiletable, (device-sycl)
// CHECK: 36: clang-offload-wrapper, {35}, object, (device-sycl)
// CHECK: 37: offload, "device-sycl (spir64-unknown-unknown)" {36}, object
// CHECK: 38: linker, {13, 30}, ir, (device-sycl)
// CHECK: 39: sycl-post-link, {38}, tempfiletable, (device-sycl)
// CHECK: 40: file-table-tform, {39}, tempfilelist, (device-sycl)
// CHECK: 41: llvm-spirv, {40}, tempfilelist, (device-sycl)
// CHECK: 42: file-table-tform, {39, 41}, tempfiletable, (device-sycl)
// CHECK: 43: clang-offload-wrapper, {42}, object, (device-sycl)
// CHECK: 44: offload, "device-sycl (spir64-unknown-unknown)" {43}, object
// CHECK: 45: linker, {8, 17, 37, 44}, image, (host-sycl)
