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
// CHECK: 29: linker, {18, {{.*}}}, ir, (device-sycl)
// CHECK: 30: linker, {4, 29}, ir, (device-sycl)
// CHECK: 31: sycl-post-link, {30}, tempfiletable, (device-sycl)
// CHECK: 32: file-table-tform, {31}, tempfilelist, (device-sycl)
// CHECK: 33: llvm-spirv, {32}, tempfilelist, (device-sycl)
// CHECK: 34: file-table-tform, {31, 33}, tempfiletable, (device-sycl)
// CHECK: 35: clang-offload-wrapper, {34}, object, (device-sycl)
// CHECK: 36: offload, "device-sycl (spir64-unknown-unknown)" {35}, object
// CHECK: 37: linker, {13, 29}, ir, (device-sycl)
// CHECK: 38: sycl-post-link, {37}, tempfiletable, (device-sycl)
// CHECK: 39: file-table-tform, {38}, tempfilelist, (device-sycl)
// CHECK: 40: llvm-spirv, {39}, tempfilelist, (device-sycl)
// CHECK: 41: file-table-tform, {38, 40}, tempfiletable, (device-sycl)
// CHECK: 42: clang-offload-wrapper, {41}, object, (device-sycl)
// CHECK: 43: offload, "device-sycl (spir64-unknown-unknown)" {42}, object
// CHECK: 44: linker, {8, 17, 36, 43}, image, (host-sycl)
