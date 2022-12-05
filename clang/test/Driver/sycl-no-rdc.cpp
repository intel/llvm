/// Tests for -fno-sycl-rdc
// RUN: touch %t1.cpp
// RUN: touch %t2.cpp
// RUN: %clang -### -fsycl -fno-sycl-rdc --sysroot=%S/Inputs/SYCL %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck -check-prefix=CHECK-PHASES %s
// RUN: %clang -### -fsycl -fno-sycl-rdc --sysroot=%S/Inputs/SYCL %t1.cpp %t2.cpp 2>&1 | FileCheck -check-prefix=CHECK-COMMAND %s

// CHECK-PHASES: 3: input, "{{.*}}1.cpp", c++, (device-sycl)
// CHECK-PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHECK-PHASES: 5: compiler, {4}, ir, (device-sycl)
// CHECK-PHASES: 13: input, "{{.*}}2.cpp", c++, (device-sycl)
// CHECK-PHASES: 14: preprocessor, {13}, c++-cpp-output, (device-sycl)
// CHECK-PHASES: 15: compiler, {14}, ir, (device-sycl)

// CHECK-PHASES: 21: input, {{.*}}libsycl-crt{{.*}}, object
// CHECK-PHASES: 22: clang-offload-unbundler, {21}, object
// CHECK-PHASES: 23: offload, " (spir64-unknown-unknown)" {22}, object
// CHECK-PHASES: 81: linker, {23, {{.*}}}, ir, (device-sycl)
// CHECK-PHASES: 82: linker, {5, 81}, ir, (device-sycl)
// CHECK-PHASES: 83: sycl-post-link, {82}, tempfiletable, (device-sycl)
// CHECK-PHASES: 84: file-table-tform, {83}, tempfilelist, (device-sycl)
// CHECK-PHASES: 85: llvm-spirv, {84}, tempfilelist, (device-sycl)
// CHECK-PHASES: 86: file-table-tform, {83, 85}, tempfiletable, (device-sycl)
// CHECK-PHASES: 87: clang-offload-wrapper, {86}, object, (device-sycl)

// CHECK-PHASES: 88: linker, {15, 81}, ir, (device-sycl)
// CHECK-PHASES: 89: sycl-post-link, {88}, tempfiletable, (device-sycl)
// CHECK-PHASES: 90: file-table-tform, {89}, tempfilelist, (device-sycl)
// CHECK-PHASES: 91: llvm-spirv, {90}, tempfilelist, (device-sycl)
// CHECK-PHASES: 92: file-table-tform, {89, 91}, tempfiletable, (device-sycl)
// CHECK-PHASES: 93: clang-offload-wrapper, {92}, object, (device-sycl)

// CHECK-PHASES: 94: offload, "host-sycl (x86_64-unknown-linux-gnu)" {{{.*}}}, "device-sycl (spir64-unknown-unknown)" {87}, "device-sycl (spir64-unknown-unknown)" {93}, image

// CHECK-COMMAND: sycl-post-link{{.*}}-sycl-rdc=false
