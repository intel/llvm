/// Tests for -fno-sycl-rdc on Windows
// REQUIRES: system-windows

/// Directory holding the compiler resource files, including the SYCL
/// compiler-rt builtins library (libclang_rt.builtins.bc).
// DEFINE: %{resource_dir} = %/S/Inputs/SYCL/lib/clang/resource_dir

// DEFINE: %{clang_install_dir} = %/S/Inputs/SYCL/bin

// RUN: touch %t1.cpp
// RUN: touch %t2.cpp
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -ccc-install-dir %{clang_install_dir} -resource-dir=%{resource_dir} %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -fsycl-instrument-device-code -ccc-install-dir %{clang_install_dir} -resource-dir=%{resource_dir}  %t1.cpp %t2.cpp 2>&1 -ccc-print-phases | FileCheck %s

// CHECK: 2: input, "{{.*}}1.cpp", c++, (device-sycl)
// CHECK: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHECK: 4: compiler, {3}, ir, (device-sycl)
// CHECK: 5: offload, "host-sycl (x86_64-pc-windows-msvc)" {1}, "device-sycl (spir64-unknown-unknown)" {4}, c++-cpp-output
// CHECK: 11: input, "{{.*}}2.cpp", c++, (device-sycl)
// CHECK: 12: preprocessor, {11}, c++-cpp-output, (device-sycl)
// CHECK: 13: compiler, {12}, ir, (device-sycl)
// CHECK: 14: offload, "host-sycl (x86_64-pc-windows-msvc)" {10}, "device-sycl (spir64-unknown-unknown)" {13}, c++-cpp-output
// CHECK: 18: input, "{{.*}}libsycl-crt{{.*}}", ir, (device-sycl)
// CHECK: 25: input, "{{.*}}libc{{.*}}", ir, (device-sycl)
// CHECK: 26: input, "{{.*}}libclang_rt.builtins{{.*}}", ir, (device-sycl)
// CHECK: 27: linker, {18, {{.*}}, 26}, ir, (device-sycl)
// CHECK: 28: linker, {4, 27}, ir, (device-sycl)
// CHECK: 29: sycl-post-link, {28}, tempfiletable, (device-sycl)
// CHECK: 30: file-table-tform, {29}, tempfilelist, (device-sycl)
// CHECK: 31: llvm-spirv, {30}, tempfilelist, (device-sycl)
// CHECK: 32: file-table-tform, {29, 31}, tempfiletable, (device-sycl)
// CHECK: 33: clang-offload-wrapper, {32}, object, (device-sycl)
// CHECK: 34: offload, "device-sycl (spir64-unknown-unknown)" {33}, object
// CHECK: 35: linker, {13, 27}, ir, (device-sycl)
// CHECK: 36: sycl-post-link, {35}, tempfiletable, (device-sycl)
// CHECK: 37: file-table-tform, {36}, tempfilelist, (device-sycl)
// CHECK: 38: llvm-spirv, {37}, tempfilelist, (device-sycl)
// CHECK: 39: file-table-tform, {36, 38}, tempfiletable, (device-sycl)
// CHECK: 40: clang-offload-wrapper, {39}, object, (device-sycl)
// CHECK: 41: offload, "device-sycl (spir64-unknown-unknown)" {40}, object
// CHECK: 42: linker, {8, 17, 34, 41}, image, (host-sycl)
// RUN: %clang -### -fsycl -fno-sycl-rdc -c -fsycl-targets=spir64_gen -ccc-install-dir %{clang_install_dir} %t1.cpp 2>&1 | FileCheck -check-prefix=CHECK-EARLY %s
// RUN: %clang_cl -### -fsycl -fno-sycl-rdc -c -fsycl-targets=spir64_gen -ccc-install-dir %{clang_install_dir} %t1.cpp 2>&1 | FileCheck -check-prefix=CHECK-EARLY %s
// CHECK-EARLY: llvm-link{{.*}}
// CHECK-EARLY-NOT: -only-needed
// CHECK-EARLY: llvm-link{{.*}}-only-needed
