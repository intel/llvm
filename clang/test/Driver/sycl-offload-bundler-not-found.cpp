/// Test that the driver reports an error instead of crashing when
/// clang-offload-bundler cannot be found next to the driver while it inspects
/// object and archive inputs for offload device code.

/// Simulate an installation directory that does not contain the bundler.
// RUN: rm -rf %t && mkdir -p %t/bin
// RUN: touch %t/obj1.o %t/obj2.o
// RUN: llvm-ar cr %t/lib.a %t/obj1.o
// RUN: llvm-ar cr %t/dep.lib %t/obj1.o

/// Object input with -fsycl-force-target.
// RUN: not %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --no-offload-new-driver -fsycl-targets=spir64_gen \
// RUN:   -fsycl-force-target=spir64 -ccc-install-dir %t/bin %t/obj1.o 2>&1 \
// RUN:   | FileCheck %s -DFILE=obj1.o

/// Multiple object inputs for native_cpu.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   -fsycl-targets=native_cpu -ccc-install-dir %t/bin \
// RUN:   %t/obj1.o %t/obj2.o 2>&1 \
// RUN:   | FileCheck %s -DFILE=obj1.o

/// Static archive input with the old offloading model.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --no-offload-new-driver -fno-sycl-rdc -ccc-install-dir %t/bin \
// RUN:   %t/lib.a 2>&1 \
// RUN:   | FileCheck %s -DFILE=lib.a

/// Import library input passed to clang-cl.
// RUN: not %clang_cl -### --target=x86_64-pc-windows-msvc -fsycl \
// RUN:   -fsycl-allow-device-image-dependencies -ccc-install-dir %t/bin \
// RUN:   %s %t/dep.lib 2>&1 \
// RUN:   | FileCheck %s -DFILE=dep.lib

// CHECK-NOT: PLEASE submit a bug report
// CHECK: error: cannot find 'clang-offload-bundler' in '{{.*}}bin', which is required to inspect '{{.*}}[[FILE]]' for offload device code
// CHECK-NOT: cannot find 'clang-offload-bundler'
// CHECK-NOT: PLEASE submit a bug report
