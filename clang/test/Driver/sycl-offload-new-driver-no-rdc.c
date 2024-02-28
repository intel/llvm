// Verify we pass -no-sycl-rdc to clang-linker-wrapper

// RUN: %clangxx -### -fsycl -fsycl-targets=spir64 --offload-new-driver %s -fno-sycl-rdc 2>&1 | FileCheck %s

// CHECK: clang-linker-wrapper{{.*}} {{.*}}"-no-sycl-rdc" "--linker-path={{.*}}"
