// End-to-end test for --use-clang-sycl-linker flag with actual SYCL device code
// This test verifies that options flow correctly from the driver through
// clang-linker-wrapper to clang-sycl-linker.
//
// REQUIRES: x86-registered-target
// REQUIRES: spirv-registered-target
//
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -c %s -o %t.o
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown %t.o -o %t.out \
// RUN:     -### 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT
// CHECK-DEFAULT: clang-linker-wrapper
// CHECK-DEFAULT-NOT: "--use-clang-sycl-linker"
//
// Test that --use-clang-sycl-linker flag is passed through
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown %t.o -o %t.out \
// RUN:     -Xlinker --use-clang-sycl-linker -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-FLAG
// CHECK-FLAG: clang-linker-wrapper{{.*}} "--use-clang-sycl-linker"
//
// Test with backend options
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown %t.o -o %t.out \
// RUN:     -Xlinker --use-clang-sycl-linker \
// RUN:     -Xsycl-target-backend=spir64-unknown-unknown "-test-backend-opt" \
// RUN:     -### 2>&1 | FileCheck %s --check-prefix=CHECK-BACKEND
// CHECK-BACKEND: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spir64-unknown-unknown=-test-backend-opt"{{.*}} "--use-clang-sycl-linker"
//
// Test with linker options
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown %t.o -o %t.out \
// RUN:     -Xlinker --use-clang-sycl-linker \
// RUN:     -Xsycl-target-linker=spir64-unknown-unknown "-test-linker-opt" \
// RUN:     -### 2>&1 | FileCheck %s --check-prefix=CHECK-LINKER
// CHECK-LINKER: clang-linker-wrapper{{.*}} "--device-linker=sycl:spir64-unknown-unknown=-test-linker-opt"{{.*}} "--use-clang-sycl-linker"
//
// Test with both backend and linker options
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown %t.o -o %t.out \
// RUN:     -Xlinker --use-clang-sycl-linker \
// RUN:     -Xsycl-target-backend=spir64-unknown-unknown "-test-backend" \
// RUN:     -Xsycl-target-linker=spir64-unknown-unknown "-test-linker" \
// RUN:     -### 2>&1 | FileCheck %s --check-prefix=CHECK-BOTH
// CHECK-BOTH: clang-linker-wrapper{{.*}} "--device-compiler=sycl:spir64-unknown-unknown=-test-backend"{{.*}} "--device-linker=sycl:spir64-unknown-unknown=-test-linker"{{.*}} "--use-clang-sycl-linker"
//
// Test with --save-temps
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown %t.o -o %t.out \
// RUN:     -Xlinker --use-clang-sycl-linker -Xlinker --save-temps \
// RUN:     -### 2>&1 | FileCheck %s --check-prefix=CHECK-SAVE-TEMPS
// CHECK-SAVE-TEMPS: clang-linker-wrapper{{.*}} "--use-clang-sycl-linker"{{.*}} "--save-temps"

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;

  int *data = sycl::malloc_shared<int>(1, q);
  *data = 0;

  q.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {
    data[idx] = 42;
  }).wait();

  int result = *data;
  sycl::free(data, q);

  return !(result == 42);
}
