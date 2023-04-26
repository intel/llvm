// RUN: %clangxx -fsycl-device-only -fsycl-native-cpu %s -### 2>&1 | FileCheck %s
// RUN: %clangxx -fsycl-device-only -fsycl-native-cpu -target aarch64-unknown-linux-gnu %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-AARCH64


// checks that the host and device triple are the same, and that the sycl-host-compilation LLVM option is set
// CHECK: clang{{.*}}"-triple" "[[TRIPLE:.*]]"{{.*}}"-aux-triple" "[[TRIPLE]]"{{.*}}"-mllvm" "-sycl-native-cpu"{{.*}}"-D" "__SYCL_NATIVE_CPU__"

// checks that the target triples are set correctly when the target is set explicitly
// CHECK-AARCH64: clang{{.*}}"-triple" "aarch64-unknown-linux-gnu"{{.*}}"-aux-triple" "aarch64-unknown-linux-gnu"{{.*}}"-mllvm" "-sycl-native-cpu"{{.*}}"-D" "__SYCL_NATIVE_CPU__"
