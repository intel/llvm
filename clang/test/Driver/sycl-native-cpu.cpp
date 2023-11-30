// RUN: %clangxx -fsycl-device-only -fsycl-targets=native_cpu %s -### 2>&1 | FileCheck %s
// RUN: %clangxx -fsycl-device-only -fsycl-targets=native_cpu -target aarch64-unknown-linux-gnu %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-AARCH64


// checks that the host and device triple are the same, and that the sycl-native-cpu LLVM option is set
// CHECK: clang{{.*}}"-triple" "[[TRIPLE:.*]]"{{.*}}"-aux-triple" "[[TRIPLE]]"{{.*}}"-fsycl-is-native-cpu"{{.*}}"-D" "__SYCL_NATIVE_CPU__"

// checks that the target triples are set correctly when the target is set explicitly
// CHECK-AARCH64: clang{{.*}}"-triple" "aarch64-unknown-linux-gnu"{{.*}}"-aux-triple" "aarch64-unknown-linux-gnu"{{.*}}"-fsycl-is-native-cpu"{{.*}}"-D" "__SYCL_NATIVE_CPU__"

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=native_cpu -g %s 2>&1 | FileCheck -check-prefix=CHECK-LINUX %s
// CHECK-LINUX: {{.*}}"-fsycl-is-device"{{.*}}"-dwarf-version=[[DVERSION:.*]]" "-debugger-tuning=gdb"
// CHECK-LINUX-DAG: {{.*}}"-fsycl-is-host"{{.*}}"-dwarf-version=[[DVERSION]]" "-debugger-tuning=gdb"
// CHECK-LINUX-NOT: codeview

// RUN:   %clang -### -target x86_64-windows-msvc -fsycl -fsycl-targets=native_cpu -g %s 2>&1 | FileCheck -check-prefix=CHECK-WIN %s
// CHECK-WIN: {{.*}}"-fsycl-is-device"{{.*}}"-gcodeview"
// CHECK-WIN-DAG: {{.*}}"-fsycl-is-host"{{.*}}"-gcodeview"
// CHECK-WIN-NOT: dwarf

// checks that -sycl-opt is not enabled by default on NativeCPU so that the full llvm optimization is enabled
// RUN:   %clang -fsycl -fsycl-targets=native_cpu -### %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// CHECK-OPTS-NOT: -sycl-opt

