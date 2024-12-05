// Test for default llvm-spirv options

// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT

// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64_fpga-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fintelfpga %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64_fpga-unknown-unknown -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fintelfpga -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64_fpga-unknown-unknown -Xssimulation %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fintelfpga -Xssimulation %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64_fpga-unknown-unknown -Xsemulator %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fintelfpga -Xsemulator %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64_gen-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64_gen-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64_x86_64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK-DEFAULT

// CHECK-DEFAULT: llvm-spirv{{.*}}-spirv-debug-info-version=nonsemantic-shader-200
// CHECK-DEFAULT-NOT: -ocl-100

// CHECL-FPGA-HW: llvm-spirv{{.*}}-ocl-100
// CHECK-FPGA-HW-NOT: spirv-debug-info-version=nonsemantic-shader-200

