///
/// Perform several driver tests for SYCL device libraries on Windows
///

// REQUIRES: system-windows

/// Directory holding the compiler resource files, including the SYCL
/// compiler-rt builtins library (libclang_rt.builtins.bc).
// DEFINE: %{resource_dir} = %/S/Inputs/SYCL/lib/clang/resource_dir

/// ###########################################################################

/// test behavior of device library default link
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -resource-dir=%{resource_dir} -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// SYCL_DEVICE_LIB_LINK_DEFAULT: llvm-link{{.*}} "{{.*}}lib\\libsycl-crt.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}lib\\libsycl-cmath.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}lib\\libsycl-imf.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libclang_rt.builtins.bc"

/// ###########################################################################

/// test behavior of disabling all device libraries
// RUN: %clangxx -fsycl %s --no-offloadlib --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB: {{.*}}clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB-NOT: libsycl-cmath.bc

/// ###########################################################################

/// test llvm-link behavior for linking device libraries
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -resource-dir=%{resource_dir} -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_DEVICE_LIB
// RUN: %clangxx -fsycl -save-temps %s --sysroot=%S/Inputs/SYCL -resource-dir=%{resource_dir} -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_DEVICE_LIB
// SYCL_LLVM_LINK_DEVICE_LIB: llvm-link{{.*}}  "{{.*}}.bc" "-o" "{{.*}}.bc" "--suppress-warnings"
// SYCL_LLVM_LINK_DEVICE_LIB: llvm-link{{.*}} "--only-needed"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-crt.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-cmath.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-imf.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libclang_rt.builtins.bc"

/// ###########################################################################
/// test clang-cl behavior for linking sycl-devicelib-host.lib by default
// RUN: %clang_cl -fsycl %s /winsysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_HOST_LIB
// SYCL_DEVICE_HOST_LIB: {{.*}} "--dependent-lib=sycl-devicelib-host" {{.*}}
