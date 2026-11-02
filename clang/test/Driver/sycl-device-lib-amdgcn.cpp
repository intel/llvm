// Tests specific to `-fsycl-targets=amdgcn-amd-amdhsa`
// Verify that the correct devicelib linking actions are spawned by the driver.
// Check also if the correct warnings are generated.

// UNSUPPORTED: system-windows

// DEFINE: %{clang_install_dir} = %/S/Inputs/SYCL/bin

// Check that llvm-link uses the "-only-needed" flag.
// Not using the flag breaks kernel bundles.
// RUN: %clangxx -### -fsycl -fsycl-targets=amdgcn-amd-amdhsa -fno-sycl-libspirv -ccc-install-dir %{clang_install_dir} \
// RUN: -Xsycl-target-backend --offload-arch=gfx908 --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ONLY-NEEDED %s

// CHK-ONLY-NEEDED: llvm-link"{{.*}}"--only-needed"{{.*}}"{{.*}}devicelib-amdgcn-amd-amdhsa.bc"{{.*}}
