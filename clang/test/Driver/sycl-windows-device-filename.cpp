// Test valid file names from -device values for GPU
// REQUIRES: system-windows

// DEFINE: %{clang_install_dir} = %/S/Inputs/SYCL/bin

// RUN: %clang -### --target=x86_64-pc-windows-msvc -fsycl \
// RUN:   -fsycl-targets=intel_gpu_pvc --offload-new-driver \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   -Xsycl-target-backend "-device arch1:arch2" %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK_COLON

// CHECK_COLON: sycl-windows-device-filename-arch1@arch2
// CHECK_COLON: arch=arch1:arch2
// CHECK_COLON-NOT: sycl-windows-device-filename-arch1:arch2

// RUN: %clang -### --target=x86_64-pc-windows-msvc -fsycl \
// RUN:   -fsycl-targets=intel_gpu_pvc --offload-new-driver \
// RUN:   -ccc-install-dir %{clang_install_dir} \
// RUN:   -Xsycl-target-backend "-device *" %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK_STAR

// CHECK_STAR: sycl-windows-device-filename-@
// CHECK_STAR: arch=*
// CHECK_STAR-NOT: sycl-windows-device-filename-*
