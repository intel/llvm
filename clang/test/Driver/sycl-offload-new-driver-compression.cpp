/// Check that '--offload-compress' and '--offload-compression-level=' are
/// forwarded to clang-linker-wrapper under the new offloading driver.
/// The wrapper consumes them in wrapSYCLBinariesFromFile to zstd-compress
/// each SYCL device image and tag it with BIF_CompressedNone.

// RUN: %clangxx -### -fsycl --offload-new-driver --offload-compress \
// RUN:     --offload-compression-level=3 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-COMPRESS
// CHECK-COMPRESS: {{.*}}clang-linker-wrapper{{.*}}"--offload-compress"{{.*}}"--offload-compression-level=3"

// Without --offload-compress, neither flag should reach the linker-wrapper.
// RUN: %clangxx -### -fsycl --offload-new-driver %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-COMPRESS
// RUN: %clangxx -### -fsycl --offload-new-driver \
// RUN:     --offload-compression-level=3 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-COMPRESS
// CHECK-NO-COMPRESS-NOT: {{.*}}clang-linker-wrapper{{.*}}"--offload-compress"
// CHECK-NO-COMPRESS-NOT: {{.*}}clang-linker-wrapper{{.*}}"--offload-compression-level={{.*}}"

// --no-offload-compress overrides an earlier --offload-compress.
// RUN: %clangxx -### -fsycl --offload-new-driver --offload-compress \
// RUN:     --no-offload-compress %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-COMPRESS
