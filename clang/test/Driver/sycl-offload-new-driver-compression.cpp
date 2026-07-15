/// Check that '--offload-compress' and '--offload-compression-level=' are
/// forwarded to clang-linker-wrapper under the new offloading driver via the
/// existing addOffloadCompressArgs() plumbing (as '--compress' and
/// '--compression-level='). The wrapper consumes them in
/// wrapSYCLBinariesFromFile to zstd-compress each SYCL device image and tag
/// it with BIF_Compressed.

// RUN: %clangxx -### -fsycl --offload-new-driver --offload-compress \
// RUN:     --offload-compression-level=3 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-COMPRESS
// CHECK-COMPRESS: {{.*}}clang-linker-wrapper{{.*}}"--compress"{{.*}}"--compression-level=3"

// RUN: %clangxx -### -fsycl --offload-new-driver %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-COMPRESS
// CHECK-NO-COMPRESS-NOT: {{.*}}clang-linker-wrapper{{.*}}"--compress"

// --no-offload-compress overrides an earlier --offload-compress.
// RUN: %clangxx -### -fsycl --offload-new-driver --offload-compress \
// RUN:     --no-offload-compress %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-COMPRESS
