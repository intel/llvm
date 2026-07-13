/// Test that -fsycl-allow-device-image-dependencies on Windows does NOT add
/// /OPT:NOREF (the fix uses /INCLUDE: for user import libs instead).

// REQUIRES: system-windows

/// Check that /OPT:NOREF is NOT passed even when both -fsycl and
/// -fsycl-allow-device-image-dependencies are specified.
// RUN: %clang_cl -fsycl --offload-new-driver /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### -- %s 2>&1 \
// RUN:  | FileCheck -check-prefix CHECK_NO_OPT_NOREF %s
// CHECK_NO_OPT_NOREF-NOT: "/OPT:NOREF"
