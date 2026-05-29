/// Test that /OPT:NOREF is correctly added when -fsycl-allow-device-image-dependencies
/// is specified on Windows. This prevents the MSVC linker from removing device image
/// DLLs that have no directly referenced host symbols.

// REQUIRES: system-windows

/// Check that /OPT:NOREF is passed to the MSVC linker when both -fsycl and
/// -fsycl-allow-device-image-dependencies are specified.
// RUN: %clang_cl -fsycl --offload-new-driver /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### -- %s 2>&1 \
// RUN:  | FileCheck -check-prefix CHECK_OPT_NOREF %s
// CHECK_OPT_NOREF: link.exe{{.*}} "/OPT:NOREF"

/// Check that /OPT:NOREF is NOT passed when -fsycl-allow-device-image-dependencies
/// is not set (even with -fsycl present).
// RUN: %clang_cl -fsycl --offload-new-driver /clang:--sysroot=%S/Inputs/SYCL \
// RUN:  /O2 -### -- %s 2>&1 | FileCheck -check-prefix CHECK_NO_OPT_NOREF %s
// CHECK_NO_OPT_NOREF-NOT: "/OPT:NOREF"

/// Check that /OPT:NOREF is NOT passed when -fsycl is not set (even with
/// -fsycl-allow-device-image-dependencies present).
// RUN: %clang_cl --offload-new-driver /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### -- %s 2>&1 \
// RUN:  | FileCheck -check-prefix CHECK_NO_FSYCL %s
// CHECK_NO_FSYCL-NOT: "/OPT:NOREF"

/// Check that our /OPT:NOREF comes after user-specified /OPT:REF, ensuring
/// our flag overrides the user's optimization setting for correctness.
// RUN: %clang_cl -fsycl --offload-new-driver /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 \
// RUN:          -### %s /link /OPT:REF 2>&1 \
// RUN:  | FileCheck -check-prefix CHECK_OPT_ORDER %s
// CHECK_OPT_ORDER: clang-linker-wrapper
// CHECK_OPT_ORDER-SAME: "/OPT:REF"
// CHECK_OPT_ORDER-SAME: "/OPT:NOREF"
