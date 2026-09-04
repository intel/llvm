/// Verify how -W[no-]sycl-undefined-func-in-image maps to what actually
/// reaches sycl-post-link. The warning is emitted by sycl-post-link, and
/// -Wno-... is expected to silence it in every driver mode.
///
/// * Legacy offload path (default): the driver forwards the raw flag as
///   a `-suppress-undefined-func-warnings` occurrence to sycl-post-link.
/// * New offload path (`--offload-new-driver`): sycl-post-link runs in
///   process inside clang-linker-wrapper and reads its own arg list, so
///   the driver instead pushes the linker-wrapper native flag
///   `--sycl-suppress-undefined-func-warnings`.
///
/// Both %clangxx (default driver) and %clang_cl (Windows-style driver)
/// must forward the flag; %clang is redundant here since -fsycl already
/// implies C++ processing.

// -----------------------------------------------------------------------
// Legacy offload path.
// -----------------------------------------------------------------------

// RUN: %clangxx -### -fsycl -fsycl-targets=spir64 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-FLAG %s
// RUN: %clang_cl -### -fsycl -fsycl-targets=spir64 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-FLAG %s
// NO-FLAG: "{{.*}}sycl-post-link{{(\.exe)?}}"
// NO-FLAG-NOT: "-suppress-undefined-func-warnings"

// RUN: %clangxx -### -fsycl -fsycl-targets=spir64 -Wno-sycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WNO %s
// RUN: %clang_cl -### -fsycl -fsycl-targets=spir64 -Wno-sycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WNO %s
// WNO: "{{.*}}sycl-post-link{{(\.exe)?}}"
// WNO-SAME: "-suppress-undefined-func-warnings"

// RUN: %clangxx -### -fsycl -fsycl-targets=spir64 -Wsycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=W %s
// RUN: %clang_cl -### -fsycl -fsycl-targets=spir64 -Wsycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=W %s
// W: "{{.*}}sycl-post-link{{(\.exe)?}}"
// W-NOT: "-suppress-undefined-func-warnings"

// -Wno- then -W ==> warning stays enabled (last -W wins).
// RUN: %clangxx -### -fsycl -fsycl-targets=spir64 -Wno-sycl-undefined-func-in-image -Wsycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LAST-W %s
// RUN: %clang_cl -### -fsycl -fsycl-targets=spir64 -Wno-sycl-undefined-func-in-image -Wsycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LAST-W %s
// LAST-W: "{{.*}}sycl-post-link{{(\.exe)?}}"
// LAST-W-NOT: "-suppress-undefined-func-warnings"

// -W then -Wno- ==> warning suppressed (last -W wins).
// RUN: %clangxx -### -fsycl -fsycl-targets=spir64 -Wsycl-undefined-func-in-image -Wno-sycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LAST-WNO %s
// RUN: %clang_cl -### -fsycl -fsycl-targets=spir64 -Wsycl-undefined-func-in-image -Wno-sycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LAST-WNO %s
// LAST-WNO: "{{.*}}sycl-post-link{{(\.exe)?}}"
// LAST-WNO-SAME: "-suppress-undefined-func-warnings"

// -----------------------------------------------------------------------
// New offload path: sycl-post-link runs in process, so the driver must
// push the linker-wrapper native flag in addition to (or in place of)
// the sycl-post-link-options= forwarding.
// -----------------------------------------------------------------------

// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-FLAG-NEW %s
// RUN: %clang_cl -### -fsycl --offload-new-driver -fsycl-targets=spir64 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-FLAG-NEW %s
// NO-FLAG-NEW: clang-linker-wrapper
// NO-FLAG-NEW-NOT: "--sycl-suppress-undefined-func-warnings"

// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64 -Wno-sycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WNO-NEW %s
// RUN: %clang_cl -### -fsycl --offload-new-driver -fsycl-targets=spir64 -Wno-sycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WNO-NEW %s
// WNO-NEW: clang-linker-wrapper
// WNO-NEW-SAME: "--sycl-suppress-undefined-func-warnings"
