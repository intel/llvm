/// Verify that -W[no-]sycl-undefined-func-in-image is forwarded correctly to
/// sycl-post-link under both the default clang driver and the clang-cl
/// driver. The clang-cl coverage exists because the underlying Options.td
/// defs need Visibility<[ClangOption, CLOption]> for the driver's
/// getLastArg(OPT_Wsycl_..., OPT_Wno_sycl_...) to see the arg under
/// --driver-mode=cl.

// RUN: %clang -### -fsycl -fsycl-targets=spir64 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-FLAG %s
// RUN: %clangxx -### -fsycl -fsycl-targets=spir64 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-FLAG %s
// RUN: %clang_cl -### -fsycl -fsycl-targets=spir64 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-FLAG %s

// Baseline: without -Wno-..., the sycl-post-link invocation must NOT
// carry -suppress-undefined-func-warnings.
// NO-FLAG: "{{.*}}sycl-post-link{{(\.exe)?}}"
// NO-FLAG-NOT: "-suppress-undefined-func-warnings"

// RUN: %clang -### -fsycl -fsycl-targets=spir64 -Wno-sycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WNO %s
// RUN: %clangxx -### -fsycl -fsycl-targets=spir64 -Wno-sycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WNO %s
// RUN: %clang_cl -### -fsycl -fsycl-targets=spir64 -Wno-sycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WNO %s

// With -Wno-...: the sycl-post-link invocation must carry the flag.
// WNO: "{{.*}}sycl-post-link{{(\.exe)?}}"
// WNO-SAME: "-suppress-undefined-func-warnings"

// RUN: %clang -### -fsycl -fsycl-targets=spir64 -Wsycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=W %s
// RUN: %clangxx -### -fsycl -fsycl-targets=spir64 -Wsycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=W %s
// RUN: %clang_cl -### -fsycl -fsycl-targets=spir64 -Wsycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=W %s

// Explicit -W... (the default, no "no-"): must NOT forward the flag.
// W: "{{.*}}sycl-post-link{{(\.exe)?}}"
// W-NOT: "-suppress-undefined-func-warnings"

// RUN: %clang -### -fsycl -fsycl-targets=spir64 -Wno-sycl-undefined-func-in-image -Wsycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LAST-W %s
// RUN: %clang_cl -### -fsycl -fsycl-targets=spir64 -Wno-sycl-undefined-func-in-image -Wsycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LAST-W %s

// -Wno-... followed by -W...: last-W wins, must NOT forward the flag.
// LAST-W: "{{.*}}sycl-post-link{{(\.exe)?}}"
// LAST-W-NOT: "-suppress-undefined-func-warnings"

// RUN: %clang -### -fsycl -fsycl-targets=spir64 -Wsycl-undefined-func-in-image -Wno-sycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LAST-WNO %s
// RUN: %clang_cl -### -fsycl -fsycl-targets=spir64 -Wsycl-undefined-func-in-image -Wno-sycl-undefined-func-in-image %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LAST-WNO %s

// -W... followed by -Wno-...: last-W wins, MUST forward the flag.
// LAST-WNO: "{{.*}}sycl-post-link{{(\.exe)?}}"
// LAST-WNO-SAME: "-suppress-undefined-func-warnings"
