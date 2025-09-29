// Test -ftarget-export-symbols behavior

// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl \
// RUN:    -fsycl-targets=spir64_gen -ftarget-export-symbols %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_EXPORT %s
// RUN: %clang_cl -### --target=x86_64-pc-windows-msvc -fsycl \
// RUN:     -fsycl-targets=spir64_gen -ftarget-export-symbols %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_EXPORT %s
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl \
// RUN:    -fsycl-targets=spir64_gen -ftarget-export-symbols \
// RUN:    -fno-target-export-symbols %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_TARGET_EXPORT %s

// TARGET_EXPORT: ocloc{{.*}} "-output"
// TARGET_EXPORT: "-options" "-library-compilation"
// NO_TARGET_EXPORT-NOT: "-library-compilation"

// 'unused' for non-spir64_gen targets
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl \
// RUN:    -fsycl-targets=spir64 -ftarget-export-symbols %s 2>&1 \
// RUN:   | FileCheck -check-prefix=UNUSED %s
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl \
// RUN:    -fsycl-targets=spir64_x86_64 -ftarget-export-symbols %s 2>&1 \
// RUN:   | FileCheck -check-prefix=UNUSED %s

// UNUSED: argument unused during compilation: '-ftarget-export-symbols'
