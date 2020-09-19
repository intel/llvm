// REQUIRES: clang-driver

// RUN: %clang -### -fsycl -c -target x86_64-unknown-windows-msvc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DEFAULT %s
// RUN: %clangxx -### -fsycl -c -target x86_64-unknown-windows-msvc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DEFAULT %s
// RUN: %clang_cl -### -fsycl -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DEFAULT %s
// RUN: %clang_cl -### -MD -fsycl -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DEFAULT %s
// RUN: %clang_cl -### -MDd -fsycl -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DEFAULT %s
// CHK-DEFAULT: "-D_MT" "-D_DLL"
// CHK-DEFAULT: "--dependent-lib=msvcrt{{d*}}"

// RUN: %clang_cl -### -MT -fsycl -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ERROR %s
// RUN: %clang_cl -### -MTd -fsycl -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ERROR %s
// CHK-ERROR: option 'MT{{d*}}' unsupported with DPC++
