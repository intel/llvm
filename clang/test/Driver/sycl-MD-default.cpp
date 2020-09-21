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
// CHK-DEFAULT-NOT: "-fsycl-is-device" {{.*}} "-D_MT" "-D_DLL"
// CHK-DEFAULT: "-D_MT" "-D_DLL" "--dependent-lib=msvcrt{{d*}}" {{.*}} "-fsycl-is-host"

// RUN: %clang_cl -### -MT -fsycl -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ERROR %s
// RUN: %clang_cl -### -MTd -fsycl -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ERROR %s
// CHK-ERROR: option 'MT{{d*}}' unsupported with DPC++
