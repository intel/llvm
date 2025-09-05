// RUN: %clang -### -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-DEFAULT %s

// CHECK-DEFAULT: "-enable-memcpyopt-without-libcalls"
