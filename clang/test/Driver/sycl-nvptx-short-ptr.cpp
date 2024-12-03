// REQUIRES: nvptx-registered-target

// RUN: %clang -### -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-DEFAULT %s

// RUN: %clang -### -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fcuda-short-ptr %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-SHORT %s


// CHECK-SHORT: "-mllvm" "--nvptx-short-ptr"
// CHECK-SHORT: "-fcuda-short-ptr"

// CHECK-DEFAULT-NOT: "--nvptx-short-ptr"
// CHECK-DEFAULT-NOT: "-fcuda-short-ptr"
