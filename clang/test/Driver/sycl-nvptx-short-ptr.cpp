// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-DEFAULT %s

// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fcuda-short-ptr %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-SHORT %s


// CHECK-SHORT: "-target-abi" "shortptr"
// CHECK-SHORT-NOT: "-fcuda-short-ptr"

// CHECK-DEFAULT-NOT: "-target-abi" "shortptr"
// CHECK-DEFAULT-NOT: "-fcuda-short-ptr"
