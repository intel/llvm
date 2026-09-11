// Memcpy optimizations are enabled unconditionally for every target since the
// libcall availability checks, and with them the
// -enable-memcpyopt-without-libcalls flag, were removed from MemCpyOpt. The
// SYCL NVPTX toolchain therefore must not try to opt into them explicitly
// anymore: the flag no longer exists, so passing it would make every device
// compilation fail with an unknown -mllvm argument.

// RUN: %clang -### -fno-sycl-libspirv -nocudalib \
// RUN:   -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-DEFAULT %s

// CHECK-DEFAULT: "-fsycl-is-device"
// CHECK-DEFAULT-NOT: "-enable-memcpyopt-without-libcalls"
