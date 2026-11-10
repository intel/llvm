/// Check that the driver passes the user's host ABI options to
/// clang-linker-wrapper, so the wrapper object matches the rest of the program.
// RUN: %clangxx -### -fsycl --offload-new-driver --no-offloadlib \
// RUN:   -fno-sycl-instrument-device-code \
// RUN:   --target=riscv64-unknown-linux-gnu -march=rv64id -mabi=lp64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ABI %s
// ABI: clang-linker-wrapper{{.*}} "--sycl-host-compiler-arg=-march=rv64id"
// ABI-SAME: "--sycl-host-compiler-arg=-mabi=lp64"

/// No host ABI options given: nothing extra is passed.
// RUN: %clangxx -### -fsycl --offload-new-driver --no-offloadlib \
// RUN:   -fno-sycl-instrument-device-code \
// RUN:   --target=x86_64-unknown-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NONE %s
// NONE: clang-linker-wrapper
// NONE-NOT: --sycl-host-compiler-arg
