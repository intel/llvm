/// Check that the driver passes the user's host ABI options to
/// clang-linker-wrapper, so the wrapper object matches the rest of the program.
// RUN: %clangxx -### -fsycl --offload-new-driver --no-offloadlib \
// RUN:   -fno-sycl-instrument-device-code \
// RUN:   --target=riscv64-unknown-linux-gnu -march=rv64id -mcpu=sifive-u74 \
// RUN:   -mabi=lp64 -mcmodel=medany %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ABI %s
// ABI: clang-linker-wrapper{{.*}} "--sycl-host-compiler-arg=-march=rv64id"
// ABI-SAME: "--sycl-host-compiler-arg=-mcpu=sifive-u74"
// ABI-SAME: "--sycl-host-compiler-arg=-mabi=lp64"
// ABI-SAME: "--sycl-host-compiler-arg=-mcmodel=medany"

/// No host ABI options given: nothing extra is passed.
// RUN: %clangxx -### -fsycl --offload-new-driver --no-offloadlib \
// RUN:   -fno-sycl-instrument-device-code \
// RUN:   --target=x86_64-unknown-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NONE %s
// NONE: clang-linker-wrapper
// NONE-NOT: --sycl-host-compiler-arg

/// The old offloading model passes the same options to the clang command that
/// compiles the wrapper.
// RUN: %clangxx -### -fsycl --no-offload-new-driver --no-offloadlib \
// RUN:   -fno-sycl-instrument-device-code \
// RUN:   --target=riscv64-unknown-linux-gnu -march=rv64id -mcpu=sifive-u74 \
// RUN:   -mabi=lp64 -mcmodel=medany %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OLD %s
// OLD: clang-offload-wrapper
// OLD: clang{{.*}} "--target=riscv64-unknown-linux-gnu" "-c"
// OLD-SAME: "-march=rv64id" "-mcpu=sifive-u74" "-mabi=lp64" "-mcmodel=medany"
