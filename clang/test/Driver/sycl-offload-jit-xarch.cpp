// Test passing of -Xarch_<arch> <option> to SYCL offload compilations.

// Verify that -Xarch_spir64 forwards options to the SYCL device compilation
// and clang-linker-wrapper call.
// RUN: %clang -fsycl --offload-new-driver -Xarch_spir64 -O3 -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=SYCL-DEVICE-O3 %s
// SYCL-DEVICE-O3: "-triple" "spir64-unknown-unknown" "-O3"{{.*}} "-fsycl-is-device"
// CLW-O3: {{"[^"]*clang-linker-wrapper[^"]*".* "--device-compiler=spirv64-unknown-unknown=-O3"}}

// Verify that `-Xarch_spir64` forwards libraries to the device linker.
// RUN: %clang -fsycl --offload-new-driver -Xarch_spir64 -Wl,-lfoo -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=DEVICE-LINKER %s
// DEVICE-LINKER: {{"[^"]*clang-linker-wrapper[^"]*".* "--device-linker=spir64-unknown-unknown=-lfoo"}}
