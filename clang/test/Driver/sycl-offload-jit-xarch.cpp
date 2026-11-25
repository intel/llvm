// Test passing of -Xarch_<arch> <option> to SYCL offload compilations.

// Forwarding -Xarch_<arch> options to clang-linker-wrapper via
// --device-compiler=/--device-linker= was reverted in intel/llvm
// (9bec7a5f9549), so only the device compilation is checked.
// Verify that -Xarch_spirv64 forwards options to the SYCL device compilation.
// RUN: %clang -fsycl --offload-targets=spirv64-unknown-unknown -Xarch_spirv64 -O3 -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SYCL-DEVICE-O3 %s
// SYCL-DEVICE-O3: "-triple" "spirv64-unknown-unknown" "-aux-triple" "{{.*}}" "-O3"{{.*}} "-fsycl-is-device"
