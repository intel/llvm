/// Check that OpenMP and SYCL device binaries are wrapped and linked to the host
/// image when program uses both OpenMP and SYCL offloading models.

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target

// RUN: touch %t.o
// RUN: %clang --target=x86_64-host-linux-gnu -fsycl -fopenmp -fopenmp-targets=x86_64-device-linux-gnu -### %t.o 2>&1 \
// RUN:   | FileCheck %s
// CHECK: clang-offload-wrapper{{(.exe)?}}" {{.*}} "-target=spir64" "-kind=sycl"
// CHECK: clang-offload-wrapper{{(.exe)?}}" {{.*}} "-kind=openmp" "-target=x86_64-device-linux-gnu"
