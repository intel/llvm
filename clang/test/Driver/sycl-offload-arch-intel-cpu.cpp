/// Tests the behaviors of using -fsycl --offload-new-driver 
//  --offload-arch=<intel-cpu-values>.

// SYCL AOT compilation to Intel CPUs using --offload-arch

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=broadwell %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=broadwell

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=coffeelake %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=coffeelake

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=icelake-client %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=icelake-client

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=skylake-avx512 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=skylake-avx512

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=core-avx2 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=core-avx2

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=corei7-avx %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=corei7-avx

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=corei7 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=corei7

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=westmere %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=westmere

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=sandybridge %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=sandybridge

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=ivybridge %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=ivybridge

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=alderlake %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=alderlake

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=skylake %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=skylake

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=skx %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=skx

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=cascadelake %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=cascadelake

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=icelake-server %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=icelake-server

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=sapphirerapids %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=sapphirerapids

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=graniterapids %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-CPU,CLANG-OFFLOAD-PACKAGER-CPU -DDEV_STR=graniterapids

// TARGET-TRIPLE-CPU: clang{{.*}} "-triple" "spir64_x86_64-unknown-unknown"
// TARGET-TRIPLE-CPU: "-D__SYCL_TARGET_INTEL_X86_64__"
// CLANG-OFFLOAD-PACKAGER-CPU: clang-offload-packager{{.*}} "--image={{.*}}triple=spir64_x86_64-unknown-unknown,arch=[[DEV_STR]],kind=sycl"

// Tests for handling a missing architecture.
//
// RUN: not %clangxx --offload-new-driver -fsycl --offload-arch= %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=MISSING %s
// RUN: not %clang_cl --offload-new-driver -fsycl --offload-arch= %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=MISSING %s

// MISSING: error: must pass in an explicit cpu or gpu architecture to '--offload-arch'

// Tests for handling a incorrect architecture.
//
// RUN: not %clangxx --offload-new-driver -fsycl --offload-arch=badArch %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=BAD-ARCH %s
// RUN: not %clang_cl --offload-new-driver -fsycl --offload-arch=badArch %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=BAD-ARCH %s

// BAD-ARCH: error: SYCL target is invalid: 'badArch'


