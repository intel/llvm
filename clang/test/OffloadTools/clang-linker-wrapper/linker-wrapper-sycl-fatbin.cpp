// REQUIRES: system-linux

// Fatbin-only output must not generate or compile a host wrapper.
// RUN: touch %t.o
// RUN: llvm-offload-binary -o %t.pack --image=file=%t.o,triple=spir64-unknown-unknown,arch=generic,kind=sycl
// RUN: clang-linker-wrapper --dry-run --emit-fatbin-only --linker-path=/usr/bin/ld %t.pack -o %t.fatbin 2>&1 | FileCheck %s
// CHECK: sycl-post-link
// CHECK: llvm-spirv
// CHECK-NOT: offload-wrapper
// CHECK-NOT: "{{.*}}clang"{{.*}} -c
// CHECK-NOT: "/usr/bin/ld"
// RUN: llvm-objdump --offloading %t.fatbin | FileCheck %s --check-prefix=CONTAINER
// CONTAINER: OFFLOADING IMAGE [0]:
// CONTAINER: kind elf
// CONTAINER: triple spir64-unknown-unknown
// CONTAINER: producer sycl

// Fatbin-only completion takes precedence over the device-link copy branch.
// RUN: clang-linker-wrapper --dry-run --emit-fatbin-only --sycl-device-link --linker-path=/usr/bin/ld %t.pack -o %t.devlink.fatbin 2>&1 | FileCheck %s
// RUN: llvm-objdump --offloading %t.devlink.fatbin | FileCheck %s --check-prefix=CONTAINER

// Keep all independent no-RDC modules in a single container.
// RUN: llvm-offload-binary -o %t.multi.pack --image=file=%t.o,triple=spir64-unknown-unknown,arch=generic,kind=sycl --image=file=%t.o,triple=spir64-unknown-unknown,arch=generic,kind=sycl
// RUN: clang-linker-wrapper --dry-run --emit-fatbin-only --no-sycl-rdc --linker-path=/usr/bin/ld %t.multi.pack -o %t.multi.fatbin 2>&1 | FileCheck %s
// RUN: llvm-objdump --offloading %t.multi.fatbin | FileCheck %s --check-prefix=MULTI
// MULTI: OFFLOADING IMAGE [0]:
// MULTI: producer sycl
// MULTI: OFFLOADING IMAGE [1]:
// MULTI: producer sycl
// MULTI-NOT: OFFLOADING IMAGE

// Reject modes that add host objects through the pipeline callback rather
// than allowing them to overwrite the device-image container.
// RUN: llvm-offload-binary -o %t.native.pack --image=file=%t.o,triple=native_cpu,arch=generic,kind=sycl
// RUN: not clang-linker-wrapper --dry-run --emit-fatbin-only --linker-path=/usr/bin/ld %t.native.pack -o %t.native.fatbin 2>&1 | FileCheck %s --check-prefix=NATIVE
// RUN: test ! -e %t.native.fatbin
// NATIVE-NOT: sycl-post-link
// NATIVE: error: --emit-fatbin-only is not supported for NativeCPU
// NATIVE-NOT: sycl-post-link

// RUN: llvm-offload-binary -o %t.nvptx.pack --image=file=%t.o,triple=nvptx64-nvidia-cuda,arch=sm_50,kind=sycl
// RUN: not clang-linker-wrapper --dry-run --emit-fatbin-only --sycl-embed-ir --linker-path=/usr/bin/ld %t.nvptx.pack -o %t.nvptx.fatbin 2>&1 | FileCheck %s --check-prefix=EMBED
// RUN: test ! -e %t.nvptx.fatbin
// RUN: llvm-offload-binary -o %t.amdgcn.pack --image=file=%t.o,triple=amdgcn-amd-amdhsa,arch=gfx908,kind=sycl
// RUN: not clang-linker-wrapper --dry-run --emit-fatbin-only --sycl-embed-ir --linker-path=/usr/bin/ld %t.amdgcn.pack -o %t.amdgcn.fatbin 2>&1 | FileCheck %s --check-prefix=EMBED
// RUN: test ! -e %t.amdgcn.fatbin
// EMBED-NOT: sycl-post-link
// EMBED: error: --emit-fatbin-only is not supported with --sycl-embed-ir for NVPTX or AMDGCN
// EMBED-NOT: sycl-post-link

// Embedding IR does not add host objects for SPIR targets.
// RUN: clang-linker-wrapper --dry-run --emit-fatbin-only --sycl-embed-ir --linker-path=/usr/bin/ld %t.pack -o %t.spir.fatbin 2>&1 | FileCheck %s
// RUN: llvm-objdump --offloading %t.spir.fatbin | FileCheck %s --check-prefix=CONTAINER
