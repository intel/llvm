// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Check -mamdgpu-cross-addr-space-atomic-memory-ordering is passed to -cc1 but
// (default) -mno-amdgpu-cross-addr-space-atomic-memory-ordering is not.

// RUN: %clang -### --target=amdgcn-amd-amdhsa -nogpuinc -nogpulib -mamdgpu-cross-addr-space-atomic-memory-ordering \
// RUN:   --offload-arch=gfx906  %s 2>&1 | FileCheck -check-prefixes=SYNCSCOPE %s
// SYNCSCOPE: "-cc1"{{.*}} "-triple" "amdgcn-amd-amdhsa" {{.*}} "-mamdgpu-cross-addr-space-atomic-memory-ordering"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx906  %s 2>&1 | FileCheck -check-prefixes=NOSYNCSCOPE %s
// NOSYNCSCOPE-NOT: "-m{{(no-)?}}amdgpu-cross-addr-space-atomic-memory-ordering"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -nogpuinc -nogpulib -mno-amdgpu-cross-addr-space-atomic-memory-ordering \
// RUN:   --offload-arch=gfx906  %s 2>&1 | FileCheck -check-prefixes=NOSYNCSCOPE %s
// NOSYNCSCOPE-NOT: "-m{{(no-)?}}amdgpu-cross-addr-space-atomic-memory-ordering"

