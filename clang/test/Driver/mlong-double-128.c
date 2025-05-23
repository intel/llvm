// RUN: %clang --target=powerpc-linux-musl -c -### %s -mlong-double-128 2>&1 | FileCheck %s
// RUN: %clang --target=powerpc64-pc-freebsd12 -c -### %s -mlong-double-128 2>&1 | FileCheck %s
// RUN: %clang --target=powerpc64le-linux-musl -c -### %s -mlong-double-128 2>&1 | FileCheck %s
// RUN: %clang --target=i686-linux-gnu -c -### %s -mlong-double-128 2>&1 | FileCheck %s

// RUN: %clang --target=x86_64-linux-musl -c -### %s -mlong-double-128 -mlong-double-80 2>&1 | FileCheck --implicit-check-not=-mlong-double-128 /dev/null
// RUN: %clang --target=x86_64-linux-musl -c -### %s -mlong-double-80 -mlong-double-128 2>&1 | FileCheck %s

// CHECK: "-mlong-double-128"

// RUN: not %clang --target=aarch64 -c -### %s -mlong-double-128 2>&1 | FileCheck --check-prefix=ERR %s
// RUN: not %clang --target=powerpc -c -### %s -mlong-double-80 2>&1 | FileCheck --check-prefix=ERR2 %s
// RUN: not %clang --target=spir64-unknown-unknown -c -### %s -mlong-double-128 2>&1 | FileCheck --check-prefix=ERR3 %s
// RUN: not %clang --target=spir64-unknown-unknown -c -### %s -mlong-double-80 2>&1 | FileCheck --check-prefix=ERR4 %s
// RUN: not %clang --target=nvptx64-nvidia-cuda -c -### %s -mlong-double-128 2>&1 | FileCheck --check-prefix=ERR5 %s
// RUN: not %clang --target=nvptx64-nvidia-cuda -c -### %s -mlong-double-80 2>&1 | FileCheck --check-prefix=ERR6 %s
// RUN: not %clang --target=amd_gpu_gfx1031 -c -### %s -mlong-double-128 2>&1 | FileCheck --check-prefix=ERR7 %s
// RUN: not %clang --target=amd_gpu_gfx1031 -c -### %s -mlong-double-80 2>&1 | FileCheck --check-prefix=ERR8 %s

// ERR: error: unsupported option '-mlong-double-128' for target 'aarch64'
// ERR2: error: unsupported option '-mlong-double-80' for target 'powerpc'
// ERR3: error: unsupported option '-mlong-double-128' for target 'spir64-unknown-unknown'
// ERR4: error: unsupported option '-mlong-double-80' for target 'spir64-unknown-unknown'
// ERR5: error: unsupported option '-mlong-double-128' for target 'nvptx64-nvidia-cuda'
// ERR6: error: unsupported option '-mlong-double-80' for target 'nvptx64-nvidia-cuda'
// ERR7: error: unsupported option '-mlong-double-128' for target 'amd_gpu_gfx1031'
// ERR8: error: unsupported option '-mlong-double-80' for target 'amd_gpu_gfx1031'
