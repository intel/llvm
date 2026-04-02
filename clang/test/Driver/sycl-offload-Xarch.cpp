// Test passing of -Xarch_<arch> <option> to SYCL offload compilations.

// RUN: %clang --offload-new-driver -fsycl -fsycl-targets=amdgcn-amd-amdhsa -nogpulib -fno-sycl-libspirv -nogpuinc \
// RUN:   -Xarch_amdgcn -march=gfx90a -Xarch_amdgcn -O3 -S -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=O3ONCE-AMD,AMD-ARCH %s
// RUN: %clang --offload-new-driver -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nogpulib -fno-sycl-libspirv -nogpuinc \
// RUN:   -Xarch_nvptx64 -march=sm_52 -Xarch_nvptx64 -O3 -S -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=O3ONCE-NVPTX,NVPTX-ARCH %s

// RUN: not %clang --offload-new-driver -fsycl -fsycl-targets=amdgcn-amd-amdhsa -nogpulib -fno-sycl-libspirv -nogpuinc \
// RUN:   -Xarch_amdgcn -march=sm_75 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=INVALID-ARCH-FOR-TARGET %s


// RUN: %clang --offload-new-driver -fsycl -fsycl-targets=amdgcn-amd-amdhsa -nogpulib -fno-sycl-libspirv -nogpuinc \
// RUN:   -Xarch_amdgcn --offload-arch=gfx90a -Xarch_amdgcn -O3 -S -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=O3ONCE-AMD,AMD-ARCH %s
// RUN: %clang --offload-new-driver -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nogpulib -fno-sycl-libspirv -nogpuinc \
// RUN:   -Xarch_nvptx64 --offload-arch=sm_52 -Xarch_nvptx64 -O3 -S -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=O3ONCE-NVPTX,NVPTX-ARCH %s


// RUN: %clang --offload-new-driver -fsycl -fsycl-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda -nogpulib -fno-sycl-libspirv -nogpuinc \
// RUN:   -Xarch_amdgcn --offload-arch=gfx90a,gfx906 -Xarch_nvptx64 --offload-arch=sm_52,sm_89 -S -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=MULTI-ARCH %s

// O3ONCE-AMD: "-triple" "amdgcn-amd-amdhsa" "-O3"
// O3ONCE-NVPTX: "-triple" "nvptx64-nvidia-cuda" "-O3"
// INVALID-ARCH-FOR-TARGET: clang: error: invalid target ID 'sm_75'; format is a processor name followed by an optional colon-delimited list of features followed by an enable/disable sign (e.g., 'gfx908:sramecc+:xnack-')
// AMD-ARCH: {{"[^"]*llvm-offload-binary[^"]*" "-o".* "--image=file=.*.bc,triple=amdgcn-amd-amdhsa,arch=gfx90a,kind=sycl"}}
// NVPTX-ARCH: {{"[^"]*llvm-offload-binary[^"]*" "-o".* "--image=file=.*.bc,triple=nvptx64-nvidia-cuda,arch=sm_52,kind=sycl"}}
// MULTI-ARCH: {{"[^"]*llvm-offload-binary[^"]*" "-o".* "--image=file=.*.bc,triple=amdgcn-amd-amdhsa,arch=gfx906,kind=sycl"}}
// MULTI-ARCH-SAME: {{"--image=file=.*.bc,triple=amdgcn-amd-amdhsa,arch=gfx90a,kind=sycl"}}
// MULTI-ARCH-SAME: {{"--image=file=.*.bc,triple=nvptx64-nvidia-cuda,arch=sm_52,kind=sycl"}}
// MULTI-ARCH-SAME: {{"--image=file=.*.bc,triple=nvptx64-nvidia-cuda,arch=sm_89,kind=sycl"}}

// Make sure that `-Xarch_amdgcn` forwards libraries to the device linker.
// RUN: %clang --offload-new-driver -fsycl --offload-arch=gfx90a -nogpulib -fno-sycl-libspirv -nogpuinc \
// RUN:   --target=x86_64-unknown-linux-gnu -Xarch_amdgcn -Wl,-lfoo -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=LIBS %s
// RUN: %clang --offload-new-driver -fsycl --offload-arch=gfx90a -nogpulib -fno-sycl-libspirv -nogpuinc \
// RUN:   -Xoffload-linker-amdgcn-amd-amdhsa -lfoo -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=LIBS %s
// LIBS: "--device-linker=amdgcn-amd-amdhsa=-lfoo"

