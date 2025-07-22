// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:     -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 --offload-host-only -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-HOST-ONLY
// CHECK-HOST-ONLY: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.*]]"], output: "[[OUTPUT:.*]]"
// CHECK-HOST-ONLY: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[OUTPUT]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:     -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 --offload-device-only -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEVICE-ONLY
// CHECK-DEVICE-ONLY: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.*]]"], output: "[[HOST_BC:.*]]"
// CHECK-DEVICE-ONLY: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_ASM:.*]]"
// CHECK-DEVICE-ONLY: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[DEVICE_ASM]]"], output: "{{.*}}-openmp-nvptx64-nvidia-cuda-sm_52.o"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:     -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_52 --offload-device-only -E -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEVICE-ONLY-PP
// CHECK-DEVICE-ONLY-PP: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT:.*]]"], output: "-"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=sm_52 --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:     -foffload-lto %s 2>&1 | FileCheck --check-prefix=CHECK-LTO-LIBRARY %s

// CHECK-LTO-LIBRARY: --device-linker={{.*}}=-lomp{{.*}}-lomptarget{{.*}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=sm_52 --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:     --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-nvptx-test.bc %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-LTO-LIBRARY %s

// CHECK-NO-LTO-LIBRARY: --device-linker={{.*}}=-lomp{{.*}}-lomptarget{{.*}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=sm_52 -nogpulib \
// RUN:     -foffload-lto %s 2>&1 | FileCheck --check-prefix=CHECK-NO-LIBRARY %s

// CHECK-NO-LIBRARY-NOT: --device-linker={{.*}}=-lomp{{.*}}-lomptarget{{.*}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=sm_52 -nogpulib \
// RUN:     -Xoffload-linker a -Xoffload-linker-nvptx64-nvidia-cuda b -Xoffload-linker-nvptx64 c \
// RUN:     %s 2>&1 | FileCheck --check-prefix=CHECK-XLINKER %s

// CHECK-XLINKER: -device-linker=a{{.*}}-device-linker=nvptx64-nvidia-cuda=b{{.*}}-device-linker=nvptx64-nvidia-cuda=c{{.*}}--

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=sm_52 -nogpulib \
// RUN:     -foffload-lto %s 2>&1 | FileCheck --check-prefix=CHECK-LTO-FEATURES %s

// CHECK-LTO-FEATURES: clang-offload-packager{{.*}}--image={{.*}}feature=+ptx{{[0-9]+}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp --offload-arch=sm_52 -nogpulib \
// RUN:     -Xopenmp-target=nvptx64-nvidia-cuda --cuda-feature=+ptx64 -foffload-lto %s 2>&1 \
// RUN:    | FileCheck --check-prefix=CHECK-SET-FEATURES %s

// CHECK-SET-FEATURES: clang-offload-packager{{.*}}--image={{.*}}feature=+ptx64

//
// Check that `-Xarch_host` works for OpenMP offloading.
//
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:     --offload-arch=sm_52 -nogpulib -nogpuinc -Xarch_host -O4 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=XARCH-HOST %s
// XARCH-HOST: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-O3"
// XARCH-HOST-NOT: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-O3"

//
// Check that `-Xarch_device` works for OpenMP offloading.
//
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:     --offload-arch=sm_52 -nogpulib -nogpuinc -Xarch_device -O4 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=XARCH-DEVICE %s
// XARCH-DEVICE: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-O3"
// XARCH-DEVICE-NOT: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-O3"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:      --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-nvptx-test.bc \
// RUN:      --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:      --offload-arch=sm_52 -nogpulibc -nogpuinc %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LIBC-GPU %s
// LIBC-GPU-NOT: clang-linker-wrapper{{.*}}"--device-linker"

//
// Check that ThinLTO works for OpenMP offloading.
//
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:     --offload-arch=gfx906 -foffload-lto=thin -nogpulib -nogpuinc %s 2>&1 \
// RUN:   | FileCheck --check-prefix=THINLTO-GFX906 %s
// THINLTO-GFX906: --device-compiler=amdgcn-amd-amdhsa=-flto=thin
// THINLTO-GFX906-SAME: --device-linker=amdgcn-amd-amdhsa=-plugin-opt=-force-import-all
// THINLTO-GFX906-SAME: --device-linker=amdgcn-amd-amdhsa=-plugin-opt=-avail-extern-to-local
// THINLTO-GFX906-SAME: --device-linker=amdgcn-amd-amdhsa=-plugin-opt=-avail-extern-gv-in-addrspace-to-local=3
// THINLTO-GFX906-SAME: --device-linker=amdgcn-amd-amdhsa=-plugin-opt=-amdgpu-internalize-symbols
//
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:     --offload-arch=sm_52 -foffload-lto=thin -nogpulib -nogpuinc %s 2>&1 \
// RUN:   | FileCheck --check-prefix=THINLTO-SM52 %s
// THINLTO-SM52: --device-compiler=nvptx64-nvidia-cuda=-flto=thin
