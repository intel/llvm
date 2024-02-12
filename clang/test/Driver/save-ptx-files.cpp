// Save PTX files during PTX target processing using -fsycl-dump-device-code option.

// Verify that -fsycl-dump-device-code saves PTX files in the user provided directory
// while targeting CUDA enabled GPUs.

// Linux
// RUN: %clang -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64-unknown-unknown --cuda-path=%S/Inputs/CUDA/usr/local/cuda -fsycl-dump-device-code=/user/input/path %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-PTX-FILES,CHECK-SPIRV-FILES

// clang --driver-mode=g++
// RUN: %clangxx -### -fsycl  -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/CUDA/usr/local/cuda -fsycl-dump-device-code=/user/input/path %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-PTX-FILES

// RUN: %clang -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64-unknown-unknown --cuda-path=%S/Inputs/CUDA/usr/local/cuda -fsycl-dump-device-code= %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-PTX-FILES-CWD,CHECK-SPIRV-FILES-CWD

// CHECK-PTX-FILES: llvm-foreach{{.*}} "--out-ext=s"{{.*}} "--out-dir=/user/input/path{{(/|\\\\)}}" "--" "{{.*}}clang-18" {{.*}} "-fsycl-is-device" {{.*}}.s{{.*}}
// CHECK-SPIRV-FILES: llvm-foreach{{.*}} "--out-dir=/user/input/path{{(/|\\\\)}}" "--" "{{.*}}llvm-spirv"
// CHECK-PTX-FILES-CWD: llvm-foreach{{.*}} "--out-ext=s"{{.*}} "--out-dir=.{{(/|\\\\)}}" "--" "{{.*}}clang-18" {{.*}} "-fsycl-is-device"
// CHECK-SPIRV-FILES-CWD: llvm-foreach{{.*}} "--out-dir=.{{(/|\\\\)}}" "--" "{{.*}}llvm-spirv"

// Windows - Check if PTX files are saved in the user provided path.
// RUN: %clang_cl -### -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-dump-device-code=/user/input/path %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-PTX-WIN %s

// Windows - Check if PTX and SPV files are saved in user provided path.
// RUN: %clang_cl -### -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda,spir64-unknown-unknown --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-dump-device-code=/user/input/path %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-PTX-WIN,CHECK-SPV-WIN %s

// Windows - Check PTX files saved in current working directory when -fsycl-dump-device-code
// is empty. 
// RUN: %clang_cl -### -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-dump-device-code= %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-PTX-WIN-CWD %s

// CHECK-PTX-WIN: llvm-foreach{{.*}} "--out-ext=s"{{.*}} "--out-dir=/user/input/path{{(/|\\\\)}}" "--" "{{.*}}clang-18" {{.*}} "-fsycl-is-device" {{.*}}.asm{{.*}}
// CHECK-PTX-WIN-CWD: llvm-foreach{{.*}} "--out-ext=s"{{.*}} "--out-dir=.{{(/|\\\\)}}" "--" "{{.*}}clang-18" {{.*}} "-fsycl-is-device" {{.*}}.asm{{.*}}
// CHECK-SPV-WIN:  llvm-foreach{{.*}} "--out-dir=/user/input/path{{(/|\\\\)}}" "--" "{{.*}}llvm-spirv"
