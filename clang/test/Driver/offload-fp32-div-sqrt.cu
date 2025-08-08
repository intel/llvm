// RUN: %clang --cuda-device-only --cuda-gpu-arch=sm_20 \
// RUN: --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda -c \
// RUN: -foffload-fp32-prec-div -### %s 2>&1 | FileCheck %s

// RUN: %clang --cuda-device-only --cuda-gpu-arch=sm_20 \
// RUN: --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda -c \
// RUN: -foffload-fp32-prec-sqrt -### %s 2>&1 | FileCheck %s

// RUN: %clang --cuda-device-only --cuda-gpu-arch=sm_20 \
// RUN: --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda -c \
// RUN: -fno-offload-fp32-prec-div -### %s 2>&1 | FileCheck %s

// RUN: %clang --cuda-device-only --cuda-gpu-arch=sm_20 \
// RUN: --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda -c \
// RUN: -fno-offload-fp32-prec-sqrt -### %s 2>&1 | FileCheck %s

// RUN: %clang --cuda-device-only --cuda-gpu-arch=sm_20 \
// RUN: --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda -c \
// RUN: -ffp-accuracy=high -fno-offload-fp32-prec-div -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=FPACC %s

// RUN: %clang --cuda-device-only --cuda-gpu-arch=sm_20 \
// RUN: --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda -c \
// RUN: -ffp-accuracy=high -fno-offload-fp32-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=FPACC %s

// RUN: %clang --cuda-device-only --cuda-gpu-arch=sm_20 \
// RUN: --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda -c \
// RUN: -fno-offload-fp32-prec-div -ffp-accuracy=high -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=FPACC %s

// RUN: %clang --cuda-device-only --cuda-gpu-arch=sm_20 \
// RUN: --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda -c \
// RUN: -fno-offload-fp32-prec-sqrt -ffp-accuracy=high  -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=FPACC %s

// CHECK-NOT: "-foffload-fp32-prec-div"
// CHECK-NOT: "-foffload-fp32-prec-sqrt"
// FPACC: "-ffp-builtin-accuracy=high"
