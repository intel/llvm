// Checking to see that __nvvm_reflect resolves to the correct llvm intrinsic
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -disable-llvm-passes -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NO_NVVM_REFLECT_PASS

// Checking to see if the correct values are substituted for the nvvm_reflect
// call when llvm passes are enabled.
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_50 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=ARCH_REFLECT_2
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_52 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=ARCH_REFLECT_3 
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_53 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=ARCH_REFLECT_4
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_60 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=ARCH_REFLECT_5
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_61 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=ARCH_REFLECT_6
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_62 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=ARCH_REFLECT_7
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_70 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=ARCH_REFLECT_8
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_72 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=ARCH_REFLECT_9
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_75 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=ARCH_REFLECT_10
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_80 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=ARCH_REFLECT_11
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_86 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=ARCH_REFLECT_12

// Check that -fcuda-prec-sqrt flag makes nvvm_reflect("__CUDA_PREC_SQRT") return 1
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -fcuda-prec-sqrt -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=PREC_SQRT_REFLECT
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NO_PREC_SQRT_REFLECT

#include "sycl.hpp"

using namespace sycl;

queue q{};

int main() {
  // NO_NVVM_REFLECT_PASS: call i32 @llvm.nvvm.reflect
  // ARCH_REFLECT_2: store i32 500
  // ARCH_REFLECT_3: store i32 520
  // ARCH_REFLECT_4: store i32 530
  // ARCH_REFLECT_5: store i32 600
  // ARCH_REFLECT_6: store i32 610
  // ARCH_REFLECT_7: store i32 620
  // ARCH_REFLECT_8: store i32 700
  // ARCH_REFLECT_9: store i32 720
  // ARCH_REFLECT_10: store i32 750
  // ARCH_REFLECT_11: store i32 800
  // ARCH_REFLECT_12: store i32 860
  q.submit([&](handler &cgh) {
    cgh.single_task(
        [=]() { printf("%d", __nvvm_reflect((char *)"__CUDA_ARCH")); });
  });

  // PREC_SQRT_REFLECT: store i32 1
  // NO_PREC_SQRT_REFLECT: store i32 0
  q.submit([&](handler &cgh) {
    cgh.single_task(
        [=]() { printf("%d", __nvvm_reflect((char *)"__CUDA_PREC_SQRT")); });
  });
}
