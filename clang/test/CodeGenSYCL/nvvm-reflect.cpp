// Checking to see that __nvvm_reflect resolves to the correct llvm intrinsic
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -disable-llvm-passes -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NO_NVVM_REFLECT_PASS

// Checking to see if the correct values are substituted for the nvvm_reflect
// call when llvm passes are enabled.
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_50 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NVVM_REFLECT_PASS_2
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_52 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NVVM_REFLECT_PASS_3 
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_53 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NVVM_REFLECT_PASS_4
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_60 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NVVM_REFLECT_PASS_5
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_61 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NVVM_REFLECT_PASS_6
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_62 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NVVM_REFLECT_PASS_7
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_70 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NVVM_REFLECT_PASS_8
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_72 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NVVM_REFLECT_PASS_9
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_75 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NVVM_REFLECT_PASS_10
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_80 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NVVM_REFLECT_PASS_11
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -target-cpu sm_86 -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefix=NVVM_REFLECT_PASS_12

#include "sycl.hpp"

using namespace sycl;

queue q{};

int main() {
  // NO_NVVM_REFLECT_PASS: call i32 @llvm.nvvm.reflect
  // NVVM_REFLECT_PASS_2: store i32 500
  // NVVM_REFLECT_PASS_3: store i32 520
  // NVVM_REFLECT_PASS_4: store i32 530
  // NVVM_REFLECT_PASS_5: store i32 600
  // NVVM_REFLECT_PASS_6: store i32 610
  // NVVM_REFLECT_PASS_7: store i32 620
  // NVVM_REFLECT_PASS_8: store i32 700
  // NVVM_REFLECT_PASS_9: store i32 720
  // NVVM_REFLECT_PASS_10: store i32 750
  // NVVM_REFLECT_PASS_11: store i32 800
  // NVVM_REFLECT_PASS_12: store i32 860
  q.submit([&](handler &cgh) {
    cgh.single_task(
        [=]() { printf("%d", __nvvm_reflect((char *)"__CUDA_ARCH")); });
  });
}
