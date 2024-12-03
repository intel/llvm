// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes \
// RUN:  -triple nvptx-nvidia-cuda -emit-llvm -fcuda-short-ptr -mllvm -nvptx-short-ptr %s -o - \
// RUN:    | FileCheck %s --check-prefix CHECK32

// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes \
// RUN:  -triple nvptx64-nvidia-cuda -emit-llvm -fcuda-short-ptr -mllvm -nvptx-short-ptr %s -o - \
// RUN:    | FileCheck %s --check-prefix CHECK64

// CHECK32: target datalayout = "e-p:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
// CHECK64: target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
