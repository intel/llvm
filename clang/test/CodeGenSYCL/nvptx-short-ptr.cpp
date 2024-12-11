// Check that when we see the expected data layouts for NVPTX when we pass the
// -nvptx-short-ptr option.

// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes \
// RUN:  -triple nvptx-nvidia-cuda -emit-llvm %s -o - \
// RUN:    | FileCheck %s --check-prefix CHECK32

// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes \
// RUN:  -triple nvptx-nvidia-cuda -emit-llvm -fcuda-short-ptr -mllvm -nvptx-short-ptr %s -o - \
// RUN:    | FileCheck %s --check-prefix CHECK32

// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes \
// RUN:  -triple nvptx64-nvidia-cuda -emit-llvm %s -o - \
// RUN:    | FileCheck %s --check-prefix CHECK64-DEFAULT

// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes \
// RUN:  -triple nvptx64-nvidia-cuda -emit-llvm -fcuda-short-ptr -mllvm -nvptx-short-ptr %s -o - \
// RUN:    | FileCheck %s --check-prefix CHECK64-SHORT

// Targeting a 32-bit NVPTX, check that we see universal 32-bit pointers (the
// option changes nothing)
// CHECK32: target datalayout = "e-p:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"

// Targeting a 64-bit NVPTX target, check that we see 32-bit pointers for
// shared (3), const (4), and local (5) address spaces only.
// CHECK64-DEFAULT: target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
// CHECK64-SHORT: target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
