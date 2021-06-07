// RUN: %clang_cc1 -triple x86_64-linux-pc  -fsycl-is-host -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s


int main() {
  auto lambda1 = [](){};
  auto lambda2 = [](){};

  (void)__builtin_sycl_unique_stable_name(decltype(lambda1));
  // CHECK: [17 x i8] c"_ZTSZ4mainEUlvE_\00"

  // Should change the unique-stable-name of the lambda.
  (void)__builtin_sycl_mark_kernel_name(decltype(lambda2));
  (void)__builtin_sycl_unique_stable_name(decltype(lambda2));
  // CHECK: [22 x i8] c"_ZTSZ4mainEUlvE10000_\00"
}


