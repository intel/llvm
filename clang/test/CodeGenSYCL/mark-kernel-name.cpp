// RUN: %clang_cc1 -triple x86_64-linux-pc  -fsycl-is-host -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-pc  -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

template<typename KN, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &F) {
  F();
}

int main() {

  kernel<class K>([]() {
    auto lambda1 = []() {};
    auto lambda2 = []() {};

    (void)__builtin_sycl_unique_stable_name(decltype(lambda1));
    // CHECK: [35 x i8] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE_\00"

    // Should change the unique-stable-name of the lambda.
    (void)__builtin_sycl_mark_kernel_name(decltype(lambda2));
    (void)__builtin_sycl_unique_stable_name(decltype(lambda2));
    // CHECK: [40 x i8] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE10000_\00"
  });
}

