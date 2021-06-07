// RUN: %clang_cc1 %s -std=c++17 -triple x86_64-linux-gnu -Wno-sycl-2020-compat -fsycl-is-device -verify -fsyntax-only -Wno-unused

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel]] void kernel_single_task(KernelType kernelFunc) { // #kernelSingleTask
  kernelFunc();
}

template <typename KN>
struct KernelInfo {
  static constexpr const char *c = __builtin_sycl_unique_stable_name(KN); // #KI_USN
};

template <typename KN>
struct FixedKernelInfo {
  static constexpr bool b = __builtin_sycl_mark_kernel_name(KN);
  // making 'c' dependent on 'b' is necessary to ensure 'b' gets called first.
  static constexpr const char *c = b
                                       ? __builtin_sycl_unique_stable_name(KN)
                                       : nullptr;
};

template <template <typename> class KI,
          typename KernelName,
          typename KernelType>
void wrapper(KernelType KernelFunc) {
  (void)KI<KernelName>::c;
  kernel_single_task<KernelName>(KernelFunc); // #SingleTaskInst
}

int main() {
  []() {
    class KernelName1;
    constexpr const char *C = __builtin_sycl_unique_stable_name(KernelName1);
    // expected-error@+2 {{kernel naming changes the result of an evaluated '__builtin_sycl_unique_stable_name'}}
    // expected-note@-2 {{'__builtin_sycl_unique_stable_name' evaluated here}}
    __builtin_sycl_mark_kernel_name(KernelName1);
  }();

  []() {
    // expected-error@#kernelSingleTask {{kernel instantiation changes the result of an evaluated '__builtin_sycl_unique_stable_name'}}
    // expected-note@#SingleTaskInst {{in instantiation of function template}}
    // expected-note@+2 {{in instantiation of function template}}
    // expected-note@#KI_USN {{'__builtin_sycl_unique_stable_name' evaluated here}}
    wrapper<KernelInfo, class KernelName2>([]() {});
  }();

  []() {
    wrapper<FixedKernelInfo, class KernelName3>([]() {});
  }();
}
