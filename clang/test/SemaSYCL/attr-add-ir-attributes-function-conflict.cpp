// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// Tests that add_ir_attributes_function causes a warning when appearing with
// potentially conflicting SYCL attributes.

constexpr const char AttrName1[] = "Attr1";
constexpr const char AttrVal1[] = "Val1";

template <const char *... Strs> struct Wrapper {
  template <typename KernelName, typename KernelType>
  [[__sycl_detail__::add_ir_attributes_function(Strs...)]] __attribute__((sycl_kernel)) void kernel_single_task(const KernelType &kernelFunc) {
    kernelFunc();
  }
};

int main() {
  Wrapper<> EmptyWrapper;
  Wrapper<AttrName1, AttrVal1> NonemptyWrapper;

  EmptyWrapper.kernel_single_task<class EK1>([]() [[sycl::reqd_work_group_size(1)]] {});
  EmptyWrapper.kernel_single_task<class EK2>([]() [[sycl::reqd_work_group_size(1,2)]] {});
  EmptyWrapper.kernel_single_task<class EK3>([]() [[sycl::reqd_work_group_size(1,2,3)]] {});
  EmptyWrapper.kernel_single_task<class EK1>([]() [[sycl::work_group_size_hint(1)]] {});
  EmptyWrapper.kernel_single_task<class EK2>([]() [[sycl::work_group_size_hint(1,2)]] {});
  EmptyWrapper.kernel_single_task<class EK3>([]() [[sycl::work_group_size_hint(1,2,3)]] {});
  EmptyWrapper.kernel_single_task<class EK7>([]() [[sycl::reqd_sub_group_size(1)]] {});

  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK1>([]() [[sycl::reqd_work_group_size(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK2>([]() [[sycl::reqd_work_group_size(1,2)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK3>([]() [[sycl::reqd_work_group_size(1,2,3)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK1>([]() [[sycl::work_group_size_hint(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK2>([]() [[sycl::work_group_size_hint(1,2)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK3>([]() [[sycl::work_group_size_hint(1,2,3)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_sub_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK7>([]() [[sycl::reqd_sub_group_size(1)]] {});
}
