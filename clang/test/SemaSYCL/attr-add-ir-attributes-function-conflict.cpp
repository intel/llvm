// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -verify %s

// Tests that add_ir_attributes_function causes a warning when appearing with
// potentially conflicting SYCL attributes.

#include "sycl.hpp"

struct NameValuePair {
  static constexpr const char *name = "Attr1";
  static constexpr const int value = 1;
};

template <typename... Pairs> struct Wrapper {
  template <typename KernelName, typename KernelType>
  [[__sycl_detail__::add_ir_attributes_function(Pairs::name..., Pairs::value...)]] __attribute__((sycl_kernel)) void kernel_single_task(const KernelType &kernelFunc) {
    kernelFunc();
  }
};

template <typename... Pairs> struct WrapperWithImplicit {
  template <typename KernelName, typename KernelType>
  [[__sycl_detail__::add_ir_attributes_function("sycl-single-task", Pairs::name..., 0, Pairs::value...)]] __attribute__((sycl_kernel)) void kernel_single_task(const KernelType &kernelFunc) {
    kernelFunc();
  }
};

template <typename... Pairs> struct WrapperWithFilter {
  template <typename KernelName, typename KernelType>
  [[__sycl_detail__::add_ir_attributes_function({"some-filter-string"}, Pairs::name..., Pairs::value...)]] __attribute__((sycl_kernel)) void kernel_single_task(const KernelType &kernelFunc) {
    kernelFunc();
  }
};

template <typename... Pairs> struct WrapperWithImplicitAndFilter {
  template <typename KernelName, typename KernelType>
  [[__sycl_detail__::add_ir_attributes_function({"some-filter-string"}, "sycl-single-task", Pairs::name..., 0, Pairs::value...)]] __attribute__((sycl_kernel)) void kernel_single_task(const KernelType &kernelFunc) {
    kernelFunc();
  }
};

int main() {
  Wrapper<> EmptyWrapper;
  Wrapper<NameValuePair> NonemptyWrapper;
  WrapperWithImplicit<> EmptyWrapperWithImplicit;
  WrapperWithImplicit<NameValuePair> NonemptyWrapperWithImplicit;
  WrapperWithFilter<> EmptyWrapperWithFilter;
  WrapperWithFilter<NameValuePair> NonemptyWrapperWithFilter;
  WrapperWithImplicitAndFilter<> EmptyWrapperWithImplicitAndFilter;
  WrapperWithImplicitAndFilter<NameValuePair> NonemptyWrapperWithImplicitAndFilter;

  EmptyWrapper.kernel_single_task<class EK1>([]() [[sycl::reqd_work_group_size(1)]] {});
  EmptyWrapper.kernel_single_task<class EK2>([]() [[sycl::reqd_work_group_size(1,2)]] {});
  EmptyWrapper.kernel_single_task<class EK3>([]() [[sycl::reqd_work_group_size(1,2,3)]] {});
  EmptyWrapper.kernel_single_task<class EK4>([]() [[sycl::work_group_size_hint(1)]] {});
  EmptyWrapper.kernel_single_task<class EK5>([]() [[sycl::work_group_size_hint(1,2)]] {});
  EmptyWrapper.kernel_single_task<class EK6>([]() [[sycl::work_group_size_hint(1,2,3)]] {});
  EmptyWrapper.kernel_single_task<class EK7>([]() [[sycl::reqd_sub_group_size(1)]] {});
  EmptyWrapper.kernel_single_task<class EK8>([]() [[sycl::device_has()]] {});
  EmptyWrapper.kernel_single_task<class EK9>([]() [[sycl::device_has(sycl::aspect::cpu)]] {});
  EmptyWrapper.kernel_single_task<class EK10>([]() [[sycl::device_has(sycl::aspect::cpu, sycl::aspect::gpu)]] {});

  EmptyWrapperWithImplicit.kernel_single_task<class EKWI1>([]() [[sycl::reqd_work_group_size(1)]] {});
  EmptyWrapperWithImplicit.kernel_single_task<class EKWI2>([]() [[sycl::reqd_work_group_size(1,2)]] {});
  EmptyWrapperWithImplicit.kernel_single_task<class EKWI3>([]() [[sycl::reqd_work_group_size(1,2,3)]] {});
  EmptyWrapperWithImplicit.kernel_single_task<class EKWI4>([]() [[sycl::work_group_size_hint(1)]] {});
  EmptyWrapperWithImplicit.kernel_single_task<class EKWI5>([]() [[sycl::work_group_size_hint(1,2)]] {});
  EmptyWrapperWithImplicit.kernel_single_task<class EKWI6>([]() [[sycl::work_group_size_hint(1,2,3)]] {});
  EmptyWrapperWithImplicit.kernel_single_task<class EKWI7>([]() [[sycl::reqd_sub_group_size(1)]] {});
  EmptyWrapperWithImplicit.kernel_single_task<class EKWI8>([]() [[sycl::device_has()]] {});
  EmptyWrapperWithImplicit.kernel_single_task<class EKWI9>([]() [[sycl::device_has(sycl::aspect::cpu)]] {});
  EmptyWrapperWithImplicit.kernel_single_task<class EKWI10>([]() [[sycl::device_has(sycl::aspect::cpu, sycl::aspect::gpu)]] {});

  EmptyWrapperWithFilter.kernel_single_task<class EKWF1>([]() [[sycl::reqd_work_group_size(1)]] {});
  EmptyWrapperWithFilter.kernel_single_task<class EKWF2>([]() [[sycl::reqd_work_group_size(1,2)]] {});
  EmptyWrapperWithFilter.kernel_single_task<class EKWF3>([]() [[sycl::reqd_work_group_size(1,2,3)]] {});
  EmptyWrapperWithFilter.kernel_single_task<class EKWF4>([]() [[sycl::work_group_size_hint(1)]] {});
  EmptyWrapperWithFilter.kernel_single_task<class EKWF5>([]() [[sycl::work_group_size_hint(1,2)]] {});
  EmptyWrapperWithFilter.kernel_single_task<class EKWF6>([]() [[sycl::work_group_size_hint(1,2,3)]] {});
  EmptyWrapperWithFilter.kernel_single_task<class EKWF7>([]() [[sycl::reqd_sub_group_size(1)]] {});
  EmptyWrapperWithFilter.kernel_single_task<class EKWF8>([]() [[sycl::device_has()]] {});
  EmptyWrapperWithFilter.kernel_single_task<class EKWF9>([]() [[sycl::device_has(sycl::aspect::cpu)]] {});
  EmptyWrapperWithFilter.kernel_single_task<class EKWF10>([]() [[sycl::device_has(sycl::aspect::cpu, sycl::aspect::gpu)]] {});

  EmptyWrapperWithImplicitAndFilter.kernel_single_task<class EKWIF1>([]() [[sycl::reqd_work_group_size(1)]] {});
  EmptyWrapperWithImplicitAndFilter.kernel_single_task<class EKWIF2>([]() [[sycl::reqd_work_group_size(1,2)]] {});
  EmptyWrapperWithImplicitAndFilter.kernel_single_task<class EKWIF3>([]() [[sycl::reqd_work_group_size(1,2,3)]] {});
  EmptyWrapperWithImplicitAndFilter.kernel_single_task<class EKWIF4>([]() [[sycl::work_group_size_hint(1)]] {});
  EmptyWrapperWithImplicitAndFilter.kernel_single_task<class EKWIF5>([]() [[sycl::work_group_size_hint(1,2)]] {});
  EmptyWrapperWithImplicitAndFilter.kernel_single_task<class EKWIF6>([]() [[sycl::work_group_size_hint(1,2,3)]] {});
  EmptyWrapperWithImplicitAndFilter.kernel_single_task<class EKWIF7>([]() [[sycl::reqd_sub_group_size(1)]] {});
  EmptyWrapperWithImplicitAndFilter.kernel_single_task<class EKWIF8>([]() [[sycl::device_has()]] {});
  EmptyWrapperWithImplicitAndFilter.kernel_single_task<class EKWIF9>([]() [[sycl::device_has(sycl::aspect::cpu)]] {});
  EmptyWrapperWithImplicitAndFilter.kernel_single_task<class EKWIF10>([]() [[sycl::device_has(sycl::aspect::cpu, sycl::aspect::gpu)]] {});

  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK1>([]() [[sycl::reqd_work_group_size(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK2>([]() [[sycl::reqd_work_group_size(1,2)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK3>([]() [[sycl::reqd_work_group_size(1,2,3)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK4>([]() [[sycl::work_group_size_hint(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK5>([]() [[sycl::work_group_size_hint(1,2)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK6>([]() [[sycl::work_group_size_hint(1,2,3)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_sub_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK7>([]() [[sycl::reqd_sub_group_size(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK8>([]() [[sycl::device_has()]] {});
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK9>([]() [[sycl::device_has(sycl::aspect::cpu)]] {});
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapper.kernel_single_task<class NEK10>([]() [[sycl::device_has(sycl::aspect::cpu, sycl::aspect::gpu)]] {});

  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicit.kernel_single_task<class NEKWI1>([]() [[sycl::reqd_work_group_size(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicit.kernel_single_task<class NEKWI2>([]() [[sycl::reqd_work_group_size(1,2)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicit.kernel_single_task<class NEKWI3>([]() [[sycl::reqd_work_group_size(1,2,3)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicit.kernel_single_task<class NEKWI4>([]() [[sycl::work_group_size_hint(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicit.kernel_single_task<class NEKWI5>([]() [[sycl::work_group_size_hint(1,2)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicit.kernel_single_task<class NEKWI6>([]() [[sycl::work_group_size_hint(1,2,3)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_sub_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicit.kernel_single_task<class NEKWI7>([]() [[sycl::reqd_sub_group_size(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicit.kernel_single_task<class NEKWI8>([]() [[sycl::device_has()]] {});
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicit.kernel_single_task<class NEKWI9>([]() [[sycl::device_has(sycl::aspect::cpu)]] {});
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicit.kernel_single_task<class NEKWI10>([]() [[sycl::device_has(sycl::aspect::cpu, sycl::aspect::gpu)]] {});

  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithFilter.kernel_single_task<class NEKWF1>([]() [[sycl::reqd_work_group_size(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithFilter.kernel_single_task<class NEKWF2>([]() [[sycl::reqd_work_group_size(1,2)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithFilter.kernel_single_task<class NEKWF3>([]() [[sycl::reqd_work_group_size(1,2,3)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithFilter.kernel_single_task<class NEKWF4>([]() [[sycl::work_group_size_hint(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithFilter.kernel_single_task<class NEKWF5>([]() [[sycl::work_group_size_hint(1,2)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithFilter.kernel_single_task<class NEKWF6>([]() [[sycl::work_group_size_hint(1,2,3)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_sub_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithFilter.kernel_single_task<class NEKWF7>([]() [[sycl::reqd_sub_group_size(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithFilter.kernel_single_task<class NEKWF8>([]() [[sycl::device_has()]] {});
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithFilter.kernel_single_task<class NEKWF9>([]() [[sycl::device_has(sycl::aspect::cpu)]] {});
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithFilter.kernel_single_task<class NEKWF10>([]() [[sycl::device_has(sycl::aspect::cpu, sycl::aspect::gpu)]] {});

  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicitAndFilter.kernel_single_task<class NEKWIF1>([]() [[sycl::reqd_work_group_size(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicitAndFilter.kernel_single_task<class NEKWIF2>([]() [[sycl::reqd_work_group_size(1,2)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicitAndFilter.kernel_single_task<class NEKWIF3>([]() [[sycl::reqd_work_group_size(1,2,3)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicitAndFilter.kernel_single_task<class NEKWIF4>([]() [[sycl::work_group_size_hint(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicitAndFilter.kernel_single_task<class NEKWIF5>([]() [[sycl::work_group_size_hint(1,2)]] {});
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicitAndFilter.kernel_single_task<class NEKWIF6>([]() [[sycl::work_group_size_hint(1,2,3)]] {});
  // expected-warning@+1 {{kernel has both attribute 'reqd_sub_group_size' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicitAndFilter.kernel_single_task<class NEKWIF7>([]() [[sycl::reqd_sub_group_size(1)]] {});
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicitAndFilter.kernel_single_task<class NEKWIF8>([]() [[sycl::device_has()]] {});
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicitAndFilter.kernel_single_task<class NEKWIF9>([]() [[sycl::device_has(sycl::aspect::cpu)]] {});
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  NonemptyWrapperWithImplicitAndFilter.kernel_single_task<class NEKWIF10>([]() [[sycl::device_has(sycl::aspect::cpu, sycl::aspect::gpu)]] {});
}
