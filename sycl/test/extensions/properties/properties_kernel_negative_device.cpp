// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-is-device -ferror-limit=0 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <sycl/sycl.hpp>

template <size_t... Is> struct KernelFunctorWithWGSizeWithAttr {
  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  void operator() [[sycl::reqd_work_group_size(32)]] () const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::work_group_size<Is...>};
  }
};

template <size_t... Is> struct KernelFunctorWithWGSizeHintWithAttr {
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  void operator() [[sycl::work_group_size_hint(32)]] () const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::work_group_size_hint<Is...>};
  }
};

template <uint32_t I> struct KernelFunctorWithSGSizeWithAttr {
  // expected-warning@+1 {{kernel has both attribute 'reqd_sub_group_size' and kernel properties; conflicting properties are ignored}}
  void operator() [[sycl::reqd_sub_group_size(32)]] () const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<I>};
  }
};

template <sycl::aspect Aspect> struct KernelFunctorWithDeviceHasWithAttr {
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  void operator() [[sycl::device_has(sycl::aspect::cpu)]] () const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::device_has<Aspect>};
  }
};

void check_work_group_size() {
  sycl::queue Q;

  Q.single_task(KernelFunctorWithWGSizeWithAttr<1>{});
}

void check_work_group_size_hint() {
  sycl::queue Q;

  Q.single_task(KernelFunctorWithWGSizeHintWithAttr<1>{});
}

void check_sub_group_size() {
  sycl::queue Q;

  Q.single_task(KernelFunctorWithSGSizeWithAttr<1>{});
}

void check_device_has() {
  sycl::queue Q;

  Q.single_task(KernelFunctorWithDeviceHasWithAttr<sycl::aspect::cpu>{});
}

int main() {
  check_work_group_size();
  check_work_group_size_hint();
  check_sub_group_size();
  check_device_has();
  return 0;
}
