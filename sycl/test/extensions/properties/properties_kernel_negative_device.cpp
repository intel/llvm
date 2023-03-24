// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-is-device -ferror-limit=0 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <sycl/sycl.hpp>

struct KernelFunctorWithOnlyWGSizeAttr {
  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  void operator() [[sycl::reqd_work_group_size(32)]] () const {}
};

template <size_t... Is> struct KernelFunctorWithWGSizeWithAttr {
  // expected-warning@+1 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  void operator() [[sycl::reqd_work_group_size(32)]] () const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::work_group_size<Is...>};
  }
};

struct KernelFunctorWithOnlyWGSizeHintAttr {
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  void operator() [[sycl::work_group_size_hint(32)]] () const {}
};

template <size_t... Is> struct KernelFunctorWithWGSizeHintWithAttr {
  // expected-warning@+1 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  void operator() [[sycl::work_group_size_hint(32)]] () const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::work_group_size_hint<Is...>};
  }
};

struct KernelFunctorWithOnlySGSizeAttr {
  // expected-warning@+1 {{kernel has both attribute 'reqd_sub_group_size' and kernel properties; conflicting properties are ignored}}
  void operator() [[sycl::reqd_sub_group_size(32)]] () const {}
};

template <uint32_t I> struct KernelFunctorWithSGSizeWithAttr {
  // expected-warning@+1 {{kernel has both attribute 'reqd_sub_group_size' and kernel properties; conflicting properties are ignored}}
  void operator() [[sycl::reqd_sub_group_size(32)]] () const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<I>};
  }
};

struct KernelFunctorWithOnlyDeviceHasAttr {
  // expected-warning@+1 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  void operator() [[sycl::device_has(sycl::aspect::cpu)]] () const {}
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

  // expected-warning@+4 {{kernel has both attribute 'reqd_work_group_size' and kernel properties; conflicting properties are ignored}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size<1>},
      []() [[sycl::reqd_work_group_size(32)]] {});

  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size<1>},
      KernelFunctorWithOnlyWGSizeAttr{});

  Q.single_task(KernelFunctorWithWGSizeWithAttr<1>{});
}

void check_work_group_size_hint() {
  sycl::queue Q;

  // expected-warning@+4 {{kernel has both attribute 'work_group_size_hint' and kernel properties; conflicting properties are ignored}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size_hint<1>},
      []() [[sycl::work_group_size_hint(32)]] {});

  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size_hint<1>},
      KernelFunctorWithOnlyWGSizeHintAttr{});

  Q.single_task(KernelFunctorWithWGSizeHintWithAttr<1>{});
}

void check_sub_group_size() {
  sycl::queue Q;

  // expected-warning@+4 {{kernel has both attribute 'reqd_sub_group_size' and kernel properties; conflicting properties are ignored}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::sub_group_size<1>},
      []() [[sycl::reqd_sub_group_size(32)]] {});

  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::sub_group_size<1>},
      KernelFunctorWithOnlySGSizeAttr{});

  Q.single_task(KernelFunctorWithSGSizeWithAttr<1>{});
}

void check_device_has() {
  sycl::queue Q;

  // expected-warning@+4 {{kernel has both attribute 'device_has' and kernel properties; conflicting properties are ignored}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::device_has<sycl::aspect::cpu>},
      []() [[sycl::device_has(sycl::aspect::cpu)]] {});

  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::device_has<sycl::aspect::cpu>},
      KernelFunctorWithOnlyDeviceHasAttr{});

  Q.single_task(KernelFunctorWithDeviceHasWithAttr<sycl::aspect::cpu>{});
}

int main() {
  check_work_group_size();
  check_work_group_size_hint();
  check_sub_group_size();
  check_device_has();
  return 0;
}
