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

int main() {
  check_work_group_size();
  check_sub_group_size();
  return 0;
}
