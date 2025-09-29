// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <sycl/sycl.hpp>

int main() {
  int *r = nullptr;
  // Must not use `sycl_ext_oneapi_reduction_properties`'s overloads:
  std::ignore =
      sycl::reduction(r, sycl::plus<int>{},
                      sycl::property::reduction::initialize_to_identity{});

  namespace sycl_exp = sycl::ext::oneapi::experimental;
  std::ignore =
      sycl::reduction(r, sycl::plus<int>{},
                      sycl_exp::properties(sycl_exp::initialize_to_identity));

  // Not a property list:
  // expected-error@+2 {{no matching function for call to 'reduction'}}
  std::ignore =
      sycl::reduction(r, sycl::plus<int>{}, sycl_exp::initialize_to_identity);

  // Not a reduction property:
  // expected-error@+2 {{no matching function for call to 'reduction'}}
  std::ignore =
      sycl::reduction(r, sycl::plus<int>{},
                      sycl_exp::properties(sycl_exp::initialize_to_identity,
                                           sycl_exp::full_group));
}
