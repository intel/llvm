// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

SYCL_EXTERNAL void cast_group_ballots(sycl::nd_item<1> item) {
  auto Mask = sycl::ext::oneapi::group_ballot(item.get_sub_group());
}
