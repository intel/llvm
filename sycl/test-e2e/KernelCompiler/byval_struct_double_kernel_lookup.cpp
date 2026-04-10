//==--- byval_struct_double_kernel_lookup.cpp --- kernel_compiler tests ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include <cassert>
#include <string>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

static constexpr const char *SYCLSource = R"===(
#include <sycl/sycl.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclex = sycl::ext::oneapi::experimental;

struct MyCoolStruct {
  float a;
  double b;
  unsigned int c;
  size_t d;
};

extern "C" {

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<1>))
void check_device_repr_double(MyCoolStruct thing, unsigned long long *status) {
  const size_t i =
      syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  if (i != 0)
    return;
  (void)thing;
  status[0] = 1ull;
}

}
)===";

int main() {
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::queue q;

  const bool ok =
      q.get_device().ext_oneapi_can_build(syclex::source_language::sycl);
  assert(ok && "Device does not support kernel_bundle-from-source (sycl-jit)");

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      q.get_context(), syclex::source_language::sycl, SYCLSource);

  std::vector<sycl::device> devs{q.get_device()};
  exe_kb kbExe = syclex::build(kbSrc, devs);

  assert(kbExe.ext_oneapi_has_kernel("check_device_repr_double"));
  (void)kbExe.ext_oneapi_get_kernel("check_device_repr_double");

  return 0;
}
