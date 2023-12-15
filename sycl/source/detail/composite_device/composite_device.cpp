//==---------- composite_device.cpp - SYCL Composite Device ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/oneapi/experimental/composite_device.hpp>
#include <sycl/platform.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
std::vector<device> get_composite_devices() {
  std::vector<device> Composites;
  auto Devs = sycl::device::get_devices();
  for (const auto &D : Devs) {
    if (D.has(sycl::aspect::ext_oneapi_is_component)) {
      auto Composite = D.get_info<info::device::composite_device>();
      // Filter out duplicates.
      if (std::find(Composites.begin(), Composites.end(), Composite) ==
          Composites.end())
        Composites.push_back(Composite);
    }
  }
  std::vector<device> Result;
  for (const auto &Composite : Composites) {
    auto Components = Composite.get_info<info::device::component_devices>();
    // Only return composite devices if all of its component devices are
    // available.
    if (std::all_of(Components.begin(), Components.end(), [&](const device &d) {
          return std::find(Devs.begin(), Devs.end(), d) != Devs.end();
        })) {
      Result.push_back(Composite);
    }
  }
  return Result;
}
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
