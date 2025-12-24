//==----- range_rounding.cpp --- SYCL range rounding utils -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <sycl/detail/range_rounding.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

void GetRangeRoundingSettings(size_t &MinFactor, size_t &GoodFactor,
                              size_t &MinRange) {
  SYCLConfig<SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS>::GetSettings(
      MinFactor, GoodFactor, MinRange);
}

std::tuple<std::array<size_t, 3>, bool> getMaxWorkGroups(const device &Device) {
  std::array<size_t, 3> UrResult = {};
  auto &DeviceImpl = getSyclObjImpl(Device);

  auto Ret = DeviceImpl->getAdapter().call_nocheck<UrApiKind::urDeviceGetInfo>(
      DeviceImpl->getHandleRef(),
      UrInfoCode<
          ext::oneapi::experimental::info::device::max_work_groups<3>>::value,
      sizeof(UrResult), &UrResult, nullptr);
  if (Ret == UR_RESULT_SUCCESS) {
    return {UrResult, true};
  }
  return {std::array<size_t, 3>{0, 0, 0}, false};
}

bool DisableRangeRounding() {
  return SYCLConfig<SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING>::get();
}

bool RangeRoundingTrace() {
  return SYCLConfig<SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE>::get();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
