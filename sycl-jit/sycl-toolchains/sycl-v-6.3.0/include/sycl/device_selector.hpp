//==------ device_selector.hpp - SYCL device selector ---------*- C++ --*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT

#include <functional>  // for function
#include <type_traits> // for enable_if_t
#include <vector>      // for vector

// 4.6.1 Device selection class

namespace sycl {
inline namespace _V1 {

// Forward declarations
class device;
class context;
enum class aspect;

namespace ext::oneapi {
class filter_selector;
} // namespace ext::oneapi

/// The SYCL 1.2.1 device_selector class provides ability to choose the
/// best SYCL device based on heuristics specified by the user.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT __SYCL2020_DEPRECATED(
    "Use SYCL 2020 callable device selectors instead.") device_selector {

public:
  virtual ~device_selector() = default;

  virtual device select_device() const;

  virtual int operator()(const device &device) const = 0;
};

/// The default selector chooses the first available SYCL device.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT __SYCL2020_DEPRECATED(
    "Use the callable sycl::default_selector_v instead.") default_selector
    : public device_selector {
public:
  int operator()(const device &dev) const override;
};

/// Selects any SYCL GPU device.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT __SYCL2020_DEPRECATED(
    "Use the callable sycl::gpu_selector_v instead.") gpu_selector
    : public device_selector {
public:
  int operator()(const device &dev) const override;
};

/// Selects any SYCL CPU device.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT __SYCL2020_DEPRECATED(
    "Use the callable sycl::cpu_selector_v instead.") cpu_selector
    : public device_selector {
public:
  int operator()(const device &dev) const override;
};

/// Selects any SYCL accelerator device.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT
__SYCL2020_DEPRECATED("Use the callable sycl::accelerator_selector_v instead.")
    accelerator_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

// -------------- SYCL 2020

// SYCL 2020 standalone selectors
__SYCL_EXPORT int default_selector_v(const device &dev);
__SYCL_EXPORT int gpu_selector_v(const device &dev);
__SYCL_EXPORT int cpu_selector_v(const device &dev);
__SYCL_EXPORT int accelerator_selector_v(const device &dev);

namespace detail {
// SYCL 2020 section 4.6.1.1 defines a negative score to reject a device from
// selection
static constexpr int REJECT_DEVICE_SCORE = -1;

using DSelectorInvocableType = std::function<int(const sycl::device &)>;

template <typename LastT>
void fill_aspect_vector(std::vector<aspect> &V, LastT L) {
  V.emplace_back(L);
}

template <typename FirstT, typename... OtherTs>
void fill_aspect_vector(std::vector<aspect> &V, FirstT F, OtherTs... O) {
  V.emplace_back(F);
  fill_aspect_vector(V, O...);
}

// Enable if DeviceSelector callable has matching signature, but
// exclude if descended from filter_selector which is not purely callable or
// if descended from it is descended from SYCL 1.2.1 device_selector.
// See [FilterSelector not Callable] in device_selector.cpp
template <typename DeviceSelector>
using EnableIfSYCL2020DeviceSelectorInvocable = std::enable_if_t<
    std::is_invocable_r_v<int, DeviceSelector &, const device &> &&
    !std::is_base_of_v<ext::oneapi::filter_selector, DeviceSelector> &&
    !std::is_base_of_v<device_selector, DeviceSelector>>;

__SYCL_EXPORT device
select_device(const DSelectorInvocableType &DeviceSelectorInvocable);

__SYCL_EXPORT device
select_device(const DSelectorInvocableType &DeviceSelectorInvocable,
              const context &SyclContext);

} // namespace detail

__SYCL_EXPORT detail::DSelectorInvocableType
aspect_selector(const std::vector<aspect> &RequireList,
                const std::vector<aspect> &DenyList = {});

template <typename... AspectListT>
detail::DSelectorInvocableType aspect_selector(AspectListT... AspectList) {
  std::vector<aspect> RequireList;
  RequireList.reserve(sizeof...(AspectList));

  detail::fill_aspect_vector(RequireList, AspectList...);

  return aspect_selector(RequireList, {});
}

template <aspect... AspectList>
detail::DSelectorInvocableType aspect_selector() {
  return aspect_selector({AspectList...}, {});
}

} // namespace _V1
} // namespace sycl
