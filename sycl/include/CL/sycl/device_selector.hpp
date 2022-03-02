//==------ device_selector.hpp - SYCL device selector ---------*- C++ --*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/aspects.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/export.hpp>

// 4.6.1 Device selection class

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declarations
class device;

/// The device_selector class provides ability to choose the best SYCL device
/// based on heuristics specified by the user.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT device_selector {
protected:
  // SYCL 1.2.1 defines a negative score to reject a device from selection
  static constexpr int REJECT_DEVICE_SCORE = -1;

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
class __SYCL_EXPORT default_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

/// Selects any SYCL GPU device.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT gpu_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

/// Selects any SYCL CPU device.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT cpu_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

/// Selects any SYCL accelerator device.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT accelerator_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

/// Selects SYCL host device.
///
/// \sa device
///
/// \ingroup sycl_api_dev_sel
class __SYCL_EXPORT host_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

class __SYCL_EXPORT aspect_selector_t : public default_selector {
  const std::vector<aspect> MRequireList;
  const std::vector<aspect> MDenyList;

public:
  aspect_selector_t(const std::vector<aspect> &AspectList,
                    const std::vector<aspect> &DenyList = {})
      : MRequireList{AspectList}, MDenyList{DenyList} {}

  int operator()(const device &dev) const override;
};

__SYCL_EXPORT aspect_selector_t
aspect_selector(const std::vector<aspect> &AspectList,
                const std::vector<aspect> &DenyList = {});

namespace detail {
template <typename LastT>
void fill_aspect_vector(std::vector<aspect> &V, LastT L) {
  V.emplace_back(L);
}

template <typename FirstT, typename... OtherTs>
void fill_aspect_vector(std::vector<aspect> &V, FirstT F, OtherTs... O) {
  V.emplace_back(F);
  fill_aspect_vector(V, O...);
}
} // namespace detail

template <typename... AspectListT>
aspect_selector_t aspect_selector(AspectListT... AspectList) {
  std::vector<aspect> AllowList;
  AllowList.reserve(sizeof...(AspectList));

  detail::fill_aspect_vector(AllowList, AspectList...);

  return aspect_selector(AllowList, {});
}

template <aspect... AspectList> aspect_selector_t aspect_selector() {
  return aspect_selector({AspectList...}, {});
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
