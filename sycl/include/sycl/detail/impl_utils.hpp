//===- impl_utils.hpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace sycl {
inline namespace _V1 {
class handler;
namespace detail {
// Note! This class relies on the fact that all SYCL interface
// classes contain "impl" field that points to implementation object. "impl"
// field should be accessible from this class.
struct ImplUtils {
  // Helper function for extracting implementation from SYCL's interface
  // objects.
  template <class Obj>
  static const decltype(Obj::impl) &getSyclObjImplPtr(const Obj &SyclObj) {
    assert(SyclObj.impl && "every constructor should create an impl");
    return SyclObj.impl;
  }
  template <class Obj>
  static const decltype(*Obj::impl) &getSyclObjImpl(const Obj &SyclObj) {
    assert(SyclObj.impl && "every constructor should create an impl");
    return *SyclObj.impl;
  }

  // Helper function for creation SYCL interface objects from implementations.
  template <typename SyclObject, typename From>
  static SyclObject createSyclObjFromImpl(From &&from) {
    if constexpr (std::is_same_v<decltype(SyclObject::impl),
                                 std::shared_ptr<std::decay_t<From>>>)
      return SyclObject{from.shared_from_this()};
    else
      return SyclObject{std::forward<From>(from)};
  }
};

template <class Obj>
auto getSyclObjImpl(const Obj &SyclObj)
    -> decltype(ImplUtils::getSyclObjImpl(SyclObj)) {
  return ImplUtils::getSyclObjImpl(SyclObj);
}
template <class Obj>
auto getSyclObjImplPtr(const Obj &SyclObj)
    -> decltype(ImplUtils::getSyclObjImplPtr(SyclObj)) {
  return ImplUtils::getSyclObjImplPtr(SyclObj);
}

template <typename SyclObject, typename From>
SyclObject createSyclObjFromImpl(From &&from) {
  return ImplUtils::createSyclObjFromImpl<SyclObject>(std::forward<From>(from));
}

template <typename T, bool SupportedOnDevice = true> struct sycl_obj_hash {
  size_t operator()(const T &Obj) const {
    if constexpr (SupportedOnDevice) {
      auto &Impl = sycl::detail::getSyclObjImplPtr(Obj);
      return std::hash<std::decay_t<decltype(Impl)>>{}(Impl);
    } else {
#ifdef __SYCL_DEVICE_ONLY__
      (void)Obj;
      return 0;
#else
      auto &Impl = sycl::detail::getSyclObjImplPtr(Obj);
      return std::hash<std::decay_t<decltype(Impl)>>{}(Impl);
#endif
    }
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
