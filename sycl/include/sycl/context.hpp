//==---------------- context.hpp - SYCL context ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/async_handler.hpp>             // for async_handler
#include <sycl/backend_types.hpp>             // for backend, backend_return_t
#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT
#include <sycl/detail/helpers.hpp>            // for context_impl
#include <sycl/detail/info_desc_helpers.hpp>  // for is_context_info_desc
#include <sycl/detail/owner_less_base.hpp>    // for OwnerLessBase
#include <sycl/detail/pi.h>                   // for pi_native_handle
#include <sycl/platform.hpp>                  // for platform
#include <sycl/property_list.hpp>             // for property_list

#ifdef __SYCL_INTERNAL_API
#include <sycl/detail/cl.h>
#endif

#include <cstddef>     // for size_t
#include <memory>      // for shared_ptr, hash, opera...
#include <type_traits> // for add_pointer_t
#include <variant>     // for hash
#include <vector>      // for vector

// 4.6.2 Context class

namespace sycl {
inline namespace _V1 {
// Forward declarations
class device;
class platform;

namespace detail {
class context_impl;
}
template <backend Backend, class SyclT>
auto get_native(const SyclT &Obj) -> backend_return_t<Backend, SyclT>;

/// The context class represents a SYCL context on which kernel functions may
/// be executed.
///
/// \ingroup sycl_api
class __SYCL_EXPORT context : public detail::OwnerLessBase<context> {
public:
  /// Constructs a SYCL context instance using an instance of default_selector.
  ///
  /// The instance of default_selector is used to select the associated platform
  /// and device(s).
  /// SYCL properties are passed to the constructed SYCL context through
  /// PropList.
  ///
  /// \param PropList is an instance of property_list.
  explicit context(const property_list &PropList = {});

  /// Constructs a SYCL context instance using an instance of default_selector.
  ///
  /// The instance of default_selector is used to select the associated platform
  /// and device(s).
  /// The constructed SYCL context will use the AsyncHandler parameter to handle
  /// exceptions.
  /// SYCL properties are passed to the constructed SYCL context through
  /// PropList.
  ///
  /// \param AsyncHandler is an instance of async_handler.
  /// \param PropList is an instance of property_list.
  explicit context(const async_handler &AsyncHandler,
                   const property_list &PropList = {});

  /// Constructs a SYCL context instance using the provided device.
  ///
  /// Newly created context is associated with the Device and the SYCL platform
  /// that is associated with the Device.
  /// SYCL properties are passed to the constructed SYCL context through
  /// PropList.
  ///
  /// \param Device is an instance of SYCL device.
  /// \param PropList is an instance of property_list.
  explicit context(const device &Device, const property_list &PropList = {});

  /// Constructs a SYCL context instance using the provided device.
  ///
  /// Newly created context is associated with the Device and the SYCL platform
  /// that is associated with the Device.
  /// The constructed SYCL context will use the AsyncHandler parameter to handle
  /// exceptions.
  /// SYCL properties are passed to the constructed SYCL context through
  /// PropList.
  ///
  /// \param Device is an instance of SYCL device.
  /// \param AsyncHandler is an instance of async_handler.
  /// \param PropList is an instance of property_list.
  explicit context(const device &Device, async_handler AsyncHandler,
                   const property_list &PropList = {});

  /// Constructs a SYCL context instance using the provided platform.
  ///
  /// Newly created context is associated with the Platform and with each
  /// SYCL device that is associated with the Platform.
  /// SYCL properties are passed to the constructed SYCL context through
  /// PropList.
  ///
  /// \param Platform is an instance of SYCL platform.
  /// \param PropList is an instance of property_list.
  explicit context(const platform &Platform,
                   const property_list &PropList = {});

  /// Constructs a SYCL context instance using the provided platform.
  ///
  /// Newly created context is associated with the Platform and with each
  /// SYCL device that is associated with the Platform.
  /// The constructed SYCL context will use the AsyncHandler parameter to handle
  /// exceptions.
  /// SYCL properties are passed to the constructed SYCL context through
  /// PropList.
  ///
  /// \param Platform is an instance of SYCL platform.
  /// \param AsyncHandler is an instance of async_handler.
  /// \param PropList is an instance of property_list.
  explicit context(const platform &Platform, async_handler AsyncHandler,
                   const property_list &PropList = {});

  /// Constructs a SYCL context instance using list of devices.
  ///
  /// Newly created context will be associated with each SYCL device in the
  /// DeviceList. This requires that all SYCL devices in the list have the same
  /// associated SYCL platform.
  /// SYCL properties are passed to the constructed SYCL context through
  /// PropList.
  ///
  /// \param DeviceList is a list of SYCL device instances.
  /// \param PropList is an instance of property_list.
  explicit context(const std::vector<device> &DeviceList,
                   const property_list &PropList = {});

  /// Constructs a SYCL context instance using list of devices.
  ///
  /// Newly created context will be associated with each SYCL device in the
  /// DeviceList. This requires that all SYCL devices in the list have the same
  /// associated SYCL platform.
  /// The constructed SYCL context will use the AsyncHandler parameter to handle
  /// exceptions.
  /// SYCL properties are passed to the constructed SYCL context through
  /// PropList.
  ///
  /// \param DeviceList is a list of SYCL device instances.
  /// \param AsyncHandler is an instance of async_handler.
  /// \param PropList is an instance of property_list.
  explicit context(const std::vector<device> &DeviceList,
                   async_handler AsyncHandler,
                   const property_list &PropList = {});

  /// Constructs a SYCL context instance from OpenCL cl_context.
  ///
  /// ClContext is retained on SYCL context instantiation.
  /// The constructed SYCL context will use the AsyncHandler parameter to handle
  /// exceptions.
  ///
  /// \param ClContext is an instance of OpenCL cl_context.
  /// \param AsyncHandler is an instance of async_handler.
#ifdef __SYCL_INTERNAL_API
  context(cl_context ClContext, async_handler AsyncHandler = {});
#endif

  /// Queries this SYCL context for information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename detail::is_context_info_desc<Param>::return_type get_info() const;

  /// Queries this SYCL context for SYCL backend-specific information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename detail::is_backend_info_desc<Param>::return_type
  get_backend_info() const;

  context(const context &rhs) = default;

  context(context &&rhs) = default;

  context &operator=(const context &rhs) = default;

  context &operator=(context &&rhs) = default;

  bool operator==(const context &rhs) const { return impl == rhs.impl; }

  bool operator!=(const context &rhs) const { return !(*this == rhs); }

  /// Checks if this context has a property of type propertyT.
  ///
  /// \return true if this context has a property of type propertyT.
  template <typename propertyT> bool has_property() const noexcept {
    return getPropList().template has_property<propertyT>();
  }

  /// Gets the specified property of this context.
  ///
  /// Throws an exception with errc::invalid error code if this context does not
  /// have a property of type propertyT.
  ///
  /// \return a copy of the property of type propertyT.
  template <typename propertyT> propertyT get_property() const {
    return getPropList().template get_property<propertyT>();
  }

  /// Gets OpenCL interoperability context.
  ///
  /// The OpenCL cl_context handle is retained on return.
  ///
  /// \return a valid instance of OpenCL cl_context.
#ifdef __SYCL_INTERNAL_API
  cl_context get() const;
#endif

  /// Returns the backend associated with this context.
  ///
  /// \return the backend associated with this context.
  backend get_backend() const noexcept;

  /// Gets platform associated with this SYCL context.
  ///
  /// \return a valid instance of SYCL platform.
  platform get_platform() const;

  /// Gets devices associated with this SYCL context.
  ///
  /// \return a vector of valid SYCL device instances.
  std::vector<device> get_devices() const;

private:
  /// Constructs a SYCL context object from a valid context_impl instance.
  context(std::shared_ptr<detail::context_impl> Impl);

  pi_native_handle getNative() const;

  std::shared_ptr<detail::context_impl> impl;

  template <backend Backend, class SyclT>
  friend auto get_native(const SyclT &Obj) -> backend_return_t<Backend, SyclT>;

  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  const property_list &getPropList() const;
};

// context.hpp depends on exception.hpp but we can't define these ctors in
// exception.hpp while context is still an incomplete type.
inline exception::exception(context Ctx, std::error_code EC,
                            const std::string &WhatArg)
    : exception(EC, std::make_shared<context>(Ctx), WhatArg) {}

inline exception::exception(context Ctx, std::error_code EC,
                            const char *WhatArg)
    : exception(Ctx, EC, std::string(WhatArg)) {}

inline exception::exception(context Ctx, std::error_code EC)
    : exception(Ctx, EC, "") {}

inline exception::exception(context Ctx, int EV,
                            const std::error_category &ECat,
                            const char *WhatArg)
    : exception(Ctx, {EV, ECat}, std::string(WhatArg)) {}

inline exception::exception(context Ctx, int EV,
                            const std::error_category &ECat,
                            const std::string &WhatArg)
    : exception(Ctx, {EV, ECat}, WhatArg) {}

inline exception::exception(context Ctx, int EV,
                            const std::error_category &ECat)
    : exception(Ctx, EV, ECat, "") {}

} // namespace _V1
} // namespace sycl

namespace std {
template <> struct hash<sycl::context> {
  size_t operator()(const sycl::context &Context) const {
    return hash<std::shared_ptr<sycl::detail::context_impl>>()(
        sycl::detail::getSyclObjImpl(Context));
  }
};
} // namespace std
