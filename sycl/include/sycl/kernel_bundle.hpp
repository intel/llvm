//==------- kernel_bundle.hpp - SYCL kernel_bundle and free functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/backend_types.hpp>          // for backend, backend_return_t
#include <sycl/detail/export.hpp>          // for __SYCL_EXPORT
#include <sycl/detail/kernel_desc.hpp>     // for get_spec_constant_symboli...
#include <sycl/detail/owner_less_base.hpp> // for OwnerLessBase
#include <sycl/detail/string_view.hpp>
#include <sycl/detail/ur.hpp> // for cast
#include <sycl/device.hpp>    // for device
#include <sycl/handler.hpp>
#include <sycl/kernel.hpp>              // for kernel, kernel_bundle
#include <sycl/kernel_bundle_enums.hpp> // for bundle_state
#include <sycl/property_list.hpp>       // for property_list
#include <sycl/sycl_span.hpp>
#include <ur_api.h>

#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>     // PropertyT
#include <sycl/ext/oneapi/properties/property.hpp>       // build_options
#include <sycl/ext/oneapi/properties/property_value.hpp> // and log

#include <array>      // for array
#include <cstddef>    // for std::byte
#include <cstring>    // for size_t, memcpy
#include <functional> // for function
#include <iterator>   // for distance
#include <memory>     // for shared_ptr, operator==, hash
#if __has_include(<span>)
#include <span>
#endif
#include <string>      // for string
#include <type_traits> // for enable_if_t, remove_refer...
#include <utility>     // for move
#include <variant>     // for hash
#include <vector>      // for vector

namespace sycl {
inline namespace _V1 {
// Forward declaration
template <backend Backend> class backend_traits;
template <backend Backend, bundle_state State>
auto get_native(const kernel_bundle<State> &Obj)
    -> backend_return_t<Backend, kernel_bundle<State>>;
class context;

namespace detail {
class kernel_id_impl;
class kernel_impl;
} // namespace detail

template <typename KernelName> kernel_id get_kernel_id();

namespace ext::oneapi::experimental {
template <auto *Func>
std::enable_if_t<is_kernel_v<Func>, kernel_id> get_kernel_id();
} // namespace ext::oneapi::experimental

/// Objects of the class identify kernel is some kernel_bundle related APIs
///
/// \ingroup sycl_api
class __SYCL_EXPORT kernel_id : public detail::OwnerLessBase<kernel_id> {
public:
  kernel_id() = delete;

  /// \returns a null-terminated string which contains the kernel name
  const char *get_name() const noexcept;

  bool operator==(const kernel_id &RHS) const { return impl == RHS.impl; }

  bool operator!=(const kernel_id &RHS) const { return !(*this == RHS); }

private:
  kernel_id(const char *Name);

  kernel_id(const std::shared_ptr<detail::kernel_id_impl> &Impl)
      : impl(std::move(Impl)) {}

  std::shared_ptr<detail::kernel_id_impl> impl;

  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);

  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);
};

namespace detail {
class device_image_impl;

// The class is used as a base for device_image for "untemplating" public
// methods.
class __SYCL_EXPORT device_image_plain {
public:
  device_image_plain(const std::shared_ptr<device_image_impl> &Impl)
      : impl(Impl) {}

  device_image_plain(std::shared_ptr<device_image_impl> &&Impl)
      : impl(std::move(Impl)) {}

  bool operator==(const device_image_plain &RHS) const {
    return impl == RHS.impl;
  }

  bool operator!=(const device_image_plain &RHS) const {
    return !(*this == RHS);
  }

  bool has_kernel(const kernel_id &KernelID) const noexcept;

  bool has_kernel(const kernel_id &KernelID, const device &Dev) const noexcept;

  ur_native_handle_t getNative() const;

protected:
  std::shared_ptr<device_image_impl> impl;

  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);

  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);

  backend ext_oneapi_get_backend_impl() const noexcept;

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
  std::pair<const std::byte *, const std::byte *>
  ext_oneapi_get_backend_content_view_impl() const;
#endif // HAS_STD_BYTE
};
} // namespace detail

/// Objects of the class represents an instance of an image in a specific state.
template <sycl::bundle_state State>
class device_image : public detail::device_image_plain,
                     public detail::OwnerLessBase<device_image<State>> {
public:
  device_image() = delete;

  /// \returns true if the device_image contains the kernel identified by the
  /// KernelID
  bool has_kernel(const kernel_id &KernelID) const noexcept {
    return device_image_plain::has_kernel(KernelID);
  }

  /// \returns true if the device_image contains the kernel identified by the
  /// KernelID and is compatible with the passed Dev
  bool has_kernel(const kernel_id &KernelID, const device &Dev) const noexcept {
    return device_image_plain::has_kernel(KernelID, Dev);
  }

  backend ext_oneapi_get_backend() const noexcept {
    return device_image_plain::ext_oneapi_get_backend_impl();
  }

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
  template <sycl::bundle_state T = State,
            typename = std::enable_if_t<T == bundle_state::executable>>
  std::vector<std::byte> ext_oneapi_get_backend_content() const {
    const auto view =
        device_image_plain::ext_oneapi_get_backend_content_view_impl();
    return std::vector(view.first, view.second);
  }

#ifdef __cpp_lib_span
  template <sycl::bundle_state T = State,
            typename = std::enable_if_t<T == bundle_state::executable>>
  std::span<const std::byte> ext_oneapi_get_backend_content_view() const {
    const auto view =
        device_image_plain::ext_oneapi_get_backend_content_view_impl();
    return std::span<const std::byte>{view.first, view.second};
  }
#endif // __cpp_lib_span
#endif // _HAS_STD_BYTE

private:
  device_image(std::shared_ptr<detail::device_image_impl> Impl)
      : device_image_plain(std::move(Impl)) {}

  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);

  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);
};

namespace detail {
class kernel_bundle_impl;
using KernelBundleImplPtr = std::shared_ptr<detail::kernel_bundle_impl>;

// The class is used as a base for kernel_bundle to "untemplate" it's methods
class __SYCL_EXPORT kernel_bundle_plain {
public:
  kernel_bundle_plain(const detail::KernelBundleImplPtr &Impl)
      : impl(std::move(Impl)) {}

  bool operator==(const kernel_bundle_plain &RHS) const {
    return impl == RHS.impl;
  }

  bool operator!=(const kernel_bundle_plain &RHS) const {
    return !(*this == RHS);
  }

  bool empty() const noexcept;

  backend get_backend() const noexcept;

  context get_context() const noexcept;

  std::vector<device> get_devices() const noexcept;

  bool has_kernel(const kernel_id &KernelID) const noexcept;

  bool has_kernel(const kernel_id &KernelID, const device &Dev) const noexcept;

  std::vector<kernel_id> get_kernel_ids() const;

  bool contains_specialization_constants() const noexcept;

  bool native_specialization_constant() const noexcept;

  bool ext_oneapi_has_kernel(const std::string &name) {
    return ext_oneapi_has_kernel(detail::string_view{name});
  }

  kernel ext_oneapi_get_kernel(const std::string &name) {
    return ext_oneapi_get_kernel(detail::string_view{name});
  }

  std::string ext_oneapi_get_raw_kernel_name(const std::string &name) {
    return std::string{
        ext_oneapi_get_raw_kernel_name(detail::string_view{name}).c_str()};
  }

  bool ext_oneapi_has_device_global(const std::string &name) {
    return ext_oneapi_has_device_global(detail::string_view{name});
  }

  void *ext_oneapi_get_device_global_address(const std::string &name,
                                             const device &dev) {
    return ext_oneapi_get_device_global_address(detail::string_view{name}, dev);
  }

  size_t ext_oneapi_get_device_global_size(const std::string &name) {
    return ext_oneapi_get_device_global_size(detail::string_view{name});
  }

protected:
  // \returns a kernel object which represents the kernel identified by
  // kernel_id passed
  kernel get_kernel(const kernel_id &KernelID) const;

  // \returns an iterator to the first device image kernel_bundle contains
  const device_image_plain *begin() const;

  // \returns an iterator to the last device image kernel_bundle contains
  const device_image_plain *end() const;

  bool has_specialization_constant_impl(const char *SpecName) const noexcept;

  void set_specialization_constant_impl(const char *SpecName, void *Value,
                                        size_t Size) noexcept;

  void get_specialization_constant_impl(const char *SpecName,
                                        void *Value) const noexcept;

  // \returns a bool value which indicates if specialization constant was set to
  // a value different from default value.
  bool is_specialization_constant_set(const char *SpecName) const noexcept;

  detail::KernelBundleImplPtr impl;

private:
  bool ext_oneapi_has_kernel(detail::string_view name);
  kernel ext_oneapi_get_kernel(detail::string_view name);
  detail::string ext_oneapi_get_raw_kernel_name(detail::string_view name);

  bool ext_oneapi_has_device_global(detail::string_view name);
  void *ext_oneapi_get_device_global_address(detail::string_view name,
                                             const device &dev);
  size_t ext_oneapi_get_device_global_size(detail::string_view name);
};

} // namespace detail

/// The kernel_bundle class represents collection of device images in a
/// particular state
///
/// \ingroup sycl_api
template <bundle_state State>
class kernel_bundle : public detail::kernel_bundle_plain,
                      public detail::OwnerLessBase<kernel_bundle<State>> {
public:
  using device_image_iterator = const device_image<State> *;

  kernel_bundle() = delete;

  /// \returns true if the kernel_bundles contains no device images
  template <
      bundle_state _State = State,
      typename = std::enable_if_t<_State != bundle_state::ext_oneapi_source>>
  bool empty() const noexcept {
    return kernel_bundle_plain::empty();
  }

  /// \returns the backend associated with the kernel bundle
  backend get_backend() const noexcept {
    return kernel_bundle_plain::get_backend();
  }

  /// \returns the context associated with the kernel_bundle
  context get_context() const noexcept {
    return kernel_bundle_plain::get_context();
  }

  /// \returns devices associated with the kernel_bundle
  std::vector<device> get_devices() const noexcept {
    return kernel_bundle_plain::get_devices();
  }

  /// \returns true if the kernel_bundle contains the kernel identified by
  /// kernel_id passed
  template <
      bundle_state _State = State,
      typename = std::enable_if_t<_State != bundle_state::ext_oneapi_source>>
  bool has_kernel(const kernel_id &KernelID) const noexcept {
    return kernel_bundle_plain::has_kernel(KernelID);
  }

  /// \returns true if the kernel_bundle contains the kernel identified by
  /// kernel_id passed and if this kernel is compatible with the device
  /// specified
  template <
      bundle_state _State = State,
      typename = std::enable_if_t<_State != bundle_state::ext_oneapi_source>>
  bool has_kernel(const kernel_id &KernelID, const device &Dev) const noexcept {
    return kernel_bundle_plain::has_kernel(KernelID, Dev);
  }

  /// \returns true only if the kernel bundle contains the kernel identified by
  /// KernelName.
  template <
      typename KernelName, bundle_state _State = State,
      typename = std::enable_if_t<_State != bundle_state::ext_oneapi_source>>
  bool has_kernel() const noexcept {
    return has_kernel(get_kernel_id<KernelName>());
  }

  /// \returns true only if the kernel bundle contains the kernel identified by
  /// KernelName and if that kernel is compatible with the device Dev.
  template <
      typename KernelName, bundle_state _State = State,
      typename = std::enable_if_t<_State != bundle_state::ext_oneapi_source>>
  bool has_kernel(const device &Dev) const noexcept {
    return has_kernel(get_kernel_id<KernelName>(), Dev);
  }

  /// \returns a vector of kernel_id's that contained in the kernel_bundle
  template <
      bundle_state _State = State,
      typename = std::enable_if_t<_State != bundle_state::ext_oneapi_source>>
  std::vector<kernel_id> get_kernel_ids() const {
    return kernel_bundle_plain::get_kernel_ids();
  }

  /// \returns true if the kernel_bundle contains at least one device image
  /// which uses specialization constants
  template <
      bundle_state _State = State,
      typename = std::enable_if_t<_State != bundle_state::ext_oneapi_source>>
  bool contains_specialization_constants() const noexcept {
    return kernel_bundle_plain::contains_specialization_constants();
  }

  /// \returns true if all specialization constants which are used in the
  /// kernel_bundle are "native specialization constants in all device images
  template <
      bundle_state _State = State,
      typename = std::enable_if_t<_State != bundle_state::ext_oneapi_source>>
  bool native_specialization_constant() const noexcept {
    return kernel_bundle_plain::native_specialization_constant();
  }

  /// \returns a kernel object which represents the kernel identified by
  /// kernel_id passed
  template <bundle_state _State = State,
            typename = std::enable_if_t<_State == bundle_state::executable>>
  kernel get_kernel(const kernel_id &KernelID) const {
    return detail::kernel_bundle_plain::get_kernel(KernelID);
  }

  /// \returns a kernel object which represents the kernel identified by
  /// KernelName.
  template <typename KernelName, bundle_state _State = State,
            typename = std::enable_if_t<_State == bundle_state::executable>>
  kernel get_kernel() const {
    return detail::kernel_bundle_plain::get_kernel(get_kernel_id<KernelName>());
  }

  /// \returns true if any device image in the kernel_bundle uses specialization
  /// constant whose address is SpecName
  template <
      auto &SpecName, bundle_state _State = State,
      typename = std::enable_if_t<_State != bundle_state::ext_oneapi_source>>
  bool has_specialization_constant() const noexcept {
    const char *SpecSymName = detail::get_spec_constant_symbolic_ID<SpecName>();
    return has_specialization_constant_impl(SpecSymName);
  }

  /// Sets the value of the specialization constant whose address is SpecName
  /// for this bundle. If the specialization constant’s value was previously set
  /// in this bundle, the value is overwritten.
  template <auto &SpecName, bundle_state _State = State,
            typename = std::enable_if_t<_State == bundle_state::input>>
  void set_specialization_constant(
      typename std::remove_reference_t<decltype(SpecName)>::value_type Value) {
    const char *SpecSymName = detail::get_spec_constant_symbolic_ID<SpecName>();
    set_specialization_constant_impl(SpecSymName, &Value,
                                     sizeof(decltype(Value)));
  }

  /// \returns the value of the specialization constant whose address is
  /// SpecName for this kernel bundle.
  template <
      auto &SpecName, bundle_state _State = State,
      typename = std::enable_if_t<_State != bundle_state::ext_oneapi_source>>
  typename std::remove_reference_t<decltype(SpecName)>::value_type
  get_specialization_constant() const {
    using SCType =
        typename std::remove_reference_t<decltype(SpecName)>::value_type;

    const char *SpecSymName = detail::get_spec_constant_symbolic_ID<SpecName>();
    SCType Res{SpecName.getDefaultValue()};
    if (!is_specialization_constant_set(SpecSymName))
      return Res;

    std::array<char, sizeof(SCType)> RetValue;
    get_specialization_constant_impl(SpecSymName, RetValue.data());
    std::memcpy(&Res, RetValue.data(), sizeof(SCType));

    return Res;
  }

  /// \returns an iterator to the first device image kernel_bundle contains
  template <
      bundle_state _State = State,
      typename = std::enable_if_t<_State != bundle_state::ext_oneapi_source>>
  device_image_iterator begin() const {
    return reinterpret_cast<device_image_iterator>(
        kernel_bundle_plain::begin());
  }

  /// \returns an iterator to the last device image kernel_bundle contains
  template <
      bundle_state _State = State,
      typename = std::enable_if_t<_State != bundle_state::ext_oneapi_source>>
  device_image_iterator end() const {
    return reinterpret_cast<device_image_iterator>(kernel_bundle_plain::end());
  }

  /////////////////////////
  // ext_oneapi_has_kernel
  //  only true if created from source and has this kernel
  /////////////////////////
  template <bundle_state _State = State,
            typename = std::enable_if_t<_State == bundle_state::executable>>
  bool ext_oneapi_has_kernel(const std::string &name) {
    return detail::kernel_bundle_plain::ext_oneapi_has_kernel(name);
  }

  template <auto *Func>
  std::enable_if_t<ext::oneapi::experimental::is_kernel_v<Func>, bool>
  ext_oneapi_has_kernel() {
    return has_kernel(ext::oneapi::experimental::get_kernel_id<Func>());
  }

  template <auto *Func>
  std::enable_if_t<ext::oneapi::experimental::is_kernel_v<Func>, bool>
  ext_oneapi_has_kernel(const device &dev) {
    return has_kernel(ext::oneapi::experimental::get_kernel_id<Func>(), dev);
  }

  template <auto *Func, bundle_state _State = State,
            typename = std::enable_if_t<_State == bundle_state::executable>>
  std::enable_if_t<ext::oneapi::experimental::is_kernel_v<Func>, kernel>
  ext_oneapi_get_kernel() {
    return detail::kernel_bundle_plain::get_kernel(
        ext::oneapi::experimental::get_kernel_id<Func>());
  }

  /////////////////////////
  // ext_oneapi_get_kernel
  //  kernel_bundle must be created from source, throws if not present
  /////////////////////////
  template <bundle_state _State = State,
            typename = std::enable_if_t<_State == bundle_state::executable>>
  kernel ext_oneapi_get_kernel(const std::string &name) {
    return detail::kernel_bundle_plain::ext_oneapi_get_kernel(name);
  }

  /////////////////////////
  // ext_oneapi_get_raw_kernel_name
  //  kernel_bundle must be created from source, throws if not present
  /////////////////////////
  template <bundle_state _State = State,
            typename = std::enable_if_t<_State == bundle_state::executable>>
  std::string ext_oneapi_get_raw_kernel_name(const std::string &name) {
    return detail::kernel_bundle_plain::ext_oneapi_get_raw_kernel_name(name);
  }

  /////////////////////////
  // ext_oneapi_has_device_global
  //  only true if kernel_bundle was created from source and has this device
  //  global
  /////////////////////////
  template <bundle_state _State = State,
            typename = std::enable_if_t<_State == bundle_state::executable>>
  bool ext_oneapi_has_device_global(const std::string &name) {
    return detail::kernel_bundle_plain::ext_oneapi_has_device_global(name);
  }

  /////////////////////////
  // ext_oneapi_get_device_global_address
  //  kernel_bundle must be created from source, throws if bundle was not built
  //  for this device, or device global is either not present or has
  //  `device_image_scope` property.
  //  Returns a USM pointer to the variable's initialized storage on the device.
  /////////////////////////
  template <bundle_state _State = State,
            typename = std::enable_if_t<_State == bundle_state::executable>>
  void *ext_oneapi_get_device_global_address(const std::string &name,
                                             const device &dev) {
    return detail::kernel_bundle_plain::ext_oneapi_get_device_global_address(
        name, dev);
  }

  /////////////////////////
  // ext_oneapi_get_device_global_size
  //  kernel_bundle must be created from source, throws if device global is not
  //  present. Returns the variable's size in bytes.
  /////////////////////////
  template <bundle_state _State = State,
            typename = std::enable_if_t<_State == bundle_state::executable>>
  size_t ext_oneapi_get_device_global_size(const std::string &name) {
    return detail::kernel_bundle_plain::ext_oneapi_get_device_global_size(name);
  }

private:
  kernel_bundle(detail::KernelBundleImplPtr Impl)
      : kernel_bundle_plain(std::move(Impl)) {}

  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);
  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);

  template <backend Backend, bundle_state StateB>
  friend auto get_native(const kernel_bundle<StateB> &Obj)
      -> backend_return_t<Backend, kernel_bundle<StateB>>;

  template <backend Backend>
  backend_return_t<Backend, kernel_bundle<State>> getNative() const {
    // NOTE: implementation assumes that the return type is a
    // derivative of std::vector.
    backend_return_t<Backend, kernel_bundle<State>> ReturnValue;
    ReturnValue.reserve(std::distance(begin(), end()));

    for (const device_image<State> &DevImg : *this) {
      ReturnValue.push_back(
          detail::ur::cast<typename decltype(ReturnValue)::value_type>(
              DevImg.getNative()));
    }

    return ReturnValue;
  }
};
template <bundle_state State>
kernel_bundle(kernel_bundle<State> &&) -> kernel_bundle<State>;

/////////////////////////
// get_kernel_id API
/////////////////////////

namespace detail {
// Internal non-template versions of get_kernel_id API which is used by public
// onces
__SYCL_EXPORT kernel_id get_kernel_id_impl(string_view KernelName);
} // namespace detail

/// \returns the kernel_id associated with the KernelName
template <typename KernelName> kernel_id get_kernel_id() {
  // FIXME: This must fail at link-time if KernelName not in any available
  // translation units.
  return detail::get_kernel_id_impl(
      detail::string_view{detail::getKernelName<KernelName>()});
}

/// \returns a vector with all kernel_id's defined in the application
__SYCL_EXPORT std::vector<kernel_id> get_kernel_ids();

/////////////////////////
// get_kernel_bundle API
/////////////////////////

namespace detail {

// Internal non-template versions of get_kernel_bundle API which is used by
// public onces
__SYCL_EXPORT detail::KernelBundleImplPtr
get_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                       bundle_state State);

__SYCL_EXPORT detail::KernelBundleImplPtr
get_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                       const sycl::span<char> &Bytes, bundle_state State);

__SYCL_EXPORT const std::vector<device>
removeDuplicateDevices(const std::vector<device> &Devs);

} // namespace detail

/// A kernel bundle in state State which contains all of the kernels in the
/// application which are compatible with at least one of the devices in Devs.
/// This does not include any device built-in kernels. The bundle’s set of
/// associated devices is Devs.
template <bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &Ctx,
                                       const std::vector<device> &Devs) {
  std::vector<device> UniqueDevices = detail::removeDuplicateDevices(Devs);

  detail::KernelBundleImplPtr Impl =
      detail::get_kernel_bundle_impl(Ctx, UniqueDevices, State);

  return detail::createSyclObjFromImpl<kernel_bundle<State>>(Impl);
}

template <bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &Ctx) {
  return get_kernel_bundle<State>(Ctx, Ctx.get_devices());
}

namespace detail {

// Internal non-template versions of get_kernel_bundle API which is used by
// public onces
__SYCL_EXPORT detail::KernelBundleImplPtr
get_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                       const std::vector<kernel_id> &KernelIDs,
                       bundle_state State);
} // namespace detail

/// \returns a kernel bundle in state State which contains all of the device
/// images that are compatible with at least one of the devices in Devs, further
/// filtered to contain only those device images that contain kernels with the
/// given identifiers. These identifiers may represent kernels that are defined
/// in the application, device built-in kernels, or a mixture of the two. Since
/// the device images may group many kernels together, the returned bundle may
/// contain additional kernels beyond those that are requested in KernelIDs. The
/// bundle’s set of associated devices is Devs.
template <bundle_state State>
kernel_bundle<State>
get_kernel_bundle(const context &Ctx, const std::vector<device> &Devs,
                  const std::vector<kernel_id> &KernelIDs) {
  std::vector<device> UniqueDevices = detail::removeDuplicateDevices(Devs);

  detail::KernelBundleImplPtr Impl =
      detail::get_kernel_bundle_impl(Ctx, UniqueDevices, KernelIDs, State);
  return detail::createSyclObjFromImpl<kernel_bundle<State>>(Impl);
}

template <bundle_state State>
kernel_bundle<State>
get_kernel_bundle(const context &Ctx, const std::vector<kernel_id> &KernelIDs) {
  return get_kernel_bundle<State>(Ctx, Ctx.get_devices(), KernelIDs);
}

template <typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &Ctx) {
  return get_kernel_bundle<State>(Ctx, Ctx.get_devices(),
                                  {get_kernel_id<KernelName>()});
}

template <typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &Ctx,
                                       const std::vector<device> &Devs) {
  return get_kernel_bundle<State>(Ctx, Devs, {get_kernel_id<KernelName>()});
}

// For free functions.
namespace ext::oneapi::experimental {
template <auto *Func, bundle_state State>
std::enable_if_t<is_kernel_v<Func>, kernel_bundle<State>>
get_kernel_bundle(const context &Ctx, const std::vector<device> &Devs) {
  return get_kernel_bundle<State>(Ctx, Devs, {get_kernel_id<Func>()});
}

template <auto *Func, bundle_state State>
std::enable_if_t<is_kernel_v<Func>, kernel_bundle<State>>
get_kernel_bundle(const context &Ctx) {
  return get_kernel_bundle<State>(Ctx, Ctx.get_devices(),
                                  {get_kernel_id<Func>()});
}

template <auto *Func>
std::enable_if_t<is_kernel_v<Func>, kernel_id> get_kernel_id() {
  return get_kernel_id_impl(detail::string_view(
      detail::FreeFunctionInfoData<Func>::getFunctionName()));
}

} // namespace ext::oneapi::experimental

namespace detail {

// Stable selector function type for passing thru library boundaries
using DevImgSelectorImpl =
    std::function<bool(const std::shared_ptr<device_image_impl> &DevImgImpl)>;

// Internal non-template versions of get_kernel_bundle API which is used by
// public onces
__SYCL_EXPORT detail::KernelBundleImplPtr
get_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                       bundle_state State, const DevImgSelectorImpl &Selector);

// Internal non-template versions of get_empty_interop_kernel_bundle API which
// is used by public onces
__SYCL_EXPORT detail::KernelBundleImplPtr
get_empty_interop_kernel_bundle_impl(const context &Ctx,
                                     const std::vector<device> &Devs);

/// make_kernel may need an empty interop kernel bundle. This function supplies
/// this.
template <bundle_state State>
kernel_bundle<State> get_empty_interop_kernel_bundle(const context &Ctx) {
  detail::KernelBundleImplPtr Impl =
      detail::get_empty_interop_kernel_bundle_impl(Ctx, Ctx.get_devices());
  return detail::createSyclObjFromImpl<sycl::kernel_bundle<State>>(Impl);
}
} // namespace detail

/// A kernel bundle in state State which contains all of the device images for
/// which the selector returns true.
template <bundle_state State, typename SelectorT>
kernel_bundle<State> get_kernel_bundle(const context &Ctx,
                                       const std::vector<device> &Devs,
                                       SelectorT Selector) {
  std::vector<device> UniqueDevices = detail::removeDuplicateDevices(Devs);

  detail::DevImgSelectorImpl SelectorWrapper =
      [Selector](const std::shared_ptr<detail::device_image_impl> &DevImg) {
        return Selector(
            detail::createSyclObjFromImpl<sycl::device_image<State>>(DevImg));
      };

  detail::KernelBundleImplPtr Impl = detail::get_kernel_bundle_impl(
      Ctx, UniqueDevices, State, SelectorWrapper);

  return detail::createSyclObjFromImpl<sycl::kernel_bundle<State>>(Impl);
}

template <bundle_state State, typename SelectorT>
kernel_bundle<State> get_kernel_bundle(const context &Ctx, SelectorT Selector) {
  return get_kernel_bundle<State>(Ctx, Ctx.get_devices(), Selector);
}

/////////////////////////
// has_kernel_bundle API
/////////////////////////

namespace detail {

__SYCL_EXPORT bool has_kernel_bundle_impl(const context &Ctx,
                                          const std::vector<device> &Devs,
                                          bundle_state State);

__SYCL_EXPORT bool
has_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                       const std::vector<kernel_id> &kernelIds,
                       bundle_state State);
} // namespace detail

/// \returns true if the following is true:
/// The application defines at least one kernel that is compatible with at
/// least one of the devices in Devs, and that kernel can be represented in a
/// device image of state State.
///
/// If State is bundle_state::input, all devices in Devs have
/// aspect::online_compiler.
///
/// If State is bundle_state::object, all devices in Devs have
/// aspect::online_linker.
template <bundle_state State>
bool has_kernel_bundle(const context &Ctx, const std::vector<device> &Devs) {
  return detail::has_kernel_bundle_impl(Ctx, Devs, State);
}

template <bundle_state State>
bool has_kernel_bundle(const context &Ctx, const std::vector<device> &Devs,
                       const std::vector<kernel_id> &KernelIDs) {
  return detail::has_kernel_bundle_impl(Ctx, Devs, KernelIDs, State);
}

template <bundle_state State> bool has_kernel_bundle(const context &Ctx) {
  return has_kernel_bundle<State>(Ctx, Ctx.get_devices());
}

template <bundle_state State>
bool has_kernel_bundle(const context &Ctx,
                       const std::vector<kernel_id> &KernelIDs) {
  return has_kernel_bundle<State>(Ctx, Ctx.get_devices(), KernelIDs);
}

template <typename KernelName, bundle_state State>
bool has_kernel_bundle(const context &Ctx) {
  return has_kernel_bundle<State>(Ctx, {get_kernel_id<KernelName>()});
}

template <typename KernelName, bundle_state State>
bool has_kernel_bundle(const context &Ctx, const std::vector<device> &Devs) {
  return has_kernel_bundle<State>(Ctx, Devs, {get_kernel_id<KernelName>()});
}

namespace ext::oneapi::experimental {
template <auto *Func, bundle_state State>
std::enable_if_t<is_kernel_v<Func>, bool>
has_kernel_bundle(const context &Ctx) {
  return has_kernel_bundle<State>(Ctx, {get_kernel_id<Func>()});
}

template <auto *Func, bundle_state State>
std::enable_if_t<is_kernel_v<Func>, bool>
has_kernel_bundle(const context &Ctx, const std::vector<device> &Devs) {
  return has_kernel_bundle<State>(Ctx, Devs, {get_kernel_id<Func>()});
}
} // namespace ext::oneapi::experimental

/////////////////////////
// is_compatible API
/////////////////////////

/// \returns true if all of the kernels identified by KernelIDs are compatible
/// with the device Dev.
__SYCL_EXPORT bool is_compatible(const std::vector<kernel_id> &KernelIDs,
                                 const device &Dev);

template <typename KernelName> bool is_compatible(const device &Dev) {
  return is_compatible({get_kernel_id<KernelName>()}, Dev);
}

namespace ext::oneapi::experimental {
template <auto *Func>
std::enable_if_t<is_kernel_v<Func>, bool> is_compatible(const device &Dev) {
  return is_compatible({get_kernel_id<Func>()}, Dev);
}
} // namespace ext::oneapi::experimental

/////////////////////////
// join API
/////////////////////////

namespace detail {

__SYCL_EXPORT std::shared_ptr<detail::kernel_bundle_impl>
join_impl(const std::vector<detail::KernelBundleImplPtr> &Bundles,
          bundle_state State);
} // namespace detail

/// \returns a new kernel bundle that represents the union of all the device
/// images in the input bundles with duplicates removed.
template <sycl::bundle_state State>
std::enable_if_t<State != sycl::bundle_state::ext_oneapi_source,
                 sycl::kernel_bundle<State>>
join(const std::vector<sycl::kernel_bundle<State>> &Bundles) {
  // Convert kernel_bundle<State> to impls to abstract template parameter away
  std::vector<detail::KernelBundleImplPtr> KernelBundleImpls;
  KernelBundleImpls.reserve(Bundles.size());
  for (const sycl::kernel_bundle<State> &Bundle : Bundles)
    KernelBundleImpls.push_back(detail::getSyclObjImpl(Bundle));

  std::shared_ptr<detail::kernel_bundle_impl> Impl =
      detail::join_impl(KernelBundleImpls, State);
  return detail::createSyclObjFromImpl<kernel_bundle<State>>(Impl);
}

/////////////////////////
// compile API
/////////////////////////

namespace detail {

__SYCL_EXPORT std::shared_ptr<detail::kernel_bundle_impl>
compile_impl(const kernel_bundle<bundle_state::input> &InputBundle,
             const std::vector<device> &Devs, const property_list &PropList);
}

/// \returns a new kernel_bundle which contains the device images from
/// InputBundle that are translated into one or more new device images of state
/// bundle_state::object. The new bundle represents all of the kernels in
/// InputBundles that are compatible with at least one of the devices in Devs.
inline kernel_bundle<bundle_state::object>
compile(const kernel_bundle<bundle_state::input> &InputBundle,
        const std::vector<device> &Devs, const property_list &PropList = {}) {
  std::vector<device> UniqueDevices = detail::removeDuplicateDevices(Devs);

  detail::KernelBundleImplPtr Impl =
      detail::compile_impl(InputBundle, UniqueDevices, PropList);
  return detail::createSyclObjFromImpl<
      kernel_bundle<sycl::bundle_state::object>>(std::move(Impl));
}

inline kernel_bundle<bundle_state::object>
compile(const kernel_bundle<bundle_state::input> &InputBundle,
        const property_list &PropList = {}) {
  return compile(InputBundle, InputBundle.get_devices(), PropList);
}

/////////////////////////
// link API
/////////////////////////

namespace detail {
__SYCL_EXPORT std::vector<sycl::device> find_device_intersection(
    const std::vector<kernel_bundle<bundle_state::object>> &ObjectBundles);

__SYCL_EXPORT std::shared_ptr<detail::kernel_bundle_impl>
link_impl(const std::vector<kernel_bundle<bundle_state::object>> &ObjectBundles,
          const std::vector<device> &Devs, const property_list &PropList);
} // namespace detail

/// \returns a new kernel_bundle which contains the device images from the
/// ObjectBundles that are translated into one or more new device images of
/// state bundle_state::executable The new bundle represents all of the kernels
/// in ObjectBundles that are compatible with at least one of the devices in
/// Devs.
inline kernel_bundle<bundle_state::executable>
link(const std::vector<kernel_bundle<bundle_state::object>> &ObjectBundles,
     const std::vector<device> &Devs, const property_list &PropList = {}) {
  std::vector<device> UniqueDevices = detail::removeDuplicateDevices(Devs);

  detail::KernelBundleImplPtr Impl =
      detail::link_impl(ObjectBundles, UniqueDevices, PropList);
  return detail::createSyclObjFromImpl<
      kernel_bundle<sycl::bundle_state::executable>>(std::move(Impl));
}

inline kernel_bundle<bundle_state::executable>
link(const kernel_bundle<bundle_state::object> &ObjectBundle,
     const property_list &PropList = {}) {
  return link(std::vector<kernel_bundle<bundle_state::object>>{ObjectBundle},
              ObjectBundle.get_devices(), PropList);
}

inline kernel_bundle<bundle_state::executable>
link(const std::vector<kernel_bundle<bundle_state::object>> &ObjectBundles,
     const property_list &PropList = {}) {
  std::vector<sycl::device> IntersectDevices =
      find_device_intersection(ObjectBundles);
  return link(ObjectBundles, IntersectDevices, PropList);
}

inline kernel_bundle<bundle_state::executable>
link(const kernel_bundle<bundle_state::object> &ObjectBundle,
     const std::vector<device> &Devs, const property_list &PropList = {}) {
  return link(std::vector<kernel_bundle<bundle_state::object>>{ObjectBundle},
              Devs, PropList);
}

/////////////////////////
// build API
/////////////////////////

namespace detail {
__SYCL_EXPORT std::shared_ptr<detail::kernel_bundle_impl>
build_impl(const kernel_bundle<bundle_state::input> &InputBundle,
           const std::vector<device> &Devs, const property_list &PropList);
}

/// \returns a new kernel_bundle which contains device images that are
/// translated into one ore more new device images of state
/// bundle_state::executable. The new bundle represents all of the kernels in
/// InputBundle that are compatible with at least one of the devices in Devs.
inline kernel_bundle<bundle_state::executable>
build(const kernel_bundle<bundle_state::input> &InputBundle,
      const std::vector<device> &Devs, const property_list &PropList = {}) {
  std::vector<device> UniqueDevices = detail::removeDuplicateDevices(Devs);

  detail::KernelBundleImplPtr Impl =
      detail::build_impl(InputBundle, UniqueDevices, PropList);
  return detail::createSyclObjFromImpl<
      kernel_bundle<sycl::bundle_state::executable>>(std::move(Impl));
}

inline kernel_bundle<bundle_state::executable>
build(const kernel_bundle<bundle_state::input> &InputBundle,
      const property_list &PropList = {}) {
  return build(InputBundle, InputBundle.get_devices(), PropList);
}

namespace ext::oneapi::experimental {

namespace detail {
struct create_bundle_from_source_props;
struct build_source_bundle_props;
} // namespace detail

/////////////////////////
// PropertyT syclex::include_files
/////////////////////////
struct include_files
    : detail::run_time_property_key<include_files,
                                    detail::PropKind::IncludeFiles> {
  include_files() {}
  include_files(const std::string &name, const std::string &content) {
    record.emplace_back(name, content);
  }
  void add(const std::string &name, const std::string &content) {
    if (std::find_if(record.begin(), record.end(), [&name](auto &p) {
          return p.first == name;
        }) != record.end()) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Include file '" + name +
                                "' is already registered");
    }
    record.emplace_back(name, content);
  }
  std::vector<std::pair<std::string, std::string>> record;
};
using include_files_key = include_files;

template <>
struct is_property_key_of<include_files_key,
                          detail::create_bundle_from_source_props>
    : std::true_type {};

/////////////////////////
// PropertyT syclex::build_options
/////////////////////////
struct build_options
    : detail::run_time_property_key<build_options,
                                    detail::PropKind::BuildOptions> {
  std::vector<std::string> opts;
  build_options() {}
  build_options(const std::string &optsArg) : opts{optsArg} {}
  build_options(const std::vector<std::string> &optsArg) : opts(optsArg) {}
  void add(const std::string &opt) { opts.push_back(opt); }
};
using build_options_key = build_options;

template <>
struct is_property_key_of<build_options_key, detail::build_source_bundle_props>
    : std::true_type {};

/////////////////////////
// PropertyT syclex::save_log
/////////////////////////
struct save_log
    : detail::run_time_property_key<save_log, detail::PropKind::BuildLog> {
  std::string *log;
  save_log(std::string *logArg) : log(logArg) {}
};
using save_log_key = save_log;

template <>
struct is_property_key_of<save_log_key, detail::build_source_bundle_props>
    : std::true_type {};

/////////////////////////
// PropertyT syclex::registered_names
/////////////////////////
struct registered_names
    : detail::run_time_property_key<registered_names,
                                    detail::PropKind::RegisteredNames> {
  std::vector<std::string> names;
  registered_names() {}
  registered_names(const std::string &name) : names{name} {}
  registered_names(const std::vector<std::string> &names) : names{names} {}
  void add(const std::string &name) { names.push_back(name); }
};
using registered_names_key = registered_names;

template <>
struct is_property_key_of<registered_names_key,
                          detail::build_source_bundle_props> : std::true_type {
};

namespace detail {
// forward decls

__SYCL_EXPORT kernel_bundle<bundle_state::ext_oneapi_source>
make_kernel_bundle_from_source(
    const context &SyclContext, source_language Language,
    sycl::detail::string_view Source,
    std::vector<std::pair<sycl::detail::string_view, sycl::detail::string_view>>
        IncludePairsVec);

inline kernel_bundle<bundle_state::ext_oneapi_source>
make_kernel_bundle_from_source(
    const context &SyclContext, source_language Language,
    const std::string &Source,
    std::vector<std::pair<std::string, std::string>> IncludePairsVec) {
  size_t n = IncludePairsVec.size();
  std::vector<std::pair<sycl::detail::string_view, sycl::detail::string_view>>
      PairVec;
  PairVec.reserve(n);
  for (auto &Pair : IncludePairsVec)
    PairVec.push_back({sycl::detail::string_view{Pair.first},
                       sycl::detail::string_view{Pair.second}});

  return make_kernel_bundle_from_source(
      SyclContext, Language, sycl::detail::string_view{Source}, PairVec);
}

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
__SYCL_EXPORT kernel_bundle<bundle_state::ext_oneapi_source>
make_kernel_bundle_from_source(
    const context &SyclContext, source_language Language,
    const std::vector<std::byte> &Bytes,
    std::vector<std::pair<sycl::detail::string_view, sycl::detail::string_view>>
        IncludePairsVec);

inline kernel_bundle<bundle_state::ext_oneapi_source>
make_kernel_bundle_from_source(
    const context &SyclContext, source_language Language,
    const std::vector<std::byte> &Bytes,
    std::vector<std::pair<std::string, std::string>> IncludePairsVec) {
  size_t n = IncludePairsVec.size();
  std::vector<std::pair<sycl::detail::string_view, sycl::detail::string_view>>
      PairVec;
  PairVec.reserve(n);
  for (auto &Pair : IncludePairsVec)
    PairVec.push_back({sycl::detail::string_view{Pair.first},
                       sycl::detail::string_view{Pair.second}});

  return make_kernel_bundle_from_source(SyclContext, Language, Bytes, PairVec);
}
#endif

__SYCL_EXPORT kernel_bundle<bundle_state::executable> build_from_source(
    kernel_bundle<bundle_state::ext_oneapi_source> &SourceKB,
    const std::vector<device> &Devices,
    const std::vector<sycl::detail::string_view> &BuildOptions,
    sycl::detail::string *LogPtr,
    const std::vector<sycl::detail::string_view> &RegisteredKernelNames);

inline kernel_bundle<bundle_state::executable>
build_from_source(kernel_bundle<bundle_state::ext_oneapi_source> &SourceKB,
                  const std::vector<device> &Devices,
                  const std::vector<std::string> &BuildOptions,
                  std::string *LogPtr,
                  const std::vector<std::string> &RegisteredKernelNames) {
  std::vector<sycl::detail::string_view> Options;
  Options.reserve(BuildOptions.size());
  for (const std::string &opt : BuildOptions)
    Options.push_back(sycl::detail::string_view{opt});

  std::vector<sycl::detail::string_view> KernelNames;
  for (const std::string &name : RegisteredKernelNames)
    KernelNames.push_back(sycl::detail::string_view{name});

  if (LogPtr) {
    sycl::detail::string Log;
    auto result =
        build_from_source(SourceKB, Devices, Options, &Log, KernelNames);
    *LogPtr = Log.c_str();
    return result;
  }
  return build_from_source(SourceKB, Devices, Options, nullptr, KernelNames);
}

__SYCL_EXPORT kernel_bundle<bundle_state::object> compile_from_source(
    kernel_bundle<bundle_state::ext_oneapi_source> &SourceKB,
    const std::vector<device> &Devices,
    const std::vector<sycl::detail::string_view> &CompileOptions,
    sycl::detail::string *LogPtr,
    const std::vector<sycl::detail::string_view> &RegisteredKernelNames);

inline kernel_bundle<bundle_state::object>
compile_from_source(kernel_bundle<bundle_state::ext_oneapi_source> &SourceKB,
                    const std::vector<device> &Devices,
                    const std::vector<std::string> &CompileOptions,
                    std::string *LogPtr,
                    const std::vector<std::string> &RegisteredKernelNames) {
  std::vector<sycl::detail::string_view> Options;
  Options.reserve(CompileOptions.size());
  for (const std::string &opt : CompileOptions)
    Options.push_back(sycl::detail::string_view{opt});

  std::vector<sycl::detail::string_view> KernelNames;
  KernelNames.reserve(RegisteredKernelNames.size());
  for (const std::string &name : RegisteredKernelNames)
    KernelNames.push_back(sycl::detail::string_view{name});

  sycl::detail::string Log;
  auto result = compile_from_source(SourceKB, Devices, Options,
                                    LogPtr ? &Log : nullptr, KernelNames);
  if (LogPtr)
    *LogPtr = Log.c_str();
  return result;
}

} // namespace detail

/////////////////////////
// syclex::create_kernel_bundle_from_source
/////////////////////////
template <typename PropertyListT = empty_properties_t,
          typename = std::enable_if_t<detail::all_are_properties_of_v<
              detail::create_bundle_from_source_props, PropertyListT>>>
kernel_bundle<bundle_state::ext_oneapi_source> create_kernel_bundle_from_source(
    const context &SyclContext, source_language Language,
    const std::string &Source, PropertyListT props = {}) {
  std::vector<std::pair<std::string, std::string>> IncludePairsVec;
  if constexpr (props.template has_property<include_files>()) {
    IncludePairsVec = props.template get_property<include_files>().record;
  }

  return detail::make_kernel_bundle_from_source(SyclContext, Language, Source,
                                                IncludePairsVec);
}

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <typename PropertyListT = empty_properties_t,
          typename = std::enable_if_t<detail::all_are_properties_of_v<
              detail::create_bundle_from_source_props, PropertyListT>>>
kernel_bundle<bundle_state::ext_oneapi_source> create_kernel_bundle_from_source(
    const context &SyclContext, source_language Language,
    const std::vector<std::byte> &Bytes, PropertyListT props = {}) {
  std::vector<std::pair<std::string, std::string>> IncludePairsVec;
  if constexpr (props.template has_property<include_files>()) {
    IncludePairsVec = props.template get_property<include_files>().record;
  }

  return detail::make_kernel_bundle_from_source(SyclContext, Language, Bytes,
                                                IncludePairsVec);
}
#endif

/////////////////////////
// syclex::compile(source_kb) => obj_kb
/////////////////////////

template <typename PropertyListT = empty_properties_t,
          typename = std::enable_if_t<detail::all_are_properties_of_v<
              detail::build_source_bundle_props, PropertyListT>>>
kernel_bundle<bundle_state::object>
compile(kernel_bundle<bundle_state::ext_oneapi_source> &SourceKB,
        const std::vector<device> &Devices, PropertyListT props = {}) {
  std::vector<std::string> CompileOptionsVec;
  std::string *LogPtr = nullptr;
  std::vector<std::string> RegisteredKernelNamesVec;
  if constexpr (props.template has_property<build_options>())
    CompileOptionsVec = props.template get_property<build_options>().opts;
  if constexpr (props.template has_property<save_log>())
    LogPtr = props.template get_property<save_log>().log;
  if constexpr (props.template has_property<registered_names>())
    RegisteredKernelNamesVec =
        props.template get_property<registered_names>().names;
  return detail::compile_from_source(SourceKB, Devices, CompileOptionsVec,
                                     LogPtr, RegisteredKernelNamesVec);
}

template <typename PropertyListT = empty_properties_t,
          typename = std::enable_if_t<detail::all_are_properties_of_v<
              detail::build_source_bundle_props, PropertyListT>>>
kernel_bundle<bundle_state::object>
compile(kernel_bundle<bundle_state::ext_oneapi_source> &SourceKB,
        PropertyListT props = {}) {
  return compile<PropertyListT>(SourceKB, SourceKB.get_devices(), props);
}

/////////////////////////
// syclex::build(source_kb) => exe_kb
/////////////////////////

template <typename PropertyListT = empty_properties_t,
          typename = std::enable_if_t<detail::all_are_properties_of_v<
              detail::build_source_bundle_props, PropertyListT>>>

kernel_bundle<bundle_state::executable>
build(kernel_bundle<bundle_state::ext_oneapi_source> &SourceKB,
      const std::vector<device> &Devices, PropertyListT props = {}) {
  std::vector<std::string> BuildOptionsVec;
  std::string *LogPtr = nullptr;
  std::vector<std::string> RegisteredKernelNamesVec;
  if constexpr (props.template has_property<build_options>()) {
    BuildOptionsVec = props.template get_property<build_options>().opts;
  }
  if constexpr (props.template has_property<save_log>()) {
    LogPtr = props.template get_property<save_log>().log;
  }
  if constexpr (props.template has_property<registered_names>()) {
    RegisteredKernelNamesVec =
        props.template get_property<registered_names>().names;
  }
  return detail::build_from_source(SourceKB, Devices, BuildOptionsVec, LogPtr,
                                   RegisteredKernelNamesVec);
}

template <typename PropertyListT = empty_properties_t,
          typename = std::enable_if_t<detail::all_are_properties_of_v<
              detail::build_source_bundle_props, PropertyListT>>>
kernel_bundle<bundle_state::executable>
build(kernel_bundle<bundle_state::ext_oneapi_source> &SourceKB,
      PropertyListT props = {}) {
  return build<PropertyListT>(SourceKB, SourceKB.get_devices(), props);
}

} // namespace ext::oneapi::experimental

template <auto &SpecName>
void handler::set_specialization_constant(
    typename std::remove_reference_t<decltype(SpecName)>::value_type Value) {

  setStateSpecConstSet();

  getKernelBundle().set_specialization_constant<SpecName>(Value);
}

template <auto &SpecName>
typename std::remove_reference_t<decltype(SpecName)>::value_type
handler::get_specialization_constant() const {

  if (isStateExplicitKernelBundle())
    throw sycl::exception(make_error_code(errc::invalid),
                          "Specialization constants cannot be read after "
                          "explicitly setting the used kernel bundle");

  return getKernelBundle().get_specialization_constant<SpecName>();
}

} // namespace _V1
} // namespace sycl

namespace std {
template <> struct hash<sycl::kernel_id> {
  size_t operator()(const sycl::kernel_id &KernelID) const {
    return hash<std::shared_ptr<sycl::detail::kernel_id_impl>>()(
        sycl::detail::getSyclObjImpl(KernelID));
  }
};

template <sycl::bundle_state State> struct hash<sycl::device_image<State>> {
  size_t operator()(const sycl::device_image<State> &DeviceImage) const {
    return hash<std::shared_ptr<sycl::detail::device_image_impl>>()(
        sycl::detail::getSyclObjImpl(DeviceImage));
  }
};

template <sycl::bundle_state State> struct hash<sycl::kernel_bundle<State>> {
  size_t operator()(const sycl::kernel_bundle<State> &KernelBundle) const {
    return hash<std::shared_ptr<sycl::detail::kernel_bundle_impl>>()(
        sycl::detail::getSyclObjImpl(KernelBundle));
  }
};
} // namespace std
