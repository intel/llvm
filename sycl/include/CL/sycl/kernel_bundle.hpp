//==------- kernel_bundle.hpp - SYCL kernel_bundle and free functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/kernel_bundle_enums.hpp>

#include <cassert>
#include <memory>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
// Forward declaration
template <backend Backend> class backend_traits;

namespace detail {
class kernel_id_impl;
}

/// Objects of the class identify kernel is some kernel_bundle related APIs
///
/// \ingroup sycl_api
class __SYCL_EXPORT kernel_id {
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
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};

namespace detail {
class device_image_impl;
using DeviceImageImplPtr = std::shared_ptr<device_image_impl>;

// The class is used as a base for device_image for "untemplating" public
// methods.
class __SYCL_EXPORT device_image_plain {
public:
  device_image_plain(const detail::DeviceImageImplPtr &Impl)
      : impl(std::move(Impl)) {}

  bool operator==(const device_image_plain &RHS) const {
    return impl == RHS.impl;
  }

  bool operator!=(const device_image_plain &RHS) const {
    return !(*this == RHS);
  }

  bool has_kernel(const kernel_id &KernelID) const noexcept;

  bool has_kernel(const kernel_id &KernelID, const device &Dev) const noexcept;

  pi_native_handle getNative() const;

protected:
  detail::DeviceImageImplPtr impl;

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};
} // namespace detail

/// Objects of the class represents an instance of an image in a specific state.
template <sycl::bundle_state State>
class device_image : public detail::device_image_plain {
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

private:
  device_image(detail::DeviceImageImplPtr Impl)
      : device_image_plain(std::move(Impl)) {}

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
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

  void get_specialization_constant_impl(const char *SpecName, void *Value) const
      noexcept;

  bool is_specialization_constant_set(const char *SpecName) const noexcept;

  detail::KernelBundleImplPtr impl;
};

} // namespace detail

/// The kernel_bundle class represents collection of device images in a
/// particular state
///
/// \ingroup sycl_api
template <bundle_state State>
class kernel_bundle : public detail::kernel_bundle_plain {
public:
  using device_image_iterator = const device_image<State> *;

  kernel_bundle() = delete;

  /// \returns true if the kernel_bundles contains no device images
  bool empty() const noexcept { return kernel_bundle_plain::empty(); }

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
  bool has_kernel(const kernel_id &KernelID) const noexcept {
    return kernel_bundle_plain::has_kernel(KernelID);
  }

  /// \returns true if the kernel_bundle contains the kernel identified by
  /// kernel_id passed and if this kernel is compatible with the device
  /// specified
  bool has_kernel(const kernel_id &KernelID, const device &Dev) const noexcept {
    return kernel_bundle_plain::has_kernel(KernelID, Dev);
  }

  /// \returns a vector of kernel_id's that contained in the kernel_bundle
  std::vector<kernel_id> get_kernel_ids() const {
    return kernel_bundle_plain::get_kernel_ids();
  }

  /// \returns true if the kernel_bundle contains at least one device image
  /// which uses specialization constants
  bool contains_specialization_constants() const noexcept {
    return kernel_bundle_plain::contains_specialization_constants();
  }

  /// \returns true if all specialization constants which are used in the
  /// kernel_bundle are "native specialization constants in all device images
  bool native_specialization_constant() const noexcept {
    return kernel_bundle_plain::native_specialization_constant();
  }

  /// \returns a kernel object which represents the kernel identified by
  /// kernel_id passed
  template <bundle_state _State = State,
            typename = detail::enable_if_t<_State == bundle_state::executable>>
  kernel get_kernel(const kernel_id &KernelID) const {
    return detail::kernel_bundle_plain::get_kernel(KernelID);
  }

  // This guard is needed because the libsycl.so can compiled with C++ <=14
  // while the code requires C++17. This code is not supposed to be used by the
  // libsycl.so so it should not be a problem.
#if __cplusplus > 201402L
  /// \returns true if any device image in the kernel_bundle uses specialization
  /// constant whose address is SpecName
  template <auto &SpecName> bool has_specialization_constant() const noexcept {
    const char *SpecSymName = detail::get_spec_constant_symbolic_ID<SpecName>();
    return has_specialization_constant_impl(SpecSymName);
  }

  /// Sets the value of the specialization constant whose address is SpecName
  /// for this bundle. If the specialization constant’s value was previously set
  /// in this bundle, the value is overwritten.
  template <auto &SpecName, bundle_state _State = State,
            typename = detail::enable_if_t<_State == bundle_state::input>>
  void set_specialization_constant(
      typename std::remove_reference_t<decltype(SpecName)>::value_type Value) {
    const char *SpecSymName = detail::get_spec_constant_symbolic_ID<SpecName>();
    set_specialization_constant_impl(SpecSymName, &Value,
                                     sizeof(decltype(Value)));
  }

  /// \returns the value of the specialization constant whose address is
  /// SpecName for this kernel bundle.
  template <auto &SpecName>
  typename std::remove_reference_t<decltype(SpecName)>::value_type
  get_specialization_constant() const {
    const char *SpecSymName = detail::get_spec_constant_symbolic_ID<SpecName>();
    if (!is_specialization_constant_set(SpecSymName))
      return SpecName.getDefaultValue();

    using SCType =
        typename std::remove_reference_t<decltype(SpecName)>::value_type;

    std::array<char *, sizeof(SCType)> RetValue;

    get_specialization_constant_impl(SpecSymName, RetValue.data());

    return *reinterpret_cast<SCType *>(RetValue.data());
  }
#endif

  /// \returns an iterator to the first device image kernel_bundle contains
  device_image_iterator begin() const {
    return reinterpret_cast<device_image_iterator>(
        kernel_bundle_plain::begin());
  }

  /// \returns an iterator to the last device image kernel_bundle contains
  device_image_iterator end() const {
    return reinterpret_cast<device_image_iterator>(kernel_bundle_plain::end());
  }

  template <backend Backend>
  std::vector<typename backend_traits<Backend>::template return_type<
      kernel_bundle<State>>>
  get_native() {
    std::vector<typename backend_traits<Backend>::template return_type<
        kernel_bundle<State>>>
        ReturnValue;
    ReturnValue.reserve(std::distance(begin(), end()));

    for (const device_image<State> &DevImg : *this) {
      ReturnValue.push_back(
          detail::pi::cast<typename backend_traits<
              Backend>::template return_type<kernel_bundle<State>>>(
              DevImg.getNative()));
    }

    return ReturnValue;
  }

private:
  kernel_bundle(detail::KernelBundleImplPtr Impl)
      : kernel_bundle_plain(std::move(Impl)) {}

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};

/////////////////////////
// get_kernel_id API
/////////////////////////

namespace detail {
// Internal non-template versions of get_kernel_id API which is used by public
// onces
__SYCL_EXPORT kernel_id get_kernel_id_impl(std::string KernelName);
} // namespace detail

/// \returns the kernel_id associated with the KernelName
template <typename KernelName> kernel_id get_kernel_id() {
  using KI = sycl::detail::KernelInfo<KernelName>;
  return detail::get_kernel_id_impl(KI::getName());
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
} // namespace detail

/// A kernel bundle in state State which contains all of the kernels in the
/// application which are compatible with at least one of the devices in Devs.
/// This does not include any device built-in kernels. The bundle’s set of
/// associated devices is Devs.
template <bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &Ctx,
                                       const std::vector<device> &Devs) {
  detail::KernelBundleImplPtr Impl =
      detail::get_kernel_bundle_impl(Ctx, Devs, State);

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
  detail::KernelBundleImplPtr Impl =
      detail::get_kernel_bundle_impl(Ctx, Devs, KernelIDs, State);
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

namespace detail {

// Stable selector function type for passing thru library boundaries
using DevImgSelectorImpl =
    std::function<bool(const detail::DeviceImageImplPtr &DevImgImpl)>;

// Internal non-template versions of get_kernel_bundle API which is used by
// public onces
__SYCL_EXPORT detail::KernelBundleImplPtr
get_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                       bundle_state State, const DevImgSelectorImpl &Selector);
} // namespace detail

/// A kernel bundle in state State which contains all of the device images for
/// which the selector returns true.
template <bundle_state State, typename SelectorT>
kernel_bundle<State> get_kernel_bundle(const context &Ctx,
                                       const std::vector<device> &Devs,
                                       SelectorT Selector) {
  detail::DevImgSelectorImpl SelectorWrapper =
      [Selector](const detail::DeviceImageImplPtr &DevImg) {
        return Selector(
            detail::createSyclObjFromImpl<sycl::device_image<State>>(DevImg));
      };

  detail::KernelBundleImplPtr Impl =
      detail::get_kernel_bundle_impl(Ctx, Devs, State, SelectorWrapper);

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

/////////////////////////
// is_compatible API
/////////////////////////

/// \returns true if all of the kernels identified by KernelIDs are compatible
/// with the device Dev.
bool is_compatible(const std::vector<kernel_id> &KernelIDs, const device &Dev);

template <typename KernelName> bool is_compatible(const device &Dev) {
  return is_compatible({get_kernel_id<KernelName>()}, Dev);
}

/////////////////////////
// join API
/////////////////////////

namespace detail {

__SYCL_EXPORT std::shared_ptr<detail::kernel_bundle_impl>
join_impl(const std::vector<detail::KernelBundleImplPtr> &Bundles);
}

/// \returns a new kernel bundle that represents the union of all the device
/// images in the input bundles with duplicates removed.
template <sycl::bundle_state State>
sycl::kernel_bundle<State>
join(const std::vector<sycl::kernel_bundle<State>> &Bundles) {
  // Convert kernel_bundle<State> to impls to abstract template parameter away
  std::vector<detail::KernelBundleImplPtr> KernelBundleImpls;
  KernelBundleImpls.reserve(Bundles.size());
  for (const sycl::kernel_bundle<State> &Bundle : Bundles)
    KernelBundleImpls.push_back(detail::getSyclObjImpl(Bundle));

  std::shared_ptr<detail::kernel_bundle_impl> Impl =
      detail::join_impl(KernelBundleImpls);
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
  detail::KernelBundleImplPtr Impl =
      detail::compile_impl(InputBundle, Devs, PropList);
  return detail::createSyclObjFromImpl<
      kernel_bundle<sycl::bundle_state::object>>(Impl);
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
}

/// \returns a new kernel_bundle which contains the device images from the
/// ObjectBundles that are translated into one or more new device images of
/// state bundle_state::executable The new bundle represents all of the kernels
/// in ObjectBundles that are compatible with at least one of the devices in
/// Devs.
inline kernel_bundle<bundle_state::executable>
link(const std::vector<kernel_bundle<bundle_state::object>> &ObjectBundles,
     const std::vector<device> &Devs, const property_list &PropList = {}) {
  detail::KernelBundleImplPtr Impl =
      detail::link_impl(ObjectBundles, Devs, PropList);
  return detail::createSyclObjFromImpl<
      kernel_bundle<sycl::bundle_state::executable>>(Impl);
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
  detail::KernelBundleImplPtr Impl =
      detail::build_impl(InputBundle, Devs, PropList);
  return detail::createSyclObjFromImpl<
      kernel_bundle<sycl::bundle_state::executable>>(Impl);
}

inline kernel_bundle<bundle_state::executable>
build(const kernel_bundle<bundle_state::input> &InputBundle,
      const property_list &PropList = {}) {
  return build(InputBundle, InputBundle.get_devices(), PropList);
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct hash<cl::sycl::kernel_id> {
  size_t operator()(const cl::sycl::kernel_id &KernelID) const {
    return hash<std::shared_ptr<cl::sycl::detail::kernel_id_impl>>()(
        cl::sycl::detail::getSyclObjImpl(KernelID));
  }
};

template <cl::sycl::bundle_state State>
struct hash<cl::sycl::device_image<State>> {
  size_t operator()(const cl::sycl::device_image<State> &DeviceImage) const {
    return hash<std::shared_ptr<cl::sycl::detail::device_image_impl>>()(
        cl::sycl::detail::getSyclObjImpl(DeviceImage));
  }
};

template <cl::sycl::bundle_state State>
struct hash<cl::sycl::kernel_bundle<State>> {
  size_t operator()(const cl::sycl::kernel_bundle<State> &KernelBundle) const {
    return hash<std::shared_ptr<cl::sycl::detail::kernel_bundle_impl>>()(
        cl::sycl::detail::getSyclObjImpl(KernelBundle));
  }
};
} // namespace std
