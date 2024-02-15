//==----------- platform.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/backend_impl.hpp>
#include <detail/config.hpp>
#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>
#include <sycl/device.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/platform.hpp>

namespace sycl {
inline namespace _V1 {

platform::platform() : platform(default_selector_v) {}

platform::platform(cl_platform_id PlatformId) {
  impl = detail::platform_impl::getOrMakePlatformImpl(
      detail::pi::cast<sycl::detail::pi::PiPlatform>(PlatformId),
      sycl::detail::pi::getPlugin<backend::opencl>());
}

// protected constructor for internal use
platform::platform(const device &Device) { *this = Device.get_platform(); }

platform::platform(const device_selector &dev_selector) {
  *this = dev_selector.select_device().get_platform();
}

cl_platform_id platform::get() const { return impl->get(); }

bool platform::has_extension(const std::string &ExtensionName) const {
  return impl->has_extension(ExtensionName);
}

bool platform::is_host() const {
  bool IsHost = impl->is_host();
  assert(!IsHost &&
         "platform::is_host should not be called in implementation.");
  return IsHost;
}

std::vector<device> platform::get_devices(info::device_type DeviceType) const {
  return impl->get_devices(DeviceType);
}

std::vector<platform> platform::get_platforms() {
  return detail::platform_impl::get_platforms();
}

backend platform::get_backend() const noexcept { return impl->getBackend(); }

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
template <typename Param>
typename detail::is_platform_info_desc<Param>::return_type
platform::get_info_impl() const {
  return impl->get_info<Param>();
}

template <typename Param>
typename ReturnType<
    typename detail::is_platform_info_desc<Param>::return_type>::type
platform::get_info_internal() const {
  if constexpr (std::is_same_v<std::string,
                               typename detail::is_platform_info_desc<
                                   Param>::return_type>) {
    return get_platform_info<Param>();
  } else if constexpr (std::is_same_v<std::vector<std::string>,
                                      typename detail::is_platform_info_desc<
                                          Param>::return_type>) {
    return get_platform_info_vector<Param>();
  } else {
    return get_info_impl<Param>();
  }
}

template <typename Param> detail::string platform::get_platform_info() const {
  return detail::string(impl->template get_info<Param>());
}

template <typename Param>
std::vector<detail::string> platform::get_platform_info_vector() const {
  std::vector<std::string> Info = impl->template get_info<Param>();
  std::vector<detail::string> Result;
  for (std::string &Str : Info) {
    Result.push_back(detail::string(Str.c_str()));
  }
  return Result;
}

// Instantiation of get_platform_info and get_platform_info_vector
#define __SYCL_GET_PLATFORM_INFO_SPEC(Desc)                                    \
  template <>                                                                  \
  detail::string platform::get_platform_info<info::platform::Desc>() const {   \
    return detail::string(impl->template get_info<info::platform::Desc>());    \
  }

__SYCL_GET_PLATFORM_INFO_SPEC(name)
__SYCL_GET_PLATFORM_INFO_SPEC(profile)
__SYCL_GET_PLATFORM_INFO_SPEC(vendor)
__SYCL_GET_PLATFORM_INFO_SPEC(version)

#define __SYCL_GET_PLATFORM_INFO_VECTOR_SPEC(Desc)                             \
  template <>                                                                  \
  std::vector<detail::string>                                                  \
  platform::get_platform_info_vector<info::platform::Desc>() const {           \
    std::vector<std::string> Info =                                            \
        impl->template get_info<info::platform::Desc>();                       \
    std::vector<detail::string> Result;                                        \
    for (std::string & Str : Info) {                                           \
      Result.push_back(detail::string(Str.c_str()));                           \
    }                                                                          \
    return Result;                                                             \
  }

__SYCL_GET_PLATFORM_INFO_VECTOR_SPEC(extensions)
#else
template <typename Param>
typename detail::is_platform_info_desc<Param>::return_type
platform::get_info() const {
  return impl->get_info<Param>();
}
#endif

pi_native_handle platform::getNative() const { return impl->getNative(); }

bool platform::has(aspect Aspect) const { return impl->has(Aspect); }

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template __SYCL_EXPORT ReturnT platform::get_info<info::platform::Desc>()    \
      const;

#include <sycl/info/platform_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

context platform::ext_oneapi_get_default_context() const {
  if (!detail::SYCLConfig<detail::SYCL_ENABLE_DEFAULT_CONTEXTS>::get())
    throw std::runtime_error("SYCL default contexts are not enabled");

  // Keeping the default context for platforms in the global cache to avoid
  // shared_ptr based circular dependency between platform and context classes
  std::unordered_map<detail::PlatformImplPtr, detail::ContextImplPtr>
      &PlatformToDefaultContextCache =
          detail::GlobalHandler::instance().getPlatformToDefaultContextCache();

  std::lock_guard<std::mutex> Lock{
      detail::GlobalHandler::instance()
          .getPlatformToDefaultContextCacheMutex()};

  auto It = PlatformToDefaultContextCache.find(impl);
  if (PlatformToDefaultContextCache.end() == It)
    std::tie(It, std::ignore) = PlatformToDefaultContextCache.insert(
        {impl, detail::getSyclObjImpl(context{get_devices()})});

  return detail::createSyclObjFromImpl<context>(It->second);
}

namespace detail {

void enable_ext_oneapi_default_context(bool Val) {
  const char *StringVal = Val ? "1" : "0";
  detail::SYCLConfig<detail::SYCL_ENABLE_DEFAULT_CONTEXTS>::resetWithValue(
      StringVal);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
