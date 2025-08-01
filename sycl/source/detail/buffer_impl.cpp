//==----------------- buffer_impl.cpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/buffer_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/memory_manager.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/xpti_registry.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/properties/buffer_properties.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
uint8_t GBufferStreamID;
#endif
void *buffer_impl::allocateMem(context_impl *Context, bool InitFromUserData,
                               void *HostPtr,
                               ur_event_handle_t &OutEventToWait) {
  bool HostPtrReadOnly = false;
  BaseT::determineHostPtr(Context, InitFromUserData, HostPtr, HostPtrReadOnly);
  assert(!(nullptr == HostPtr && BaseT::useHostPtr() && !Context) &&
         "Internal error. Allocating memory on the host "
         "while having use_host_ptr property");
  return MemoryManager::allocateMemBuffer(
      Context, this, HostPtr, HostPtrReadOnly, BaseT::getSizeInBytes(),
      BaseT::MInteropEvent, BaseT::MInteropContext.get(), MProps,
      OutEventToWait);
}
void buffer_impl::constructorNotification(const detail::code_location &CodeLoc,
                                          void *UserObj, const void *HostObj,
                                          const void *Type, uint32_t Dim,
                                          uint32_t ElemSize, size_t Range[3]) {
  XPTIRegistry::bufferConstructorNotification(UserObj, CodeLoc, HostObj, Type,
                                              Dim, ElemSize, Range);
}

void buffer_impl::destructorNotification(void *UserObj) {
  XPTIRegistry::bufferDestructorNotification(UserObj);
}

void buffer_impl::addInteropObject(
    std::vector<ur_native_handle_t> &Handles) const {
  if (MOpenCLInterop) {
    if (std::find(Handles.begin(), Handles.end(),
                  ur::cast<ur_native_handle_t>(MInteropMemObject)) ==
        Handles.end()) {
      adapter_impl &Adapter = getAdapter();
      Adapter.call<UrApiKind::urMemRetain>(
          ur::cast<ur_mem_handle_t>(MInteropMemObject));
      ur_native_handle_t NativeHandle = 0;
      Adapter.call<UrApiKind::urMemGetNativeHandle>(MInteropMemObject, nullptr,
                                                    &NativeHandle);
      Handles.push_back(NativeHandle);
    }
  }
}

std::vector<ur_native_handle_t>
buffer_impl::getNativeVector(backend BackendName) const {
  std::vector<ur_native_handle_t> Handles{};
  if (!MRecord) {
    addInteropObject(Handles);
    return Handles;
  }

  for (auto &Cmd : MRecord->MAllocaCommands) {
    ur_mem_handle_t NativeMem =
        ur::cast<ur_mem_handle_t>(Cmd->getMemAllocation());
    auto Ctx = Cmd->getWorkerContext();
    // If Host Shared Memory is not supported then there is alloca for host that
    // doesn't have context and platform
    if (!Ctx)
      continue;
    const platform_impl &Platform = Ctx->getPlatformImpl();
    if (Platform.getBackend() != BackendName)
      continue;

    adapter_impl &Adapter = Platform.getAdapter();
    ur_native_handle_t Handle = 0;
    // When doing buffer interop we don't know what device the memory should be
    // resident on, so pass nullptr for Device param. Buffer interop may not be
    // supported by all backends.
    Adapter.call<UrApiKind::urMemGetNativeHandle>(NativeMem, /*Dev*/ nullptr,
                                                  &Handle);
    Handles.push_back(Handle);

    if (Platform.getBackend() == backend::opencl) {
      __SYCL_OCL_CALL(clRetainMemObject, ur::cast<cl_mem>(Handle));
    }
  }

  addInteropObject(Handles);
  return Handles;
}

void buffer_impl::verifyProps(const property_list &Props) const {
  auto CheckDataLessProperties = [](int PropertyKind) {
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  case NS_QUALIFIER::PROP_NAME::getKind():                                     \
    return true;
#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)
    switch (PropertyKind) {
#include <sycl/properties/buffer_properties.def>
    default:
      return false;
    }
  };
  auto CheckPropertiesWithData = [](int PropertyKind) {
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)
#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)                  \
  case NS_QUALIFIER::PROP_NAME::getKind():                                     \
    return true;
    switch (PropertyKind) {
#include <sycl/properties/buffer_properties.def>
    default:
      return false;
    }
  };
  detail::PropertyValidator::checkPropsAndThrow(Props, CheckDataLessProperties,
                                                CheckPropertiesWithData);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
