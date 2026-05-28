//==------- ipc_memory.cpp --- SYCL inter-process communicable memory ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/physical_mem_impl.hpp>
#include <sycl/context.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/usm/usm_pointer_info.hpp>

#include <utility>

namespace sycl {
inline namespace _V1 {

namespace detail {

__SYCL_EXPORT void *openIPCMemHandle(const std::byte *HandleData,
                                     size_t HandleDataSize,
                                     const sycl::context &Ctx,
                                     const sycl::device &Dev) {
  if (!Dev.has(aspect::ext_oneapi_ipc_memory))
    throw sycl::exception(
        sycl::make_error_code(errc::feature_not_supported),
        "Device does not support aspect::ext_oneapi_ipc_memory.");

  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  // TODO: UMF and UR currently requires the handle data to be non-const, so we
  //       need const-cast the data pointer. Once this has been changed, the
  //       const-cast can be removed.
  //       CMPLRLLVM-71181
  //       https://github.com/oneapi-src/unified-memory-framework/issues/1536
  std::byte *NonConstHandleData = const_cast<std::byte *>(HandleData);

  void *Ptr = nullptr;
  ur_result_t UrRes =
      Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCOpenMemHandleExp>(
          CtxImpl->getHandleRef(), getSyclObjImpl(Dev)->getHandleRef(),
          NonConstHandleData, HandleDataSize, &Ptr);
  if (UrRes == UR_RESULT_ERROR_INVALID_VALUE)
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "HandleData data size does not correspond "
                          "to the target platform's IPC memory handle size.");
  Adapter.checkUrResult(UrRes);
  if (Ptr == nullptr)
    throw sycl::exception(
        sycl::make_error_code(errc::runtime),
        "urIPCOpenMemHandleExp returned success but did not produce a "
        "valid memory pointer.");

  return Ptr;
}

__SYCL_EXPORT ext::oneapi::experimental::physical_mem
openIPCPhysMemHandle(const std::byte *HandleData, size_t HandleDataSize,
                     const sycl::context &Ctx, const sycl::device &Dev) {
  if (!Dev.has(aspect::ext_oneapi_ipc_physical_memory))
    throw sycl::exception(
        sycl::make_error_code(errc::feature_not_supported),
        "Device does not support aspect::ext_oneapi_ipc_physical_memory.");

  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  ur_physical_mem_handle_t PhysMemHandle = nullptr;
  ur_result_t UrRes =
      Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCOpenPhysMemHandleExp>(
          CtxImpl->getHandleRef(), getSyclObjImpl(Dev)->getHandleRef(),
          HandleData, HandleDataSize, &PhysMemHandle);
  if (UrRes == UR_RESULT_ERROR_INVALID_VALUE)
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "HandleData data size does not correspond to the target platform's "
        "IPC physical memory handle size.");
  Adapter.checkUrResult(UrRes);
  if (PhysMemHandle == nullptr)
    throw sycl::exception(
        sycl::make_error_code(errc::runtime),
        "urIPCOpenPhysMemHandleExp returned success but did not produce a "
        "valid physical memory handle.");

  // Any failure after this point must release PhysMemHandle to avoid leaking
  // the GPU resource.  Wrap in try-catch so both urPhysicalMemGetInfo failures
  // and std::bad_alloc from make_shared are handled.
  try {
    // Query the actual allocation size from the opened handle so that
    // physical_mem::size() returns the correct value.
    size_t NumBytes = 0;
    Adapter.call<sycl::detail::UrApiKind::urPhysicalMemGetInfo>(
        PhysMemHandle, UR_PHYSICAL_MEM_INFO_SIZE, sizeof(size_t), &NumBytes,
        nullptr);

    auto PhysMemImpl = std::make_shared<sycl::detail::physical_mem_impl>(
        *getSyclObjImpl(Dev), Ctx, NumBytes, PhysMemHandle);
    return sycl::detail::createSyclObjFromImpl<
        ext::oneapi::experimental::physical_mem>(PhysMemImpl);
  } catch (...) {
    Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCClosePhysMemHandleExp>(
        CtxImpl->getHandleRef(), PhysMemHandle);
    throw;
  }
}

} // namespace detail

namespace ext::oneapi::experimental::ipc::memory {
namespace detail {

std::pair<void *, size_t> get(void *Ptr, const sycl::context &Ctx) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  // If the API fails, check that the device actually supported it. We only do
  // this if UR fails to avoid the device-lookup overhead.
  auto CheckDeviceSupport = [Ptr, &Ctx]() {
    sycl::device Dev = get_pointer_device(Ptr, Ctx);
    if (!Dev.has(aspect::ext_oneapi_ipc_memory))
      throw sycl::exception(
          sycl::make_error_code(errc::feature_not_supported),
          "Device does not support aspect::ext_oneapi_ipc_memory.");
  };

  void *HandlePtr = nullptr;
  size_t HandleSize = 0;
  auto UrRes =
      Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCGetMemHandleExp>(
          CtxImpl->getHandleRef(), Ptr, &HandlePtr, &HandleSize);
  if (UrRes != UR_RESULT_SUCCESS) {
    CheckDeviceSupport();
    Adapter.checkUrResult(UrRes);
  }
  if (HandlePtr == nullptr)
    throw sycl::exception(
        sycl::make_error_code(errc::runtime),
        "urIPCGetMemHandleExp returned success but did not produce a "
        "valid IPC handle.");
  return {HandlePtr, HandleSize};
}

void put(std::byte *HandleData, const sycl::context &Ctx) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  CtxImpl->getAdapter().call<sycl::detail::UrApiKind::urIPCPutMemHandleExp>(
      CtxImpl->getHandleRef(), HandleData);
}

void close(void *Ptr, const sycl::context &Ctx) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  CtxImpl->getAdapter().call<sycl::detail::UrApiKind::urIPCCloseMemHandleExp>(
      CtxImpl->getHandleRef(), Ptr);
}

} // namespace detail

__SYCL_EXPORT handle get(void *Ptr, const sycl::context &Ctx) {
  std::pair<void *, size_t> RetHandle = detail::get(Ptr, Ctx);
  return {RetHandle.first, RetHandle.second};
}

__SYCL_EXPORT void put(handle &Handle, const sycl::context &Ctx) {
  detail::put(Handle.MData, Ctx);
}

__SYCL_EXPORT void close(void *Ptr, const sycl::context &Ctx) {
  detail::close(Ptr, Ctx);
}
} // namespace ext::oneapi::experimental::ipc::memory

namespace ext::oneapi::experimental::ipc::physical_memory {
namespace detail {

std::pair<void *, size_t>
get(const ext::oneapi::experimental::physical_mem &PhysMem) {
  if (!PhysMem.ext_oneapi_ipc_enabled())
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "physical_mem was not created with inter-process sharing enabled "
        "via the enable_ipc property.");

  auto CheckDeviceSupport = [&PhysMem]() {
    sycl::device Dev = PhysMem.get_device();
    if (!Dev.has(aspect::ext_oneapi_ipc_physical_memory))
      throw sycl::exception(
          sycl::make_error_code(errc::feature_not_supported),
          "Device does not support aspect::ext_oneapi_ipc_physical_memory.");
  };

  auto PhysMemImpl = sycl::detail::getSyclObjImpl(PhysMem);
  auto CtxImpl = sycl::detail::getSyclObjImpl(PhysMem.get_context());
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  void *HandlePtr = nullptr;
  size_t HandleSize = 0;
  auto UrRes =
      Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCGetPhysMemHandleExp>(
          CtxImpl->getHandleRef(), PhysMemImpl->getHandleRef(), &HandlePtr,
          &HandleSize);
  if (UrRes != UR_RESULT_SUCCESS) {
    CheckDeviceSupport();
    Adapter.checkUrResult(UrRes);
  }
  if (HandlePtr == nullptr)
    throw sycl::exception(
        sycl::make_error_code(errc::runtime),
        "urIPCGetPhysMemHandleExp returned success but did not produce a "
        "valid IPC handle.");
  return {HandlePtr, HandleSize};
}

void put(std::byte *HandleData, const sycl::context &Ctx) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  CtxImpl->getAdapter().call<sycl::detail::UrApiKind::urIPCPutPhysMemHandleExp>(
      CtxImpl->getHandleRef(), HandleData);
}

ext::oneapi::experimental::physical_mem open(const std::byte *HandleData,
                                             size_t HandleDataSize,
                                             const sycl::context &Ctx,
                                             const sycl::device &Dev) {
  return sycl::detail::openIPCPhysMemHandle(HandleData, HandleDataSize, Ctx,
                                            Dev);
}

} // namespace detail

__SYCL_EXPORT ipc::handle
get(const ext::oneapi::experimental::physical_mem &PhysMem) {
  std::pair<void *, size_t> RetHandle = detail::get(PhysMem);
  return {RetHandle.first, RetHandle.second};
}

__SYCL_EXPORT void put(ipc::handle &Handle, const sycl::context &Ctx) {
  detail::put(Handle.MData, Ctx);
}

__SYCL_EXPORT ext::oneapi::experimental::physical_mem
open(const ipc::handle_data_t &HandleData, const sycl::context &Ctx,
     const sycl::device &Dev) {
  return detail::open(HandleData.data(), HandleData.size(), Ctx, Dev);
}
} // namespace ext::oneapi::experimental::ipc::physical_memory

namespace ext::oneapi::experimental::ipc_memory {
__SYCL_SUPPRESS_DEPRECATED_PUSH
__SYCL_EXPORT handle get(void *Ptr, const sycl::context &Ctx) {
  std::pair<void *, size_t> RetHandle =
      ext::oneapi::experimental::ipc::memory::detail::get(Ptr, Ctx);
  return {RetHandle.first, RetHandle.second};
}

__SYCL_EXPORT void put(handle &Handle, const sycl::context &Ctx) {
  ext::oneapi::experimental::ipc::memory::detail::put(Handle.MData, Ctx);
}

__SYCL_EXPORT void close(void *Ptr, const sycl::context &Ctx) {
  ext::oneapi::experimental::ipc::memory::detail::close(Ptr, Ctx);
}
__SYCL_SUPPRESS_DEPRECATED_POP
} // namespace ext::oneapi::experimental::ipc_memory
} // namespace _V1
} // namespace sycl
