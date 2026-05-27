//==------- ipc_physical_memory.cpp -- SYCL inter-process for physical mem -==//
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
#include <sycl/ext/oneapi/experimental/ipc_physical_memory.hpp>

namespace sycl {
inline namespace _V1 {

namespace detail {

__SYCL_EXPORT sycl::ext::oneapi::experimental::physical_mem
openIPCPhysicalMemHandle(const std::byte *HandleData, size_t HandleDataSize,
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

namespace ext::oneapi::experimental::ipc::physical_memory {

__SYCL_EXPORT handle get(physical_mem &physmem) {
  if (!physmem.ipc_enabled())
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "physical_mem was not created with inter-process sharing enabled "
        "via the enable_ipc property.");

  auto PhysMemImpl = sycl::detail::getSyclObjImpl(physmem);
  auto CtxImpl = sycl::detail::getSyclObjImpl(physmem.get_context());
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  void *HandlePtr = nullptr;
  size_t HandleSize = 0;
  auto UrRes = Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCGetPhysMemHandleExp>(
    CtxImpl->getHandleRef(), PhysMemImpl->getHandleRef(), &HandlePtr,
    &HandleSize);
  Adapter.checkUrResult(UrRes);

  return {HandlePtr, HandleSize};
}

__SYCL_EXPORT void put(handle &ipc_handle, const sycl::context &ctx) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  ur_result_t UrRes = Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCPutPhysMemHandleExp>(
      CtxImpl->getHandleRef(), ipc_handle.MData);
  Adapter.checkUrResult(UrRes);
}

} // namespace ext::oneapi::experimental::ipc::physical_memory
} // namespace _V1
} // namespace sycl
