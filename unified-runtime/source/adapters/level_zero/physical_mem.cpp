//===---------------- physical_mem.cpp - Level Zero Adapter ---------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "physical_mem.hpp"
#include "common.hpp"
#include "device.hpp"

#ifdef __linux__
#include <fcntl.h>
#endif

#ifdef UR_ADAPTER_LEVEL_ZERO_V2
#include "v2/context.hpp"
#else
#include "context.hpp"
#endif

namespace ur::level_zero {

ur_result_t urPhysicalMemCreate(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    [[maybe_unused]] const ur_physical_mem_properties_t *pProperties,
    ur_physical_mem_handle_t *phPhysicalMem) {
  ZeStruct<ze_physical_mem_desc_t> PhysicalMemDesc;
  PhysicalMemDesc.flags = 0;
  PhysicalMemDesc.size = size;

  bool EnableIpc =
      pProperties && (pProperties->flags & UR_PHYSICAL_MEM_FLAG_ENABLE_IPC);

  ze_physical_mem_handle_t ZePhysicalMem;
  ZE2UR_CALL(zePhysicalMemCreate, (hContext->getZeHandle(), hDevice->ZeDevice,
                                   &PhysicalMemDesc, &ZePhysicalMem));
  try {
    *phPhysicalMem = new ur_physical_mem_handle_t_(ZePhysicalMem, hContext,
                                                   hDevice, size, EnableIpc);
  } catch (const std::bad_alloc &) {
    zePhysicalMemDestroy(hContext->getZeHandle(), ZePhysicalMem);
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    zePhysicalMemDestroy(hContext->getZeHandle(), ZePhysicalMem);
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urPhysicalMemRetain(ur_physical_mem_handle_t hPhysicalMem) {
  hPhysicalMem->RefCount.retain();
  return UR_RESULT_SUCCESS;
}

ur_result_t urPhysicalMemRelease(ur_physical_mem_handle_t hPhysicalMem) {
  if (!hPhysicalMem->RefCount.release())
    return UR_RESULT_SUCCESS;

  if (hPhysicalMem->IpcVirtualAddress) {
    // IPC-opened handle: close the IPC virtual mapping instead of destroying
    // a physical mem handle (there is no ZePhysicalMem on the consumer side).
    if (checkL0LoaderTeardown()) {
      ZE2UR_CALL(zeMemCloseIpcHandle, (hPhysicalMem->Context->getZeHandle(),
                                       hPhysicalMem->IpcVirtualAddress));
    }
  } else if (hPhysicalMem->ZePhysicalMem) {
    if (checkL0LoaderTeardown()) {
      ZE2UR_CALL(zePhysicalMemDestroy, (hPhysicalMem->Context->getZeHandle(),
                                        hPhysicalMem->ZePhysicalMem));
    }
  }
  delete hPhysicalMem;

  return UR_RESULT_SUCCESS;
}

ur_result_t urPhysicalMemGetInfo(ur_physical_mem_handle_t hPhysicalMem,
                                 ur_physical_mem_info_t propName,
                                 size_t propSize, void *pPropValue,
                                 size_t *pPropSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_PHYSICAL_MEM_INFO_CONTEXT:
    return ReturnValue(hPhysicalMem->Context);
  case UR_PHYSICAL_MEM_INFO_DEVICE:
    return ReturnValue(hPhysicalMem->Device);
  case UR_PHYSICAL_MEM_INFO_SIZE:
    return ReturnValue(hPhysicalMem->Size);
  case UR_PHYSICAL_MEM_INFO_PROPERTIES: {
    ur_physical_mem_flags_t Flags = static_cast<ur_physical_mem_flags_t>(0);
    if (hPhysicalMem->EnableIpc)
      Flags = UR_PHYSICAL_MEM_FLAG_ENABLE_IPC;
    ur_physical_mem_properties_t Props = {
        UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES, nullptr, Flags};
    return ReturnValue(Props);
  }
  case UR_PHYSICAL_MEM_INFO_REFERENCE_COUNT:
    return ReturnValue(hPhysicalMem->RefCount.getCount());
  case UR_PHYSICAL_MEM_INFO_IPC_VIRTUAL_ADDRESS:
    return ReturnValue(hPhysicalMem->IpcVirtualAddress);
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urIPCGetPhysMemHandleExp(ur_context_handle_t hContext,
                                     ur_physical_mem_handle_t hPhysMem,
                                     void **ppIPCPhysMemHandleData,
                                     size_t *pIPCPhysMemHandleDataSizeRet) {
#ifdef __linux__
  if (!hPhysMem->EnableIpc)
    return UR_RESULT_ERROR_INVALID_ARGUMENT;

  // IPC-opened handles (consumer side) have ZePhysicalMem == nullptr; they
  // cannot be re-exported.
  if (!hPhysMem->ZePhysicalMem)
    return UR_RESULT_ERROR_INVALID_ARGUMENT;

  // Pass the physical memory handle directly to zeMemGetIpcHandleWithProperties
  // No prior virtual mapping is required or expected; the physical handle
  // itself is the key.
  ze_ipc_mem_handle_type_ext_desc_t HandleTypeDesc = {};
  HandleTypeDesc.stype = ZE_STRUCTURE_TYPE_IPC_MEM_HANDLE_TYPE_EXT_DESC;
  HandleTypeDesc.pNext = nullptr;
  HandleTypeDesc.typeFlags = ZE_IPC_MEM_HANDLE_TYPE_FLAG_DEFAULT;

  ze_ipc_mem_handle_t IpcHandle = {};
  ze_result_t ZeRes =
      ZE_CALL_NOCHECK(zeMemGetIpcHandleWithProperties,
                      (hContext->getZeHandle(),
                       reinterpret_cast<const void *>(hPhysMem->ZePhysicalMem),
                       &HandleTypeDesc, &IpcHandle));
  // On drivers that do not support physical-mem IPC the function may return
  // ZE_RESULT_ERROR_UNSUPPORTED_FEATURE or ZE_RESULT_ERROR_INVALID_ARGUMENT.
  // Map both to UR_RESULT_ERROR_UNSUPPORTED_FEATURE so callers can skip.
  if (ZeRes == ZE_RESULT_ERROR_INVALID_ARGUMENT ||
      ZeRes == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  if (ZeRes != ZE_RESULT_SUCCESS)
    return ze2urResult(ZeRes);

  // Some driver versions accept this call but return an fd-based handle even
  // when ZE_IPC_MEM_HANDLE_TYPE_FLAG_DEFAULT is requested, because they do not
  // yet support passing a ze_physical_mem_handle_t directly.  An fd-based
  // handle cannot be serialized to a plain byte buffer for cross-process
  // transfer without SCM_RIGHTS socket transfer.  Detect this case: if the
  // first bytes of the handle form a valid open file descriptor in this
  // process, the handle is fd-based.  Release it and return UNSUPPORTED_FEATURE
  // so callers can skip gracefully.
  {
    int FdVal = 0;
    static_assert(sizeof(IpcHandle.data) >= sizeof(FdVal));
    memcpy(&FdVal, IpcHandle.data, sizeof(FdVal));
    if (FdVal > 0 && ::fcntl(FdVal, F_GETFD) >= 0) {
      zeMemPutIpcHandle(hContext->getZeHandle(), IpcHandle);
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }
  }

  auto *HandleData = new (std::nothrow) ZeIPCPhysMemHandleData;
  if (!HandleData) {
    zeMemPutIpcHandle(hContext->getZeHandle(), IpcHandle);
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  HandleData->IpcHandle = IpcHandle;
  HandleData->Size = hPhysMem->Size;

  *ppIPCPhysMemHandleData = HandleData;
  *pIPCPhysMemHandleDataSizeRet = sizeof(ZeIPCPhysMemHandleData);
  return UR_RESULT_SUCCESS;
#else
  (void)hContext;
  (void)hPhysMem;
  (void)ppIPCPhysMemHandleData;
  (void)pIPCPhysMemHandleDataSizeRet;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
#endif // __linux__
}

ur_result_t urIPCPutPhysMemHandleExp(ur_context_handle_t hContext,
                                     const void *pIPCPhysMemHandleData) {
#ifdef __linux__
  auto *HandleData =
      static_cast<const ZeIPCPhysMemHandleData *>(pIPCPhysMemHandleData);
  ze_result_t ZeResult = ZE_CALL_NOCHECK(
      zeMemPutIpcHandle, (hContext->getZeHandle(), HandleData->IpcHandle));
  delete HandleData;
  return ze2urResult(ZeResult);
#else
  (void)hContext;
  (void)pIPCPhysMemHandleData;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
#endif // __linux__
}

ur_result_t urIPCOpenPhysMemHandleExp(ur_context_handle_t hContext,
                                      ur_device_handle_t hDevice,
                                      const void *pIPCPhysMemHandleData,
                                      size_t ipcPhysMemHandleDataSize,
                                      ur_physical_mem_handle_t *phPhysMem) {
#ifdef __linux__
  if (ipcPhysMemHandleDataSize != sizeof(ZeIPCPhysMemHandleData))
    return UR_RESULT_ERROR_INVALID_VALUE;

  auto *HandleData =
      static_cast<const ZeIPCPhysMemHandleData *>(pIPCPhysMemHandleData);

  if (HandleData->Size == 0)
    return UR_RESULT_ERROR_INVALID_VALUE;

  // Open the IPC handle in this process.  zeMemOpenIpcHandle creates a virtual
  // mapping backed by the exporter's physical memory and returns a pointer to
  // it.  No separate zeVirtualMemMap call is needed; the memory is immediately
  // accessible at the returned address.
  void *VirtualAddress = nullptr;
  ZE2UR_CALL(zeMemOpenIpcHandle,
             (hContext->getZeHandle(), hDevice->ZeDevice, HandleData->IpcHandle,
              ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED, &VirtualAddress));
  if (!VirtualAddress)
    return UR_RESULT_ERROR_UNKNOWN;

  try {
    *phPhysMem = new ur_physical_mem_handle_t_(
        /*ZePhysicalMem=*/nullptr, hContext, hDevice, HandleData->Size,
        /*EnableIpc=*/true, VirtualAddress);
  } catch (const std::bad_alloc &) {
    zeMemCloseIpcHandle(hContext->getZeHandle(), VirtualAddress);
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    zeMemCloseIpcHandle(hContext->getZeHandle(), VirtualAddress);
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
#else
  (void)hContext;
  (void)hDevice;
  (void)pIPCPhysMemHandleData;
  (void)ipcPhysMemHandleDataSize;
  (void)phPhysMem;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
#endif // __linux__
}

ur_result_t urIPCClosePhysMemHandleExp(ur_context_handle_t hContext,
                                       ur_physical_mem_handle_t hPhysMem) {
  (void)hContext;
  // Delegate to urPhysicalMemRelease so the refcount is respected.  For
  // IPC-opened handles (IpcVirtualAddress != nullptr) urPhysicalMemRelease
  // calls zeMemCloseIpcHandle; for regular handles it calls
  // zePhysicalMemDestroy.
  return ur::level_zero::urPhysicalMemRelease(hPhysMem);
}

} // namespace ur::level_zero
