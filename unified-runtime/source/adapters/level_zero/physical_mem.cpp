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
#include <cerrno>
#include <sys/syscall.h>
#include <unistd.h>
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

#ifdef __linux__
  // For pre-Xe2 devices (DG2, PVC, MTL, ARL), chain the export descriptor so
  // the driver allocates the physical memory in DMA-BUF exportable form.
  // For BMG and newer (Xe2+) the opaque IPC handle path requires no extension
  // at creation time.
  ze_external_memory_export_desc_t ExportDesc = {};
  ExportDesc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC;
  ExportDesc.pNext = nullptr;
  ExportDesc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD;
  if (EnableIpc && !hDevice->isIntelBMGOrNewer())
    PhysicalMemDesc.pNext = &ExportDesc;
#endif

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
  if (!hContext)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  if (!hPhysMem)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  if (!ppIPCPhysMemHandleData)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  if (!pIPCPhysMemHandleDataSizeRet)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  if (!hPhysMem->EnableIpc)
    return UR_RESULT_ERROR_INVALID_ARGUMENT;

  // IPC-opened handles (consumer side) have ZePhysicalMem == nullptr; they
  // cannot be re-exported.
  if (!hPhysMem->ZePhysicalMem)
    return UR_RESULT_ERROR_INVALID_ARGUMENT;

  auto *HandleData = new (std::nothrow) ZeIPCPhysMemHandleData;
  if (!HandleData)
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  memset(HandleData, 0, sizeof(*HandleData));
  HandleData->Size = hPhysMem->Size;

  if (hPhysMem->Device->isIntelBMGOrNewer()) {
    // BMG and newer (Xe2+): obtain an opaque 64-byte IPC handle that can be
    // transferred to another process as raw bytes without fd passing.
    ze_ipc_mem_handle_type_ext_desc_t HandleTypeDesc = {};
    HandleTypeDesc.stype = ZE_STRUCTURE_TYPE_IPC_MEM_HANDLE_TYPE_EXT_DESC;
    HandleTypeDesc.pNext = nullptr;
    HandleTypeDesc.typeFlags = ZE_IPC_MEM_HANDLE_TYPE_FLAG_DEFAULT;

    ze_ipc_mem_handle_t IpcHandle = {};
    ze_result_t ZeRes = ZE_CALL_NOCHECK(
        zeMemGetIpcHandleWithProperties,
        (hContext->getZeHandle(),
         reinterpret_cast<const void *>(hPhysMem->ZePhysicalMem),
         &HandleTypeDesc, &IpcHandle));
    if (ZeRes == ZE_RESULT_ERROR_INVALID_ARGUMENT ||
        ZeRes == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      delete HandleData;
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }
    if (ZeRes != ZE_RESULT_SUCCESS) {
      delete HandleData;
      return ze2urResult(ZeRes);
    }

    HandleData->Kind = ZeIPCPhysMemHandleKind::OpaqueBMGOrNewer;
    HandleData->IpcHandle = IpcHandle;
  } else {
    // Pre-Xe2 (DG2, PVC, MTL, ARL): export via a DMA-BUF file descriptor.
    // Initialize fd to -1 so we can safely close it on error even if the
    // driver only partially completed the call.
    ze_external_memory_export_fd_t ExportFd = {};
    ExportFd.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD;
    ExportFd.pNext = nullptr;
    ExportFd.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD;
    ExportFd.fd = -1;

    ze_physical_mem_properties_t Props = {};
    Props.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES;
    Props.pNext = &ExportFd;

    ze_result_t ZeRes = ZE_CALL_NOCHECK(
        zePhysicalMemGetProperties,
        (hContext->getZeHandle(), hPhysMem->ZePhysicalMem, &Props));
    if (ZeRes != ZE_RESULT_SUCCESS) {
      if (ExportFd.fd >= 0)
        close(ExportFd.fd);
      delete HandleData;
      return ze2urResult(ZeRes);
    }
    if (ExportFd.fd < 0) {
      // Driver returned success but did not populate the fd — unsupported.
      delete HandleData;
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    HandleData->Kind = ZeIPCPhysMemHandleKind::DmaBufFd;
    HandleData->Pid = getpid();
    HandleData->Fd = ExportFd.fd;
  }

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
  if (!hContext)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  if (!pIPCPhysMemHandleData)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;

  auto *HandleData =
      static_cast<const ZeIPCPhysMemHandleData *>(pIPCPhysMemHandleData);

  ur_result_t Res = UR_RESULT_SUCCESS;
  if (HandleData->Kind == ZeIPCPhysMemHandleKind::OpaqueBMGOrNewer) {
    // BMG and newer (Xe2+): release the opaque IPC handle back to the driver.
    ze_result_t ZeResult = ZE_CALL_NOCHECK(
        zeMemPutIpcHandle, (hContext->getZeHandle(), HandleData->IpcHandle));
    Res = ze2urResult(ZeResult);
  } else if (HandleData->Kind == ZeIPCPhysMemHandleKind::DmaBufFd) {
    // Pre-Xe2 (DG2, PVC, MTL, ARL): close the exported DMA-BUF fd.
    close(HandleData->Fd);
  } else {
    Res = UR_RESULT_ERROR_INVALID_ARGUMENT;
  }
  delete HandleData;
  return Res;
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
  if (!hContext)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  if (!hDevice)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  if (!pIPCPhysMemHandleData)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  if (!phPhysMem)
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  if (ipcPhysMemHandleDataSize != sizeof(ZeIPCPhysMemHandleData))
    return UR_RESULT_ERROR_INVALID_VALUE;

  auto *HandleData =
      static_cast<const ZeIPCPhysMemHandleData *>(pIPCPhysMemHandleData);

  if (HandleData->Size == 0)
    return UR_RESULT_ERROR_INVALID_VALUE;

  if (HandleData->Kind == ZeIPCPhysMemHandleKind::OpaqueBMGOrNewer) {
    // BMG and newer (Xe2+): zeMemOpenIpcHandle creates a virtual mapping backed
    // by the exporter's physical memory and returns a pointer to it.
    // The consumer handle has ZePhysicalMem == nullptr; urPhysicalMemRelease
    // will call zeMemCloseIpcHandle when the refcount reaches zero.
    void *VirtualAddress = nullptr;
    ZE2UR_CALL(zeMemOpenIpcHandle,
               (hContext->getZeHandle(), hDevice->ZeDevice,
                HandleData->IpcHandle, ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED,
                &VirtualAddress));
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
  } else if (HandleData->Kind == ZeIPCPhysMemHandleKind::DmaBufFd) {
    // Pre-Xe2 (DG2, PVC, MTL, ARL): obtain a usable fd in the current process
    // via dup() (same-process) or pidfd_getfd(2) (cross-process, Linux 5.6+).
    if (HandleData->Fd < 0 || HandleData->Pid <= 0)
      return UR_RESULT_ERROR_INVALID_VALUE;

    int ImportFdNum = -1;
    if (HandleData->Pid == getpid()) {
      ImportFdNum = dup(HandleData->Fd);
      if (ImportFdNum < 0)
        return UR_RESULT_ERROR_INVALID_VALUE;
    } else {
      int PidFd = static_cast<int>(syscall(SYS_pidfd_open, HandleData->Pid, 0));
      if (PidFd < 0)
        return errno == EPERM ? UR_RESULT_ERROR_INVALID_ARGUMENT
                              : UR_RESULT_ERROR_INVALID_VALUE;
      ImportFdNum =
          static_cast<int>(syscall(SYS_pidfd_getfd, PidFd, HandleData->Fd, 0));
      int SavedErrno = errno; // save before close() may overwrite it
      close(PidFd);
      if (ImportFdNum < 0)
        return SavedErrno == EPERM ? UR_RESULT_ERROR_INVALID_ARGUMENT
                                   : UR_RESULT_ERROR_INVALID_VALUE;
    }

    ze_external_memory_import_fd_t ImportFd = {};
    ImportFd.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
    ImportFd.pNext = nullptr;
    ImportFd.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD;
    ImportFd.fd = ImportFdNum;

    ZeStruct<ze_physical_mem_desc_t> PhysMemDesc;
    PhysMemDesc.pNext = &ImportFd;
    PhysMemDesc.flags = 0;
    PhysMemDesc.size = HandleData->Size;

    ze_physical_mem_handle_t ZePhysMem;
    ze_result_t ZeRes = ZE_CALL_NOCHECK(
        zePhysicalMemCreate,
        (hContext->getZeHandle(), hDevice->ZeDevice, &PhysMemDesc, &ZePhysMem));
    // The driver has dup'd ImportFdNum internally; close our copy now.
    close(ImportFdNum);

    if (ZeRes != ZE_RESULT_SUCCESS)
      return ze2urResult(ZeRes);

    try {
      *phPhysMem = new ur_physical_mem_handle_t_(ZePhysMem, hContext, hDevice,
                                                 HandleData->Size,
                                                 /*EnableIpc=*/true);
    } catch (const std::bad_alloc &) {
      zePhysicalMemDestroy(hContext->getZeHandle(), ZePhysMem);
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
      zePhysicalMemDestroy(hContext->getZeHandle(), ZePhysMem);
      return UR_RESULT_ERROR_UNKNOWN;
    }
  } else {
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
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
  if (!hContext)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  if (!hPhysMem)
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;

  (void)hContext;
  // Delegate to urPhysicalMemRelease so the refcount is respected.  For
  // IPC-opened BMG+ handles (IpcVirtualAddress != nullptr) urPhysicalMemRelease
  // calls zeMemCloseIpcHandle; for pre-Xe2 IPC-opened handles it calls
  // zePhysicalMemDestroy.
  return ur::level_zero::urPhysicalMemRelease(hPhysMem);
}

} // namespace ur::level_zero
