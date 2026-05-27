//===---------- physical_mem.cpp - Level Zero Adapter v2 ----------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "physical_mem.hpp"
#include "../common.hpp"
#include "../device.hpp"

#ifdef __linux__
#include <cerrno>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "context.hpp"

namespace ur::level_zero {

ur_result_t urPhysicalMemCreate(ur_context_handle_t hContext,
                                ur_device_handle_t hDevice, size_t size,
                                const ur_physical_mem_properties_t *pProperties,
                                ur_physical_mem_handle_t *phPhysicalMem) {
  ZeStruct<ze_physical_mem_desc_t> PhysicalMemDesc;
  PhysicalMemDesc.flags = 0;
  PhysicalMemDesc.size = size;

  // If IPC export is requested, chain in the export descriptor so the
  // physical memory can later be shared via urIPCGetPhysMemHandleExp.
  ze_external_memory_export_desc_t ExportDesc = {};
  ExportDesc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC;
  ExportDesc.pNext = nullptr;
  ExportDesc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD;
  bool EnableIpc =
      pProperties && (pProperties->flags & UR_PHYSICAL_MEM_FLAG_ENABLE_IPC);
  if (EnableIpc)
    PhysicalMemDesc.pNext = &ExportDesc;

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

  if (checkL0LoaderTeardown()) {
    ZE2UR_CALL(zePhysicalMemDestroy, (hPhysicalMem->Context->getZeHandle(),
                                      hPhysicalMem->ZePhysicalMem));
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

  // Export the physical memory object as an opaque file descriptor.
  ze_external_memory_export_fd_t ExportFd = {};
  ExportFd.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD;
  ExportFd.pNext = nullptr;
  ExportFd.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD;
  // Initialize to -1 so we can safely detect if the driver opened an fd even
  // on a partially-completed call that then returns an error.
  ExportFd.fd = -1;

  ze_physical_mem_properties_t Props = {};
  Props.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES;
  Props.pNext = &ExportFd;

  // Use ZE_CALL_NOCHECK so we can close any fd the driver may have opened
  // before returning an error, rather than leaking it via an early return.
  ze_result_t ZeRes = ZE_CALL_NOCHECK(
      zePhysicalMemGetProperties,
      (hContext->getZeHandle(), hPhysMem->ZePhysicalMem, &Props));
  if (ZeRes != ZE_RESULT_SUCCESS) {
    if (ExportFd.fd >= 0)
      close(ExportFd.fd);
    return ze2urResult(ZeRes);
  }

  auto *HandleData = new (std::nothrow) ZeIPCPhysMemHandleData;
  if (!HandleData) {
    close(ExportFd.fd);
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  // Store the exporting process's PID and fd.  The fd stays open until
  // urIPCPutPhysMemHandleExp is called.  Cross-process consumers use
  // pidfd_getfd(2) to obtain their own duplicate of this fd.
  HandleData->Pid = getpid();
  HandleData->Fd = ExportFd.fd;
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

ur_result_t
urIPCPutPhysMemHandleExp([[maybe_unused]] ur_context_handle_t hContext,
                         const void *pIPCPhysMemHandleData) {
#ifdef __linux__
  auto *HandleData =
      static_cast<const ZeIPCPhysMemHandleData *>(pIPCPhysMemHandleData);
  close(HandleData->Fd);
  delete HandleData;
  return UR_RESULT_SUCCESS;
#else
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

  if (HandleData->Fd < 0 || HandleData->Pid <= 0 || HandleData->Size == 0)
    return UR_RESULT_ERROR_INVALID_VALUE;

  // Obtain a usable fd in the current process.  For same-process opens
  // (e.g. conformance tests) dup() suffices.  For cross-process opens
  // use pidfd_getfd(2) which requires the exporting process to be
  // ptrace-accessible (e.g. via prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)).
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
  ze_result_t ZeRes = zePhysicalMemCreate(
      hContext->getZeHandle(), hDevice->ZeDevice, &PhysMemDesc, &ZePhysMem);
  // Driver has dup'd ImportFdNum internally; close our copy now.
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

ur_result_t
urIPCClosePhysMemHandleExp([[maybe_unused]] ur_context_handle_t hContext,
                           ur_physical_mem_handle_t hPhysMem) {
  // Delegate to urPhysicalMemRelease so the refcount is respected: if the
  // handle has been retained (refcount > 1) it will not be destroyed until
  // all references are released.
  return ur::level_zero::urPhysicalMemRelease(hPhysMem);
}

} // namespace ur::level_zero
