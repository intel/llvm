//===----------------------------------------------------------------------===//
/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_interceptor.cpp
 *
 */

#include "msan_interceptor.hpp"
#include "msan_ddi.hpp"
#include "msan_options.hpp"
#include "msan_report.hpp"
#include "msan_shadow.hpp"
#include "sanitizer_common/sanitizer_stacktrace.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {
namespace msan {

MsanInterceptor::MsanInterceptor() {}

MsanInterceptor::~MsanInterceptor() {
    // We must release these objects before releasing adapters, since
    // they may use the adapter in their destructor
    for (const auto &[_, DeviceInfo] : m_DeviceMap) {
        DeviceInfo->Shadow->Destory();
    }

    m_MemBufferMap.clear();
    m_AllocationMap.clear();
    m_KernelMap.clear();
    m_ContextMap.clear();

    for (auto Adapter : m_Adapters) {
        getContext()->urDdiTable.Global.pfnAdapterRelease(Adapter);
    }
}

ur_result_t MsanInterceptor::allocateMemory(ur_context_handle_t Context,
                                            ur_device_handle_t Device,
                                            const ur_usm_desc_t *Properties,
                                            ur_usm_pool_handle_t Pool,
                                            size_t Size, void **ResultPtr) {

    auto ContextInfo = getContextInfo(Context);
    std::shared_ptr<DeviceInfo> DeviceInfo =
        Device ? getDeviceInfo(Device) : nullptr;

    void *Allocated = nullptr;

    UR_CALL(getContext()->urDdiTable.USM.pfnDeviceAlloc(
        Context, Device, Properties, Pool, Size, &Allocated));

    *ResultPtr = Allocated;

    auto AI =
        std::make_shared<MsanAllocInfo>(MsanAllocInfo{(uptr)Allocated,
                                                      Size,
                                                      false,
                                                      Context,
                                                      Device,
                                                      GetCurrentBacktrace(),
                                                      {}});

    AI->print();

    // For updating shadow memory
    ContextInfo->insertAllocInfo({Device}, AI);

    // For memory release
    {
        std::scoped_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
        m_AllocationMap.emplace(AI->AllocBegin, std::move(AI));
    }

    return UR_RESULT_SUCCESS;
}

ur_result_t MsanInterceptor::preLaunchKernel(ur_kernel_handle_t Kernel,
                                             ur_queue_handle_t Queue,
                                             USMLaunchInfo &LaunchInfo) {
    auto Context = GetContext(Queue);
    auto Device = GetDevice(Queue);
    auto ContextInfo = getContextInfo(Context);
    auto DeviceInfo = getDeviceInfo(Device);

    ManagedQueue InternalQueue(Context, Device);
    if (!InternalQueue) {
        getContext()->logger.error("Failed to create internal queue");
        return UR_RESULT_ERROR_INVALID_QUEUE;
    }

    UR_CALL(prepareLaunch(DeviceInfo, InternalQueue, Kernel, LaunchInfo));

    UR_CALL(updateShadowMemory(ContextInfo, DeviceInfo, InternalQueue));

    return UR_RESULT_SUCCESS;
}

ur_result_t MsanInterceptor::postLaunchKernel(ur_kernel_handle_t Kernel,
                                              ur_queue_handle_t Queue,
                                              USMLaunchInfo &LaunchInfo) {
    // FIXME: We must use block operation here, until we support urEventSetCallback
    auto Result = getContext()->urDdiTable.Queue.pfnFinish(Queue);

    if (Result == UR_RESULT_SUCCESS) {
        const auto &Report = LaunchInfo.Data->Report;

        if (!Report.Flag) {
            return Result;
        }

        ReportUsesUninitializedValue(LaunchInfo.Data->Report, Kernel);

        exitWithErrors();
    }

    return Result;
}

ur_result_t
MsanInterceptor::enqueueAllocInfo(std::shared_ptr<DeviceInfo> &DeviceInfo,
                                  ur_queue_handle_t Queue,
                                  std::shared_ptr<MsanAllocInfo> &AI) {
    return DeviceInfo->Shadow->EnqueuePoisonShadow(Queue, AI->AllocBegin,
                                                   AI->AllocSize, 0xff);
}

ur_result_t
MsanInterceptor::updateShadowMemory(std::shared_ptr<ContextInfo> &ContextInfo,
                                    std::shared_ptr<DeviceInfo> &DeviceInfo,
                                    ur_queue_handle_t Queue) {
    auto &AllocInfos = ContextInfo->AllocInfosMap[DeviceInfo->Handle];
    std::scoped_lock<ur_shared_mutex> Guard(AllocInfos.Mutex);

    for (auto &AI : AllocInfos.List) {
        UR_CALL(enqueueAllocInfo(DeviceInfo, Queue, AI));
    }
    AllocInfos.List.clear();

    return UR_RESULT_SUCCESS;
}

ur_result_t MsanInterceptor::registerProgram(ur_program_handle_t Program) {
    ur_result_t Result = UR_RESULT_SUCCESS;

    getContext()->logger.info("registerSpirKernels");
    Result = registerSpirKernels(Program);
    if (Result != UR_RESULT_SUCCESS) {
        return Result;
    }

    return Result;
}

ur_result_t MsanInterceptor::unregisterProgram(ur_program_handle_t) {
    return UR_RESULT_SUCCESS;
}

ur_result_t MsanInterceptor::registerSpirKernels(ur_program_handle_t Program) {
    auto Context = GetContext(Program);
    std::vector<ur_device_handle_t> Devices = GetDevices(Program);

    for (auto Device : Devices) {
        size_t MetadataSize;
        void *MetadataPtr;
        ur_result_t Result =
            getContext()->urDdiTable.Program.pfnGetGlobalVariablePointer(
                Device, Program, kSPIR_MsanSpirKernelMetadata, &MetadataSize,
                &MetadataPtr);
        if (Result != UR_RESULT_SUCCESS) {
            continue;
        }

        const uint64_t NumOfSpirKernel = MetadataSize / sizeof(SpirKernelInfo);
        assert((MetadataSize % sizeof(SpirKernelInfo) == 0) &&
               "SpirKernelMetadata size is not correct");

        ManagedQueue Queue(Context, Device);

        std::vector<SpirKernelInfo> SKInfo(NumOfSpirKernel);
        Result = getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
            Queue, true, &SKInfo[0], MetadataPtr,
            sizeof(SpirKernelInfo) * NumOfSpirKernel, 0, nullptr, nullptr);
        if (Result != UR_RESULT_SUCCESS) {
            getContext()->logger.error("Can't read the value of <{}>: {}",
                                       kSPIR_MsanSpirKernelMetadata, Result);
            return Result;
        }

        auto PI = getProgramInfo(Program);
        for (const auto &SKI : SKInfo) {
            if (SKI.Size == 0) {
                continue;
            }
            std::vector<char> KernelNameV(SKI.Size);
            Result = getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
                Queue, true, KernelNameV.data(), (void *)SKI.KernelName,
                sizeof(char) * SKI.Size, 0, nullptr, nullptr);
            if (Result != UR_RESULT_SUCCESS) {
                getContext()->logger.error("Can't read kernel name: {}",
                                           Result);
                return Result;
            }

            std::string KernelName =
                std::string(KernelNameV.begin(), KernelNameV.end());

            getContext()->logger.info(
                "SpirKernel(name='{}', isInstrumented={})", KernelName, true);

            PI->InstrumentedKernels.insert(KernelName);
        }
        getContext()->logger.info("Number of sanitized kernel: {}",
                                  PI->InstrumentedKernels.size());
    }

    return UR_RESULT_SUCCESS;
}

ur_result_t MsanInterceptor::insertContext(ur_context_handle_t Context,
                                           std::shared_ptr<ContextInfo> &CI) {
    std::scoped_lock<ur_shared_mutex> Guard(m_ContextMapMutex);

    if (m_ContextMap.find(Context) != m_ContextMap.end()) {
        CI = m_ContextMap.at(Context);
        return UR_RESULT_SUCCESS;
    }

    CI = std::make_shared<ContextInfo>(Context);

    // Don't move CI, since it's a return value as well
    m_ContextMap.emplace(Context, CI);

    return UR_RESULT_SUCCESS;
}

ur_result_t MsanInterceptor::eraseContext(ur_context_handle_t Context) {
    std::scoped_lock<ur_shared_mutex> Guard(m_ContextMapMutex);
    assert(m_ContextMap.find(Context) != m_ContextMap.end());
    m_ContextMap.erase(Context);
    // TODO: Remove devices in each context
    return UR_RESULT_SUCCESS;
}

ur_result_t MsanInterceptor::insertDevice(ur_device_handle_t Device,
                                          std::shared_ptr<DeviceInfo> &DI) {
    std::scoped_lock<ur_shared_mutex> Guard(m_DeviceMapMutex);

    if (m_DeviceMap.find(Device) != m_DeviceMap.end()) {
        DI = m_DeviceMap.at(Device);
        return UR_RESULT_SUCCESS;
    }

    DI = std::make_shared<DeviceInfo>(Device);

    DI->IsSupportSharedSystemUSM = GetDeviceUSMCapability(
        Device, UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT);

    // Query alignment
    UR_CALL(getContext()->urDdiTable.Device.pfnGetInfo(
        Device, UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN, sizeof(DI->Alignment),
        &DI->Alignment, nullptr));

    // Don't move DI, since it's a return value as well
    m_DeviceMap.emplace(Device, DI);

    return UR_RESULT_SUCCESS;
}

ur_result_t MsanInterceptor::eraseDevice(ur_device_handle_t Device) {
    std::scoped_lock<ur_shared_mutex> Guard(m_DeviceMapMutex);
    assert(m_DeviceMap.find(Device) != m_DeviceMap.end());
    m_DeviceMap.erase(Device);
    // TODO: Remove devices in each context
    return UR_RESULT_SUCCESS;
}

ur_result_t MsanInterceptor::insertProgram(ur_program_handle_t Program) {
    std::scoped_lock<ur_shared_mutex> Guard(m_ProgramMapMutex);
    if (m_ProgramMap.find(Program) != m_ProgramMap.end()) {
        return UR_RESULT_SUCCESS;
    }
    m_ProgramMap.emplace(Program, std::make_shared<ProgramInfo>(Program));
    return UR_RESULT_SUCCESS;
}

ur_result_t MsanInterceptor::eraseProgram(ur_program_handle_t Program) {
    std::scoped_lock<ur_shared_mutex> Guard(m_ProgramMapMutex);
    assert(m_ProgramMap.find(Program) != m_ProgramMap.end());
    m_ProgramMap.erase(Program);
    return UR_RESULT_SUCCESS;
}

ur_result_t MsanInterceptor::insertKernel(ur_kernel_handle_t Kernel) {
    std::scoped_lock<ur_shared_mutex> Guard(m_KernelMapMutex);
    if (m_KernelMap.find(Kernel) != m_KernelMap.end()) {
        return UR_RESULT_SUCCESS;
    }
    m_KernelMap.emplace(Kernel, std::make_shared<KernelInfo>(Kernel));
    return UR_RESULT_SUCCESS;
}

ur_result_t MsanInterceptor::eraseKernel(ur_kernel_handle_t Kernel) {
    std::scoped_lock<ur_shared_mutex> Guard(m_KernelMapMutex);
    assert(m_KernelMap.find(Kernel) != m_KernelMap.end());
    m_KernelMap.erase(Kernel);
    return UR_RESULT_SUCCESS;
}

ur_result_t
MsanInterceptor::insertMemBuffer(std::shared_ptr<MemBuffer> MemBuffer) {
    std::scoped_lock<ur_shared_mutex> Guard(m_MemBufferMapMutex);
    assert(m_MemBufferMap.find(ur_cast<ur_mem_handle_t>(MemBuffer.get())) ==
           m_MemBufferMap.end());
    m_MemBufferMap.emplace(reinterpret_cast<ur_mem_handle_t>(MemBuffer.get()),
                           MemBuffer);
    return UR_RESULT_SUCCESS;
}

ur_result_t MsanInterceptor::eraseMemBuffer(ur_mem_handle_t MemHandle) {
    std::scoped_lock<ur_shared_mutex> Guard(m_MemBufferMapMutex);
    assert(m_MemBufferMap.find(MemHandle) != m_MemBufferMap.end());
    m_MemBufferMap.erase(MemHandle);
    return UR_RESULT_SUCCESS;
}

std::shared_ptr<MemBuffer>
MsanInterceptor::getMemBuffer(ur_mem_handle_t MemHandle) {
    std::shared_lock<ur_shared_mutex> Guard(m_MemBufferMapMutex);
    if (m_MemBufferMap.find(MemHandle) != m_MemBufferMap.end()) {
        return m_MemBufferMap[MemHandle];
    }
    return nullptr;
}

ur_result_t MsanInterceptor::prepareLaunch(
    std::shared_ptr<DeviceInfo> &DeviceInfo, ur_queue_handle_t Queue,
    ur_kernel_handle_t Kernel, USMLaunchInfo &LaunchInfo) {
    auto Program = GetProgram(Kernel);

    auto EnqueueWriteGlobal =
        [&Queue, &Program](const char *Name, const void *Value, size_t Size) {
            auto Result =
                getContext()->urDdiTable.Enqueue.pfnDeviceGlobalVariableWrite(
                    Queue, Program, Name, false, Size, 0, Value, 0, nullptr,
                    nullptr);
            if (Result != UR_RESULT_SUCCESS) {
                getContext()->logger.error(
                    "Failed to write device global \"{}\": {}", Name, Result);
                return Result;
            }
            return UR_RESULT_SUCCESS;
        };

    // Set membuffer arguments
    auto KernelInfo = getKernelInfo(Kernel);
    assert(KernelInfo && "Kernel must be instrumented");

    for (const auto &[ArgIndex, MemBuffer] : KernelInfo->BufferArgs) {
        char *ArgPointer = nullptr;
        UR_CALL(MemBuffer->getHandle(DeviceInfo->Handle, ArgPointer));
        ur_result_t URes = getContext()->urDdiTable.Kernel.pfnSetArgPointer(
            Kernel, ArgIndex, nullptr, ArgPointer);
        if (URes != UR_RESULT_SUCCESS) {
            getContext()->logger.error(
                "Failed to set buffer {} as the {} arg to kernel {}: {}",
                ur_cast<ur_mem_handle_t>(MemBuffer.get()), ArgIndex, Kernel,
                URes);
        }
    }

    // Set LaunchInfo
    LaunchInfo.Data->GlobalShadowOffset = DeviceInfo->Shadow->ShadowBegin;
    LaunchInfo.Data->GlobalShadowOffsetEnd = DeviceInfo->Shadow->ShadowEnd;
    LaunchInfo.Data->DeviceTy = DeviceInfo->Type;
    LaunchInfo.Data->Debug = getOptions().Debug ? 1 : 0;

    getContext()->logger.info(
        "launch_info {} (GlobalShadow={}, Device={}, Debug={})",
        (void *)LaunchInfo.Data, LaunchInfo.Data->GlobalShadowOffset,
        ToString(LaunchInfo.Data->DeviceTy), LaunchInfo.Data->Debug);

    UR_CALL(
        EnqueueWriteGlobal("__MsanLaunchInfo", &LaunchInfo.Data, sizeof(uptr)));

    return UR_RESULT_SUCCESS;
}

std::optional<MsanAllocationIterator>
MsanInterceptor::findAllocInfoByAddress(uptr Address) {
    std::shared_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
    auto It = m_AllocationMap.upper_bound(Address);
    if (It == m_AllocationMap.begin()) {
        return std::optional<MsanAllocationIterator>{};
    }
    --It;
    // Make sure we got the right MsanAllocInfo
    assert(Address >= It->second->AllocBegin &&
           Address < It->second->AllocBegin + It->second->AllocSize &&
           "Wrong MsanAllocInfo for the address");
    return It;
}

std::vector<MsanAllocationIterator>
MsanInterceptor::findAllocInfoByContext(ur_context_handle_t Context) {
    std::shared_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
    std::vector<MsanAllocationIterator> AllocInfos;
    for (auto It = m_AllocationMap.begin(); It != m_AllocationMap.end(); It++) {
        const auto &[_, AI] = *It;
        if (AI->Context == Context) {
            AllocInfos.emplace_back(It);
        }
    }
    return AllocInfos;
}

ur_result_t DeviceInfo::allocShadowMemory(ur_context_handle_t Context) {
    Shadow = GetMsanShadowMemory(Context, Handle, Type);
    assert(Shadow && "Failed to get shadow memory");
    UR_CALL(Shadow->Setup());
    getContext()->logger.info("ShadowMemory(Global): {} - {}",
                              (void *)Shadow->ShadowBegin,
                              (void *)Shadow->ShadowEnd);
    return UR_RESULT_SUCCESS;
}

bool ProgramInfo::isKernelInstrumented(ur_kernel_handle_t Kernel) const {
    const auto Name = GetKernelName(Kernel);
    return InstrumentedKernels.find(Name) != InstrumentedKernels.end();
}

ContextInfo::~ContextInfo() {
    [[maybe_unused]] auto Result =
        getContext()->urDdiTable.Context.pfnRelease(Handle);
    assert(Result == UR_RESULT_SUCCESS);
}

ur_result_t USMLaunchInfo::initialize() {
    UR_CALL(getContext()->urDdiTable.Context.pfnRetain(Context));
    UR_CALL(getContext()->urDdiTable.Device.pfnRetain(Device));
    UR_CALL(getContext()->urDdiTable.USM.pfnSharedAlloc(
        Context, Device, nullptr, nullptr, sizeof(MsanLaunchInfo),
        (void **)&Data));
    *Data = MsanLaunchInfo{};
    return UR_RESULT_SUCCESS;
}

USMLaunchInfo::~USMLaunchInfo() {
    [[maybe_unused]] ur_result_t Result;
    if (Data) {
        Result = getContext()->urDdiTable.USM.pfnFree(Context, (void *)Data);
        assert(Result == UR_RESULT_SUCCESS);
    }
    Result = getContext()->urDdiTable.Context.pfnRelease(Context);
    assert(Result == UR_RESULT_SUCCESS);
    Result = getContext()->urDdiTable.Device.pfnRelease(Device);
    assert(Result == UR_RESULT_SUCCESS);
}

} // namespace msan

using namespace msan;

static MsanInterceptor *interceptor;

MsanInterceptor *getMsanInterceptor() { return interceptor; }

void initMsanInterceptor() {
    if (interceptor) {
        return;
    }
    interceptor = new MsanInterceptor();
}

void destroyMsanInterceptor() {
    delete interceptor;
    interceptor = nullptr;
}

} // namespace ur_sanitizer_layer
