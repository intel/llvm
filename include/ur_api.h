/*
 *
 * Copyright (C) 2022 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_api.h
 * @version v0.9-r0
 *
 */
#ifndef UR_API_H_INCLUDED
#define UR_API_H_INCLUDED
#if defined(__cplusplus)
#pragma once
#endif

// standard headers
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

// Intel 'oneAPI' Unified Runtime function registry
#if !defined(__GNUC__)
#pragma region registry
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Defines unique stable identifiers for all functions
typedef enum ur_function_t {
    UR_FUNCTION_CONTEXT_CREATE = 1,                                            ///< Enumerator for ::urContextCreate
    UR_FUNCTION_CONTEXT_RETAIN = 2,                                            ///< Enumerator for ::urContextRetain
    UR_FUNCTION_CONTEXT_RELEASE = 3,                                           ///< Enumerator for ::urContextRelease
    UR_FUNCTION_CONTEXT_GET_INFO = 4,                                          ///< Enumerator for ::urContextGetInfo
    UR_FUNCTION_CONTEXT_GET_NATIVE_HANDLE = 5,                                 ///< Enumerator for ::urContextGetNativeHandle
    UR_FUNCTION_CONTEXT_CREATE_WITH_NATIVE_HANDLE = 6,                         ///< Enumerator for ::urContextCreateWithNativeHandle
    UR_FUNCTION_CONTEXT_SET_EXTENDED_DELETER = 7,                              ///< Enumerator for ::urContextSetExtendedDeleter
    UR_FUNCTION_DEVICE_GET = 8,                                                ///< Enumerator for ::urDeviceGet
    UR_FUNCTION_DEVICE_GET_INFO = 9,                                           ///< Enumerator for ::urDeviceGetInfo
    UR_FUNCTION_DEVICE_RETAIN = 10,                                            ///< Enumerator for ::urDeviceRetain
    UR_FUNCTION_DEVICE_RELEASE = 11,                                           ///< Enumerator for ::urDeviceRelease
    UR_FUNCTION_DEVICE_PARTITION = 12,                                         ///< Enumerator for ::urDevicePartition
    UR_FUNCTION_DEVICE_SELECT_BINARY = 13,                                     ///< Enumerator for ::urDeviceSelectBinary
    UR_FUNCTION_DEVICE_GET_NATIVE_HANDLE = 14,                                 ///< Enumerator for ::urDeviceGetNativeHandle
    UR_FUNCTION_DEVICE_CREATE_WITH_NATIVE_HANDLE = 15,                         ///< Enumerator for ::urDeviceCreateWithNativeHandle
    UR_FUNCTION_DEVICE_GET_GLOBAL_TIMESTAMPS = 16,                             ///< Enumerator for ::urDeviceGetGlobalTimestamps
    UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH = 17,                                    ///< Enumerator for ::urEnqueueKernelLaunch
    UR_FUNCTION_ENQUEUE_EVENTS_WAIT = 18,                                      ///< Enumerator for ::urEnqueueEventsWait
    UR_FUNCTION_ENQUEUE_EVENTS_WAIT_WITH_BARRIER = 19,                         ///< Enumerator for ::urEnqueueEventsWaitWithBarrier
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_READ = 20,                                  ///< Enumerator for ::urEnqueueMemBufferRead
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_WRITE = 21,                                 ///< Enumerator for ::urEnqueueMemBufferWrite
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_READ_RECT = 22,                             ///< Enumerator for ::urEnqueueMemBufferReadRect
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_WRITE_RECT = 23,                            ///< Enumerator for ::urEnqueueMemBufferWriteRect
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_COPY = 24,                                  ///< Enumerator for ::urEnqueueMemBufferCopy
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_COPY_RECT = 25,                             ///< Enumerator for ::urEnqueueMemBufferCopyRect
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_FILL = 26,                                  ///< Enumerator for ::urEnqueueMemBufferFill
    UR_FUNCTION_ENQUEUE_MEM_IMAGE_READ = 27,                                   ///< Enumerator for ::urEnqueueMemImageRead
    UR_FUNCTION_ENQUEUE_MEM_IMAGE_WRITE = 28,                                  ///< Enumerator for ::urEnqueueMemImageWrite
    UR_FUNCTION_ENQUEUE_MEM_IMAGE_COPY = 29,                                   ///< Enumerator for ::urEnqueueMemImageCopy
    UR_FUNCTION_ENQUEUE_MEM_BUFFER_MAP = 30,                                   ///< Enumerator for ::urEnqueueMemBufferMap
    UR_FUNCTION_ENQUEUE_MEM_UNMAP = 31,                                        ///< Enumerator for ::urEnqueueMemUnmap
    UR_FUNCTION_ENQUEUE_USM_FILL = 32,                                         ///< Enumerator for ::urEnqueueUSMFill
    UR_FUNCTION_ENQUEUE_USM_MEMCPY = 33,                                       ///< Enumerator for ::urEnqueueUSMMemcpy
    UR_FUNCTION_ENQUEUE_USM_PREFETCH = 34,                                     ///< Enumerator for ::urEnqueueUSMPrefetch
    UR_FUNCTION_ENQUEUE_USM_ADVISE = 35,                                       ///< Enumerator for ::urEnqueueUSMAdvise
    UR_FUNCTION_ENQUEUE_DEVICE_GLOBAL_VARIABLE_WRITE = 38,                     ///< Enumerator for ::urEnqueueDeviceGlobalVariableWrite
    UR_FUNCTION_ENQUEUE_DEVICE_GLOBAL_VARIABLE_READ = 39,                      ///< Enumerator for ::urEnqueueDeviceGlobalVariableRead
    UR_FUNCTION_EVENT_GET_INFO = 40,                                           ///< Enumerator for ::urEventGetInfo
    UR_FUNCTION_EVENT_GET_PROFILING_INFO = 41,                                 ///< Enumerator for ::urEventGetProfilingInfo
    UR_FUNCTION_EVENT_WAIT = 42,                                               ///< Enumerator for ::urEventWait
    UR_FUNCTION_EVENT_RETAIN = 43,                                             ///< Enumerator for ::urEventRetain
    UR_FUNCTION_EVENT_RELEASE = 44,                                            ///< Enumerator for ::urEventRelease
    UR_FUNCTION_EVENT_GET_NATIVE_HANDLE = 45,                                  ///< Enumerator for ::urEventGetNativeHandle
    UR_FUNCTION_EVENT_CREATE_WITH_NATIVE_HANDLE = 46,                          ///< Enumerator for ::urEventCreateWithNativeHandle
    UR_FUNCTION_EVENT_SET_CALLBACK = 47,                                       ///< Enumerator for ::urEventSetCallback
    UR_FUNCTION_KERNEL_CREATE = 48,                                            ///< Enumerator for ::urKernelCreate
    UR_FUNCTION_KERNEL_SET_ARG_VALUE = 49,                                     ///< Enumerator for ::urKernelSetArgValue
    UR_FUNCTION_KERNEL_SET_ARG_LOCAL = 50,                                     ///< Enumerator for ::urKernelSetArgLocal
    UR_FUNCTION_KERNEL_GET_INFO = 51,                                          ///< Enumerator for ::urKernelGetInfo
    UR_FUNCTION_KERNEL_GET_GROUP_INFO = 52,                                    ///< Enumerator for ::urKernelGetGroupInfo
    UR_FUNCTION_KERNEL_GET_SUB_GROUP_INFO = 53,                                ///< Enumerator for ::urKernelGetSubGroupInfo
    UR_FUNCTION_KERNEL_RETAIN = 54,                                            ///< Enumerator for ::urKernelRetain
    UR_FUNCTION_KERNEL_RELEASE = 55,                                           ///< Enumerator for ::urKernelRelease
    UR_FUNCTION_KERNEL_SET_ARG_POINTER = 56,                                   ///< Enumerator for ::urKernelSetArgPointer
    UR_FUNCTION_KERNEL_SET_EXEC_INFO = 57,                                     ///< Enumerator for ::urKernelSetExecInfo
    UR_FUNCTION_KERNEL_SET_ARG_SAMPLER = 58,                                   ///< Enumerator for ::urKernelSetArgSampler
    UR_FUNCTION_KERNEL_SET_ARG_MEM_OBJ = 59,                                   ///< Enumerator for ::urKernelSetArgMemObj
    UR_FUNCTION_KERNEL_SET_SPECIALIZATION_CONSTANTS = 60,                      ///< Enumerator for ::urKernelSetSpecializationConstants
    UR_FUNCTION_KERNEL_GET_NATIVE_HANDLE = 61,                                 ///< Enumerator for ::urKernelGetNativeHandle
    UR_FUNCTION_KERNEL_CREATE_WITH_NATIVE_HANDLE = 62,                         ///< Enumerator for ::urKernelCreateWithNativeHandle
    UR_FUNCTION_MEM_IMAGE_CREATE = 63,                                         ///< Enumerator for ::urMemImageCreate
    UR_FUNCTION_MEM_BUFFER_CREATE = 64,                                        ///< Enumerator for ::urMemBufferCreate
    UR_FUNCTION_MEM_RETAIN = 65,                                               ///< Enumerator for ::urMemRetain
    UR_FUNCTION_MEM_RELEASE = 66,                                              ///< Enumerator for ::urMemRelease
    UR_FUNCTION_MEM_BUFFER_PARTITION = 67,                                     ///< Enumerator for ::urMemBufferPartition
    UR_FUNCTION_MEM_GET_NATIVE_HANDLE = 68,                                    ///< Enumerator for ::urMemGetNativeHandle
    UR_FUNCTION_ENQUEUE_READ_HOST_PIPE = 69,                                   ///< Enumerator for ::urEnqueueReadHostPipe
    UR_FUNCTION_MEM_GET_INFO = 70,                                             ///< Enumerator for ::urMemGetInfo
    UR_FUNCTION_MEM_IMAGE_GET_INFO = 71,                                       ///< Enumerator for ::urMemImageGetInfo
    UR_FUNCTION_PLATFORM_GET = 72,                                             ///< Enumerator for ::urPlatformGet
    UR_FUNCTION_PLATFORM_GET_INFO = 73,                                        ///< Enumerator for ::urPlatformGetInfo
    UR_FUNCTION_PLATFORM_GET_API_VERSION = 74,                                 ///< Enumerator for ::urPlatformGetApiVersion
    UR_FUNCTION_PLATFORM_GET_NATIVE_HANDLE = 75,                               ///< Enumerator for ::urPlatformGetNativeHandle
    UR_FUNCTION_PLATFORM_CREATE_WITH_NATIVE_HANDLE = 76,                       ///< Enumerator for ::urPlatformCreateWithNativeHandle
    UR_FUNCTION_PROGRAM_CREATE_WITH_IL = 78,                                   ///< Enumerator for ::urProgramCreateWithIL
    UR_FUNCTION_PROGRAM_CREATE_WITH_BINARY = 79,                               ///< Enumerator for ::urProgramCreateWithBinary
    UR_FUNCTION_PROGRAM_BUILD = 80,                                            ///< Enumerator for ::urProgramBuild
    UR_FUNCTION_PROGRAM_COMPILE = 81,                                          ///< Enumerator for ::urProgramCompile
    UR_FUNCTION_PROGRAM_LINK = 82,                                             ///< Enumerator for ::urProgramLink
    UR_FUNCTION_PROGRAM_RETAIN = 83,                                           ///< Enumerator for ::urProgramRetain
    UR_FUNCTION_PROGRAM_RELEASE = 84,                                          ///< Enumerator for ::urProgramRelease
    UR_FUNCTION_PROGRAM_GET_FUNCTION_POINTER = 85,                             ///< Enumerator for ::urProgramGetFunctionPointer
    UR_FUNCTION_PROGRAM_GET_INFO = 86,                                         ///< Enumerator for ::urProgramGetInfo
    UR_FUNCTION_PROGRAM_GET_BUILD_INFO = 87,                                   ///< Enumerator for ::urProgramGetBuildInfo
    UR_FUNCTION_PROGRAM_SET_SPECIALIZATION_CONSTANTS = 88,                     ///< Enumerator for ::urProgramSetSpecializationConstants
    UR_FUNCTION_PROGRAM_GET_NATIVE_HANDLE = 89,                                ///< Enumerator for ::urProgramGetNativeHandle
    UR_FUNCTION_PROGRAM_CREATE_WITH_NATIVE_HANDLE = 90,                        ///< Enumerator for ::urProgramCreateWithNativeHandle
    UR_FUNCTION_QUEUE_GET_INFO = 91,                                           ///< Enumerator for ::urQueueGetInfo
    UR_FUNCTION_QUEUE_CREATE = 92,                                             ///< Enumerator for ::urQueueCreate
    UR_FUNCTION_QUEUE_RETAIN = 93,                                             ///< Enumerator for ::urQueueRetain
    UR_FUNCTION_QUEUE_RELEASE = 94,                                            ///< Enumerator for ::urQueueRelease
    UR_FUNCTION_QUEUE_GET_NATIVE_HANDLE = 95,                                  ///< Enumerator for ::urQueueGetNativeHandle
    UR_FUNCTION_QUEUE_CREATE_WITH_NATIVE_HANDLE = 96,                          ///< Enumerator for ::urQueueCreateWithNativeHandle
    UR_FUNCTION_QUEUE_FINISH = 97,                                             ///< Enumerator for ::urQueueFinish
    UR_FUNCTION_QUEUE_FLUSH = 98,                                              ///< Enumerator for ::urQueueFlush
    UR_FUNCTION_SAMPLER_CREATE = 101,                                          ///< Enumerator for ::urSamplerCreate
    UR_FUNCTION_SAMPLER_RETAIN = 102,                                          ///< Enumerator for ::urSamplerRetain
    UR_FUNCTION_SAMPLER_RELEASE = 103,                                         ///< Enumerator for ::urSamplerRelease
    UR_FUNCTION_SAMPLER_GET_INFO = 104,                                        ///< Enumerator for ::urSamplerGetInfo
    UR_FUNCTION_SAMPLER_GET_NATIVE_HANDLE = 105,                               ///< Enumerator for ::urSamplerGetNativeHandle
    UR_FUNCTION_SAMPLER_CREATE_WITH_NATIVE_HANDLE = 106,                       ///< Enumerator for ::urSamplerCreateWithNativeHandle
    UR_FUNCTION_USM_HOST_ALLOC = 107,                                          ///< Enumerator for ::urUSMHostAlloc
    UR_FUNCTION_USM_DEVICE_ALLOC = 108,                                        ///< Enumerator for ::urUSMDeviceAlloc
    UR_FUNCTION_USM_SHARED_ALLOC = 109,                                        ///< Enumerator for ::urUSMSharedAlloc
    UR_FUNCTION_USM_FREE = 110,                                                ///< Enumerator for ::urUSMFree
    UR_FUNCTION_USM_GET_MEM_ALLOC_INFO = 111,                                  ///< Enumerator for ::urUSMGetMemAllocInfo
    UR_FUNCTION_USM_POOL_CREATE = 112,                                         ///< Enumerator for ::urUSMPoolCreate
    UR_FUNCTION_COMMAND_BUFFER_CREATE_EXP = 113,                               ///< Enumerator for ::urCommandBufferCreateExp
    UR_FUNCTION_PLATFORM_GET_BACKEND_OPTION = 114,                             ///< Enumerator for ::urPlatformGetBackendOption
    UR_FUNCTION_MEM_BUFFER_CREATE_WITH_NATIVE_HANDLE = 115,                    ///< Enumerator for ::urMemBufferCreateWithNativeHandle
    UR_FUNCTION_MEM_IMAGE_CREATE_WITH_NATIVE_HANDLE = 116,                     ///< Enumerator for ::urMemImageCreateWithNativeHandle
    UR_FUNCTION_ENQUEUE_WRITE_HOST_PIPE = 117,                                 ///< Enumerator for ::urEnqueueWriteHostPipe
    UR_FUNCTION_USM_POOL_RETAIN = 118,                                         ///< Enumerator for ::urUSMPoolRetain
    UR_FUNCTION_USM_POOL_RELEASE = 119,                                        ///< Enumerator for ::urUSMPoolRelease
    UR_FUNCTION_USM_POOL_GET_INFO = 120,                                       ///< Enumerator for ::urUSMPoolGetInfo
    UR_FUNCTION_COMMAND_BUFFER_RETAIN_EXP = 121,                               ///< Enumerator for ::urCommandBufferRetainExp
    UR_FUNCTION_COMMAND_BUFFER_RELEASE_EXP = 122,                              ///< Enumerator for ::urCommandBufferReleaseExp
    UR_FUNCTION_COMMAND_BUFFER_FINALIZE_EXP = 123,                             ///< Enumerator for ::urCommandBufferFinalizeExp
    UR_FUNCTION_COMMAND_BUFFER_APPEND_KERNEL_LAUNCH_EXP = 125,                 ///< Enumerator for ::urCommandBufferAppendKernelLaunchExp
    UR_FUNCTION_COMMAND_BUFFER_ENQUEUE_EXP = 128,                              ///< Enumerator for ::urCommandBufferEnqueueExp
    UR_FUNCTION_USM_PITCHED_ALLOC_EXP = 132,                                   ///< Enumerator for ::urUSMPitchedAllocExp
    UR_FUNCTION_BINDLESS_IMAGES_UNSAMPLED_IMAGE_HANDLE_DESTROY_EXP = 133,      ///< Enumerator for ::urBindlessImagesUnsampledImageHandleDestroyExp
    UR_FUNCTION_BINDLESS_IMAGES_SAMPLED_IMAGE_HANDLE_DESTROY_EXP = 134,        ///< Enumerator for ::urBindlessImagesSampledImageHandleDestroyExp
    UR_FUNCTION_BINDLESS_IMAGES_IMAGE_ALLOCATE_EXP = 135,                      ///< Enumerator for ::urBindlessImagesImageAllocateExp
    UR_FUNCTION_BINDLESS_IMAGES_IMAGE_FREE_EXP = 136,                          ///< Enumerator for ::urBindlessImagesImageFreeExp
    UR_FUNCTION_BINDLESS_IMAGES_UNSAMPLED_IMAGE_CREATE_EXP = 137,              ///< Enumerator for ::urBindlessImagesUnsampledImageCreateExp
    UR_FUNCTION_BINDLESS_IMAGES_SAMPLED_IMAGE_CREATE_EXP = 138,                ///< Enumerator for ::urBindlessImagesSampledImageCreateExp
    UR_FUNCTION_BINDLESS_IMAGES_IMAGE_COPY_EXP = 139,                          ///< Enumerator for ::urBindlessImagesImageCopyExp
    UR_FUNCTION_BINDLESS_IMAGES_IMAGE_GET_INFO_EXP = 140,                      ///< Enumerator for ::urBindlessImagesImageGetInfoExp
    UR_FUNCTION_BINDLESS_IMAGES_MIPMAP_GET_LEVEL_EXP = 141,                    ///< Enumerator for ::urBindlessImagesMipmapGetLevelExp
    UR_FUNCTION_BINDLESS_IMAGES_MIPMAP_FREE_EXP = 142,                         ///< Enumerator for ::urBindlessImagesMipmapFreeExp
    UR_FUNCTION_BINDLESS_IMAGES_IMPORT_OPAQUE_FD_EXP = 143,                    ///< Enumerator for ::urBindlessImagesImportOpaqueFDExp
    UR_FUNCTION_BINDLESS_IMAGES_MAP_EXTERNAL_ARRAY_EXP = 144,                  ///< Enumerator for ::urBindlessImagesMapExternalArrayExp
    UR_FUNCTION_BINDLESS_IMAGES_RELEASE_INTEROP_EXP = 145,                     ///< Enumerator for ::urBindlessImagesReleaseInteropExp
    UR_FUNCTION_BINDLESS_IMAGES_IMPORT_EXTERNAL_SEMAPHORE_OPAQUE_FD_EXP = 146, ///< Enumerator for ::urBindlessImagesImportExternalSemaphoreOpaqueFDExp
    UR_FUNCTION_BINDLESS_IMAGES_DESTROY_EXTERNAL_SEMAPHORE_EXP = 147,          ///< Enumerator for ::urBindlessImagesDestroyExternalSemaphoreExp
    UR_FUNCTION_BINDLESS_IMAGES_WAIT_EXTERNAL_SEMAPHORE_EXP = 148,             ///< Enumerator for ::urBindlessImagesWaitExternalSemaphoreExp
    UR_FUNCTION_BINDLESS_IMAGES_SIGNAL_EXTERNAL_SEMAPHORE_EXP = 149,           ///< Enumerator for ::urBindlessImagesSignalExternalSemaphoreExp
    UR_FUNCTION_ENQUEUE_USM_FILL_2D = 151,                                     ///< Enumerator for ::urEnqueueUSMFill2D
    UR_FUNCTION_ENQUEUE_USM_MEMCPY_2D = 152,                                   ///< Enumerator for ::urEnqueueUSMMemcpy2D
    UR_FUNCTION_VIRTUAL_MEM_GRANULARITY_GET_INFO = 153,                        ///< Enumerator for ::urVirtualMemGranularityGetInfo
    UR_FUNCTION_VIRTUAL_MEM_RESERVE = 154,                                     ///< Enumerator for ::urVirtualMemReserve
    UR_FUNCTION_VIRTUAL_MEM_FREE = 155,                                        ///< Enumerator for ::urVirtualMemFree
    UR_FUNCTION_VIRTUAL_MEM_MAP = 156,                                         ///< Enumerator for ::urVirtualMemMap
    UR_FUNCTION_VIRTUAL_MEM_UNMAP = 157,                                       ///< Enumerator for ::urVirtualMemUnmap
    UR_FUNCTION_VIRTUAL_MEM_SET_ACCESS = 158,                                  ///< Enumerator for ::urVirtualMemSetAccess
    UR_FUNCTION_VIRTUAL_MEM_GET_INFO = 159,                                    ///< Enumerator for ::urVirtualMemGetInfo
    UR_FUNCTION_PHYSICAL_MEM_CREATE = 160,                                     ///< Enumerator for ::urPhysicalMemCreate
    UR_FUNCTION_PHYSICAL_MEM_RETAIN = 161,                                     ///< Enumerator for ::urPhysicalMemRetain
    UR_FUNCTION_PHYSICAL_MEM_RELEASE = 162,                                    ///< Enumerator for ::urPhysicalMemRelease
    UR_FUNCTION_USM_IMPORT_EXP = 163,                                          ///< Enumerator for ::urUSMImportExp
    UR_FUNCTION_USM_RELEASE_EXP = 164,                                         ///< Enumerator for ::urUSMReleaseExp
    UR_FUNCTION_USM_P2P_ENABLE_PEER_ACCESS_EXP = 165,                          ///< Enumerator for ::urUsmP2PEnablePeerAccessExp
    UR_FUNCTION_USM_P2P_DISABLE_PEER_ACCESS_EXP = 166,                         ///< Enumerator for ::urUsmP2PDisablePeerAccessExp
    UR_FUNCTION_USM_P2P_PEER_ACCESS_GET_INFO_EXP = 167,                        ///< Enumerator for ::urUsmP2PPeerAccessGetInfoExp
    UR_FUNCTION_LOADER_CONFIG_CREATE = 172,                                    ///< Enumerator for ::urLoaderConfigCreate
    UR_FUNCTION_LOADER_CONFIG_RELEASE = 173,                                   ///< Enumerator for ::urLoaderConfigRelease
    UR_FUNCTION_LOADER_CONFIG_RETAIN = 174,                                    ///< Enumerator for ::urLoaderConfigRetain
    UR_FUNCTION_LOADER_CONFIG_GET_INFO = 175,                                  ///< Enumerator for ::urLoaderConfigGetInfo
    UR_FUNCTION_LOADER_CONFIG_ENABLE_LAYER = 176,                              ///< Enumerator for ::urLoaderConfigEnableLayer
    UR_FUNCTION_ADAPTER_RELEASE = 177,                                         ///< Enumerator for ::urAdapterRelease
    UR_FUNCTION_ADAPTER_GET = 178,                                             ///< Enumerator for ::urAdapterGet
    UR_FUNCTION_ADAPTER_RETAIN = 179,                                          ///< Enumerator for ::urAdapterRetain
    UR_FUNCTION_ADAPTER_GET_LAST_ERROR = 180,                                  ///< Enumerator for ::urAdapterGetLastError
    UR_FUNCTION_ADAPTER_GET_INFO = 181,                                        ///< Enumerator for ::urAdapterGetInfo
    UR_FUNCTION_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_EXP = 182,                 ///< Enumerator for ::urCommandBufferUpdateKernelLaunchExp
    UR_FUNCTION_PROGRAM_BUILD_EXP = 197,                                       ///< Enumerator for ::urProgramBuildExp
    UR_FUNCTION_PROGRAM_COMPILE_EXP = 198,                                     ///< Enumerator for ::urProgramCompileExp
    UR_FUNCTION_PROGRAM_LINK_EXP = 199,                                        ///< Enumerator for ::urProgramLinkExp
    UR_FUNCTION_LOADER_CONFIG_SET_CODE_LOCATION_CALLBACK = 200,                ///< Enumerator for ::urLoaderConfigSetCodeLocationCallback
    UR_FUNCTION_LOADER_INIT = 201,                                             ///< Enumerator for ::urLoaderInit
    UR_FUNCTION_LOADER_TEAR_DOWN = 202,                                        ///< Enumerator for ::urLoaderTearDown
    UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_MEMCPY_EXP = 203,                    ///< Enumerator for ::urCommandBufferAppendUSMMemcpyExp
    UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_FILL_EXP = 204,                      ///< Enumerator for ::urCommandBufferAppendUSMFillExp
    UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_COPY_EXP = 205,               ///< Enumerator for ::urCommandBufferAppendMemBufferCopyExp
    UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_WRITE_EXP = 206,              ///< Enumerator for ::urCommandBufferAppendMemBufferWriteExp
    UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_READ_EXP = 207,               ///< Enumerator for ::urCommandBufferAppendMemBufferReadExp
    UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_COPY_RECT_EXP = 208,          ///< Enumerator for ::urCommandBufferAppendMemBufferCopyRectExp
    UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_WRITE_RECT_EXP = 209,         ///< Enumerator for ::urCommandBufferAppendMemBufferWriteRectExp
    UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_READ_RECT_EXP = 210,          ///< Enumerator for ::urCommandBufferAppendMemBufferReadRectExp
    UR_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_FILL_EXP = 211,               ///< Enumerator for ::urCommandBufferAppendMemBufferFillExp
    UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_PREFETCH_EXP = 212,                  ///< Enumerator for ::urCommandBufferAppendUSMPrefetchExp
    UR_FUNCTION_COMMAND_BUFFER_APPEND_USM_ADVISE_EXP = 213,                    ///< Enumerator for ::urCommandBufferAppendUSMAdviseExp
    UR_FUNCTION_ENQUEUE_COOPERATIVE_KERNEL_LAUNCH_EXP = 214,                   ///< Enumerator for ::urEnqueueCooperativeKernelLaunchExp
    UR_FUNCTION_KERNEL_SUGGEST_MAX_COOPERATIVE_GROUP_COUNT_EXP = 215,          ///< Enumerator for ::urKernelSuggestMaxCooperativeGroupCountExp
    UR_FUNCTION_COMMAND_BUFFER_RETAIN_COMMAND_EXP = 216,                       ///< Enumerator for ::urCommandBufferRetainCommandExp
    UR_FUNCTION_COMMAND_BUFFER_RELEASE_COMMAND_EXP = 217,                      ///< Enumerator for ::urCommandBufferReleaseCommandExp
    UR_FUNCTION_COMMAND_BUFFER_GET_INFO_EXP = 218,                             ///< Enumerator for ::urCommandBufferGetInfoExp
    UR_FUNCTION_COMMAND_BUFFER_COMMAND_GET_INFO_EXP = 219,                     ///< Enumerator for ::urCommandBufferCommandGetInfoExp
    /// @cond
    UR_FUNCTION_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_function_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines structure types
typedef enum ur_structure_type_t {
    UR_STRUCTURE_TYPE_CONTEXT_PROPERTIES = 0,                                ///< ::ur_context_properties_t
    UR_STRUCTURE_TYPE_IMAGE_DESC = 1,                                        ///< ::ur_image_desc_t
    UR_STRUCTURE_TYPE_BUFFER_PROPERTIES = 2,                                 ///< ::ur_buffer_properties_t
    UR_STRUCTURE_TYPE_BUFFER_REGION = 3,                                     ///< ::ur_buffer_region_t
    UR_STRUCTURE_TYPE_BUFFER_CHANNEL_PROPERTIES = 4,                         ///< ::ur_buffer_channel_properties_t
    UR_STRUCTURE_TYPE_BUFFER_ALLOC_LOCATION_PROPERTIES = 5,                  ///< ::ur_buffer_alloc_location_properties_t
    UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES = 6,                                ///< ::ur_program_properties_t
    UR_STRUCTURE_TYPE_USM_DESC = 7,                                          ///< ::ur_usm_desc_t
    UR_STRUCTURE_TYPE_USM_HOST_DESC = 8,                                     ///< ::ur_usm_host_desc_t
    UR_STRUCTURE_TYPE_USM_DEVICE_DESC = 9,                                   ///< ::ur_usm_device_desc_t
    UR_STRUCTURE_TYPE_USM_POOL_DESC = 10,                                    ///< ::ur_usm_pool_desc_t
    UR_STRUCTURE_TYPE_USM_POOL_LIMITS_DESC = 11,                             ///< ::ur_usm_pool_limits_desc_t
    UR_STRUCTURE_TYPE_DEVICE_BINARY = 12,                                    ///< ::ur_device_binary_t
    UR_STRUCTURE_TYPE_SAMPLER_DESC = 13,                                     ///< ::ur_sampler_desc_t
    UR_STRUCTURE_TYPE_QUEUE_PROPERTIES = 14,                                 ///< ::ur_queue_properties_t
    UR_STRUCTURE_TYPE_QUEUE_INDEX_PROPERTIES = 15,                           ///< ::ur_queue_index_properties_t
    UR_STRUCTURE_TYPE_CONTEXT_NATIVE_PROPERTIES = 16,                        ///< ::ur_context_native_properties_t
    UR_STRUCTURE_TYPE_KERNEL_NATIVE_PROPERTIES = 17,                         ///< ::ur_kernel_native_properties_t
    UR_STRUCTURE_TYPE_QUEUE_NATIVE_PROPERTIES = 18,                          ///< ::ur_queue_native_properties_t
    UR_STRUCTURE_TYPE_MEM_NATIVE_PROPERTIES = 19,                            ///< ::ur_mem_native_properties_t
    UR_STRUCTURE_TYPE_EVENT_NATIVE_PROPERTIES = 20,                          ///< ::ur_event_native_properties_t
    UR_STRUCTURE_TYPE_PLATFORM_NATIVE_PROPERTIES = 21,                       ///< ::ur_platform_native_properties_t
    UR_STRUCTURE_TYPE_DEVICE_NATIVE_PROPERTIES = 22,                         ///< ::ur_device_native_properties_t
    UR_STRUCTURE_TYPE_PROGRAM_NATIVE_PROPERTIES = 23,                        ///< ::ur_program_native_properties_t
    UR_STRUCTURE_TYPE_SAMPLER_NATIVE_PROPERTIES = 24,                        ///< ::ur_sampler_native_properties_t
    UR_STRUCTURE_TYPE_QUEUE_NATIVE_DESC = 25,                                ///< ::ur_queue_native_desc_t
    UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES = 26,                      ///< ::ur_device_partition_properties_t
    UR_STRUCTURE_TYPE_KERNEL_ARG_MEM_OBJ_PROPERTIES = 27,                    ///< ::ur_kernel_arg_mem_obj_properties_t
    UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES = 28,                          ///< ::ur_physical_mem_properties_t
    UR_STRUCTURE_TYPE_KERNEL_ARG_POINTER_PROPERTIES = 29,                    ///< ::ur_kernel_arg_pointer_properties_t
    UR_STRUCTURE_TYPE_KERNEL_ARG_SAMPLER_PROPERTIES = 30,                    ///< ::ur_kernel_arg_sampler_properties_t
    UR_STRUCTURE_TYPE_KERNEL_EXEC_INFO_PROPERTIES = 31,                      ///< ::ur_kernel_exec_info_properties_t
    UR_STRUCTURE_TYPE_KERNEL_ARG_VALUE_PROPERTIES = 32,                      ///< ::ur_kernel_arg_value_properties_t
    UR_STRUCTURE_TYPE_KERNEL_ARG_LOCAL_PROPERTIES = 33,                      ///< ::ur_kernel_arg_local_properties_t
    UR_STRUCTURE_TYPE_USM_ALLOC_LOCATION_DESC = 35,                          ///< ::ur_usm_alloc_location_desc_t
    UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC = 0x1000,                      ///< ::ur_exp_command_buffer_desc_t
    UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC = 0x1001, ///< ::ur_exp_command_buffer_update_kernel_launch_desc_t
    UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_MEMOBJ_ARG_DESC = 0x1002,    ///< ::ur_exp_command_buffer_update_memobj_arg_desc_t
    UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC = 0x1003,   ///< ::ur_exp_command_buffer_update_pointer_arg_desc_t
    UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC = 0x1004,     ///< ::ur_exp_command_buffer_update_value_arg_desc_t
    UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_EXEC_INFO_DESC = 0x1005,     ///< ::ur_exp_command_buffer_update_exec_info_desc_t
    UR_STRUCTURE_TYPE_EXP_SAMPLER_MIP_PROPERTIES = 0x2000,                   ///< ::ur_exp_sampler_mip_properties_t
    UR_STRUCTURE_TYPE_EXP_INTEROP_MEM_DESC = 0x2001,                         ///< ::ur_exp_interop_mem_desc_t
    UR_STRUCTURE_TYPE_EXP_INTEROP_SEMAPHORE_DESC = 0x2002,                   ///< ::ur_exp_interop_semaphore_desc_t
    UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR = 0x2003,                          ///< ::ur_exp_file_descriptor_t
    UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE = 0x2004,                             ///< ::ur_exp_win32_handle_t
    UR_STRUCTURE_TYPE_EXP_SAMPLER_ADDR_MODES = 0x2005,                       ///< ::ur_exp_sampler_addr_modes_t
    /// @cond
    UR_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_structure_type_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime API common types
#if !defined(__GNUC__)
#pragma region common
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef UR_MAKE_VERSION
/// @brief Generates generic 'oneAPI' API versions
#define UR_MAKE_VERSION(_major, _minor) ((_major << 16) | (_minor & 0x0000ffff))
#endif // UR_MAKE_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_MAJOR_VERSION
/// @brief Extracts 'oneAPI' API major version
#define UR_MAJOR_VERSION(_ver) (_ver >> 16)
#endif // UR_MAJOR_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_MINOR_VERSION
/// @brief Extracts 'oneAPI' API minor version
#define UR_MINOR_VERSION(_ver) (_ver & 0x0000ffff)
#endif // UR_MINOR_VERSION

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_APICALL
#if defined(_WIN32)
/// @brief Calling convention for all API functions
#define UR_APICALL __cdecl
#else
#define UR_APICALL
#endif // defined(_WIN32)
#endif // UR_APICALL

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_APIEXPORT
#if defined(_WIN32)
/// @brief Microsoft-specific dllexport storage-class attribute
#define UR_APIEXPORT __declspec(dllexport)
#else
#define UR_APIEXPORT
#endif // defined(_WIN32)
#endif // UR_APIEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_DLLEXPORT
#if defined(_WIN32)
/// @brief Microsoft-specific dllexport storage-class attribute
#define UR_DLLEXPORT __declspec(dllexport)
#endif // defined(_WIN32)
#endif // UR_DLLEXPORT

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_DLLEXPORT
#if __GNUC__ >= 4
/// @brief GCC-specific dllexport storage-class attribute
#define UR_DLLEXPORT __attribute__((visibility("default")))
#else
#define UR_DLLEXPORT
#endif // __GNUC__ >= 4
#endif // UR_DLLEXPORT

///////////////////////////////////////////////////////////////////////////////
/// @brief compiler-independent type
typedef uint8_t ur_bool_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a loader config object
typedef struct ur_loader_config_handle_t_ *ur_loader_config_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of an adapter instance
typedef struct ur_adapter_handle_t_ *ur_adapter_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a platform instance
typedef struct ur_platform_handle_t_ *ur_platform_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of platform's device object
typedef struct ur_device_handle_t_ *ur_device_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of context object
typedef struct ur_context_handle_t_ *ur_context_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of event object
typedef struct ur_event_handle_t_ *ur_event_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of Program object
typedef struct ur_program_handle_t_ *ur_program_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of program's Kernel object
typedef struct ur_kernel_handle_t_ *ur_kernel_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a queue object
typedef struct ur_queue_handle_t_ *ur_queue_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a native object
typedef struct ur_native_handle_t_ *ur_native_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a Sampler object
typedef struct ur_sampler_handle_t_ *ur_sampler_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of memory object which can either be buffer or image
typedef struct ur_mem_handle_t_ *ur_mem_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of physical memory object
typedef struct ur_physical_mem_handle_t_ *ur_physical_mem_handle_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_BIT
/// @brief Generic macro for enumerator bit masks
#define UR_BIT(_i) (1 << _i)
#endif // UR_BIT

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines Return/Error codes
typedef enum ur_result_t {
    UR_RESULT_SUCCESS = 0,                                                    ///< Success
    UR_RESULT_ERROR_INVALID_OPERATION = 1,                                    ///< Invalid operation
    UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES = 2,                             ///< Invalid queue properties
    UR_RESULT_ERROR_INVALID_QUEUE = 3,                                        ///< Invalid queue
    UR_RESULT_ERROR_INVALID_VALUE = 4,                                        ///< Invalid Value
    UR_RESULT_ERROR_INVALID_CONTEXT = 5,                                      ///< Invalid context
    UR_RESULT_ERROR_INVALID_PLATFORM = 6,                                     ///< Invalid platform
    UR_RESULT_ERROR_INVALID_BINARY = 7,                                       ///< Invalid binary
    UR_RESULT_ERROR_INVALID_PROGRAM = 8,                                      ///< Invalid program
    UR_RESULT_ERROR_INVALID_SAMPLER = 9,                                      ///< Invalid sampler
    UR_RESULT_ERROR_INVALID_BUFFER_SIZE = 10,                                 ///< Invalid buffer size
    UR_RESULT_ERROR_INVALID_MEM_OBJECT = 11,                                  ///< Invalid memory object
    UR_RESULT_ERROR_INVALID_EVENT = 12,                                       ///< Invalid event
    UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST = 13,                             ///< Returned when the event wait list or the events in the wait list are
                                                                              ///< invalid.
    UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET = 14,                        ///< Misaligned sub buffer offset
    UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE = 15,                             ///< Invalid work group size
    UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE = 16,                              ///< Compiler not available
    UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE = 17,                        ///< Profiling info not available
    UR_RESULT_ERROR_DEVICE_NOT_FOUND = 18,                                    ///< Device not found
    UR_RESULT_ERROR_INVALID_DEVICE = 19,                                      ///< Invalid device
    UR_RESULT_ERROR_DEVICE_LOST = 20,                                         ///< Device hung, reset, was removed, or adapter update occurred
    UR_RESULT_ERROR_DEVICE_REQUIRES_RESET = 21,                               ///< Device requires a reset
    UR_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE = 22,                           ///< Device currently in low power state
    UR_RESULT_ERROR_DEVICE_PARTITION_FAILED = 23,                             ///< Device partitioning failed
    UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT = 24,                      ///< Invalid counts provided with ::UR_DEVICE_PARTITION_BY_COUNTS
    UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE = 25,                              ///< Invalid work item size
    UR_RESULT_ERROR_INVALID_WORK_DIMENSION = 26,                              ///< Invalid work dimension
    UR_RESULT_ERROR_INVALID_KERNEL_ARGS = 27,                                 ///< Invalid kernel args
    UR_RESULT_ERROR_INVALID_KERNEL = 28,                                      ///< Invalid kernel
    UR_RESULT_ERROR_INVALID_KERNEL_NAME = 29,                                 ///< [Validation] kernel name is not found in the program
    UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX = 30,                       ///< [Validation] kernel argument index is not valid for kernel
    UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE = 31,                        ///< [Validation] kernel argument size does not match kernel
    UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE = 32,                      ///< [Validation] value of kernel attribute is not valid for the kernel or
                                                                              ///< device
    UR_RESULT_ERROR_INVALID_IMAGE_SIZE = 33,                                  ///< Invalid image size
    UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR = 34,                     ///< Invalid image format descriptor
    UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED = 35,                          ///< Image format not supported
    UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE = 36,                       ///< Memory object allocation failure
    UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE = 37,                          ///< Program object parameter is invalid.
    UR_RESULT_ERROR_UNINITIALIZED = 38,                                       ///< [Validation] adapter is not initialized or specific entry-point is not
                                                                              ///< implemented
    UR_RESULT_ERROR_OUT_OF_HOST_MEMORY = 39,                                  ///< Insufficient host memory to satisfy call
    UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY = 40,                                ///< Insufficient device memory to satisfy call
    UR_RESULT_ERROR_OUT_OF_RESOURCES = 41,                                    ///< Out of resources
    UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE = 42,                               ///< Error occurred when building program, see build log for details
    UR_RESULT_ERROR_PROGRAM_LINK_FAILURE = 43,                                ///< Error occurred when linking programs, see build log for details
    UR_RESULT_ERROR_UNSUPPORTED_VERSION = 44,                                 ///< [Validation] generic error code for unsupported versions
    UR_RESULT_ERROR_UNSUPPORTED_FEATURE = 45,                                 ///< [Validation] generic error code for unsupported features
    UR_RESULT_ERROR_INVALID_ARGUMENT = 46,                                    ///< [Validation] generic error code for invalid arguments
    UR_RESULT_ERROR_INVALID_NULL_HANDLE = 47,                                 ///< [Validation] handle argument is not valid
    UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE = 48,                                ///< [Validation] object pointed to by handle still in-use by device
    UR_RESULT_ERROR_INVALID_NULL_POINTER = 49,                                ///< [Validation] pointer argument may not be nullptr
    UR_RESULT_ERROR_INVALID_SIZE = 50,                                        ///< [Validation] invalid size or dimensions (e.g., must not be zero, or is
                                                                              ///< out of bounds)
    UR_RESULT_ERROR_UNSUPPORTED_SIZE = 51,                                    ///< [Validation] size argument is not supported by the device (e.g., too
                                                                              ///< large)
    UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT = 52,                               ///< [Validation] alignment argument is not supported by the device (e.g.,
                                                                              ///< too small)
    UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT = 53,                      ///< [Validation] synchronization object in invalid state
    UR_RESULT_ERROR_INVALID_ENUMERATION = 54,                                 ///< [Validation] enumerator argument is not valid
    UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION = 55,                             ///< [Validation] enumerator argument is not supported by the device
    UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT = 56,                            ///< [Validation] image format is not supported by the device
    UR_RESULT_ERROR_INVALID_NATIVE_BINARY = 57,                               ///< [Validation] native binary is not supported by the device
    UR_RESULT_ERROR_INVALID_GLOBAL_NAME = 58,                                 ///< [Validation] global variable is not found in the program
    UR_RESULT_ERROR_INVALID_FUNCTION_NAME = 59,                               ///< [Validation] function name is not found in the program
    UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION = 60,                        ///< [Validation] group size dimension is not valid for the kernel or
                                                                              ///< device
    UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION = 61,                      ///< [Validation] global width dimension is not valid for the kernel or
                                                                              ///< device
    UR_RESULT_ERROR_PROGRAM_UNLINKED = 62,                                    ///< [Validation] compiled program or program with imports needs to be
                                                                              ///< linked before kernels can be created from it.
    UR_RESULT_ERROR_OVERLAPPING_REGIONS = 63,                                 ///< [Validation] copy operations do not support overlapping regions of
                                                                              ///< memory
    UR_RESULT_ERROR_INVALID_HOST_PTR = 64,                                    ///< Invalid host pointer
    UR_RESULT_ERROR_INVALID_USM_SIZE = 65,                                    ///< Invalid USM size
    UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE = 66,                           ///< Objection allocation failure
    UR_RESULT_ERROR_ADAPTER_SPECIFIC = 67,                                    ///< An adapter specific warning/error has been reported and can be
                                                                              ///< retrieved via the urPlatformGetLastError entry point.
    UR_RESULT_ERROR_LAYER_NOT_PRESENT = 68,                                   ///< A requested layer was not found by the loader.
    UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP = 0x1000,                      ///< Invalid Command-Buffer
    UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP = 0x1001,           ///< Sync point is not valid for the command-buffer
    UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP = 0x1002, ///< Sync point wait list is invalid
    UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP = 0x1003,       ///< Handle to command-buffer command is invalid
    UR_RESULT_ERROR_UNKNOWN = 0x7ffffffe,                                     ///< Unknown or internal error
    /// @cond
    UR_RESULT_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_result_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all properties types
typedef struct ur_base_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure

} ur_base_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all descriptor types
typedef struct ur_base_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure

} ur_base_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief 3D offset argument passed to buffer rect operations
typedef struct ur_rect_offset_t {
    uint64_t x; ///< [in] x offset (bytes)
    uint64_t y; ///< [in] y offset (scalar)
    uint64_t z; ///< [in] z offset (scalar)

} ur_rect_offset_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief 3D region argument passed to buffer rect operations
typedef struct ur_rect_region_t {
    uint64_t width;  ///< [in] width (bytes)
    uint64_t height; ///< [in] height (scalar)
    uint64_t depth;  ///< [in] scalar (scalar)

} ur_rect_region_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs for Loader
#if !defined(__GNUC__)
#pragma region loader
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device initialization flags
typedef uint32_t ur_device_init_flags_t;
typedef enum ur_device_init_flag_t {
    UR_DEVICE_INIT_FLAG_GPU = UR_BIT(0),  ///< initialize GPU device adapters.
    UR_DEVICE_INIT_FLAG_CPU = UR_BIT(1),  ///< initialize CPU device adapters.
    UR_DEVICE_INIT_FLAG_FPGA = UR_BIT(2), ///< initialize FPGA device adapters.
    UR_DEVICE_INIT_FLAG_MCA = UR_BIT(3),  ///< initialize MCA device adapters.
    UR_DEVICE_INIT_FLAG_VPU = UR_BIT(4),  ///< initialize VPU device adapters.
    /// @cond
    UR_DEVICE_INIT_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_init_flag_t;
/// @brief Bit Mask for validating ur_device_init_flags_t
#define UR_DEVICE_INIT_FLAGS_MASK 0xffffffe0

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a loader config object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phLoaderConfig`
UR_APIEXPORT ur_result_t UR_APICALL
urLoaderConfigCreate(
    ur_loader_config_handle_t *phLoaderConfig ///< [out] Pointer to handle of loader config object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the loader config object.
///
/// @details
///     - Get a reference to the loader config handle. Increment its reference
///       count
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hLoaderConfig`
UR_APIEXPORT ur_result_t UR_APICALL
urLoaderConfigRetain(
    ur_loader_config_handle_t hLoaderConfig ///< [in] loader config handle to retain
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Release config handle.
///
/// @details
///     - Decrement reference count and destroy the config handle if reference
///       count becomes zero.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hLoaderConfig`
UR_APIEXPORT ur_result_t UR_APICALL
urLoaderConfigRelease(
    ur_loader_config_handle_t hLoaderConfig ///< [in] config handle to release
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported loader info
typedef enum ur_loader_config_info_t {
    UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS = 0, ///< [char[]] Null-terminated, semi-colon separated list of available
                                                ///< layers.
    UR_LOADER_CONFIG_INFO_REFERENCE_COUNT = 1,  ///< [uint32_t] Reference count of the loader config object.
    /// @cond
    UR_LOADER_CONFIG_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_loader_config_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves various information about the loader.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hLoaderConfig`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_LOADER_CONFIG_INFO_REFERENCE_COUNT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the loader.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urLoaderConfigGetInfo(
    ur_loader_config_handle_t hLoaderConfig, ///< [in] handle of the loader config object
    ur_loader_config_info_t propName,        ///< [in] type of the info to retrieve
    size_t propSize,                         ///< [in] the number of bytes pointed to by pPropValue.
    void *pPropValue,                        ///< [out][optional][typename(propName, propSize)] array of bytes holding
                                             ///< the info.
                                             ///< If propSize is not equal to or greater than the real number of bytes
                                             ///< needed to return the info
                                             ///< then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                             ///< pPropValue is not used.
    size_t *pPropSizeRet                     ///< [out][optional] pointer to the actual size in bytes of the queried propName.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enable a layer for the specified loader config.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hLoaderConfig`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pLayerName`
///     - ::UR_RESULT_ERROR_LAYER_NOT_PRESENT
///         + If layer specified with `pLayerName` can't be found by the loader.
UR_APIEXPORT ur_result_t UR_APICALL
urLoaderConfigEnableLayer(
    ur_loader_config_handle_t hLoaderConfig, ///< [in] Handle to config object the layer will be enabled for.
    const char *pLayerName                   ///< [in] Null terminated string containing the name of the layer to
                                             ///< enable.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Code location data
typedef struct ur_code_location_t {
    const char *functionName; ///< [in][out] Function name.
    const char *sourceFile;   ///< [in][out] Source code file.
    uint32_t lineNumber;      ///< [in][out] Source code line number.
    uint32_t columnNumber;    ///< [in][out] Source code column number.

} ur_code_location_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Code location callback with user data.
typedef ur_code_location_t (*ur_code_location_callback_t)(
    void *pUserData ///< [in][out] pointer to data to be passed to callback
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Set a function callback for use by the loader to retrieve code
///        location information.
///
/// @details
///     - The code location callback is optional and provides additional
///       information to the tracing layer about the entry point of the current
///       execution flow.
///     - This functionality can be used to match traced unified runtime
///       function calls with higher-level user calls.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hLoaderConfig`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pfnCodeloc`
UR_APIEXPORT ur_result_t UR_APICALL
urLoaderConfigSetCodeLocationCallback(
    ur_loader_config_handle_t hLoaderConfig, ///< [in] Handle to config object the layer will be enabled for.
    ur_code_location_callback_t pfnCodeloc,  ///< [in] Function pointer to code location callback.
    void *pUserData                          ///< [in][out][optional] pointer to data to be passed to callback.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Initialize the 'oneAPI' loader
///
/// @details
///     - The application must call this function before calling any other
///       function.
///     - If this function is not called then all other functions will return
///       ::UR_RESULT_ERROR_UNINITIALIZED.
///     - Only one instance of the loader will be initialized per process.
///     - The application may call this function multiple times with different
///       flags or environment variables enabled.
///     - The application must call this function after forking new processes.
///       Each forked process must call this function.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe for scenarios
///       where multiple libraries may initialize the loader simultaneously.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_DEVICE_INIT_FLAGS_MASK & device_flags`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urLoaderInit(
    ur_device_init_flags_t device_flags,    ///< [in] device initialization flags.
                                            ///< must be 0 (default) or a combination of ::ur_device_init_flag_t.
    ur_loader_config_handle_t hLoaderConfig ///< [in][optional] Handle of loader config handle.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Tear down the 'oneAPI' loader and release all its resources
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urLoaderTearDown(
    void);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs for Adapter
#if !defined(__GNUC__)
#pragma region adapter
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves all available adapters
///
/// @details
///     - Adapter implementations must return exactly one adapter handle from
///       this entry point.
///     - The loader may return more than one adapter handle when there are
///       multiple available.
///     - Each returned adapter has its reference count incremented and should
///       be released with a subsequent call to ::urAdapterRelease.
///     - Adapters may perform adapter-specific state initialization when the
///       first reference to them is taken.
///     - An application may call this entry point multiple times to acquire
///       multiple references to the adapter handle(s).
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_SIZE
UR_APIEXPORT ur_result_t UR_APICALL
urAdapterGet(
    uint32_t NumEntries,             ///< [in] the number of adapters to be added to phAdapters.
                                     ///< If phAdapters is not NULL, then NumEntries should be greater than
                                     ///< zero, otherwise ::UR_RESULT_ERROR_INVALID_SIZE,
                                     ///< will be returned.
    ur_adapter_handle_t *phAdapters, ///< [out][optional][range(0, NumEntries)] array of handle of adapters.
                                     ///< If NumEntries is less than the number of adapters available, then
                                     ///< ::urAdapterGet shall only retrieve that number of platforms.
    uint32_t *pNumAdapters           ///< [out][optional] returns the total number of adapters available.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Releases the adapter handle reference indicating end of its usage
///
/// @details
///     - When the reference count of the adapter reaches zero, the adapter may
///       perform adapter-specififc resource teardown. Resources must be left in
///       a state where it safe for the adapter to be subsequently reinitialized
///       with ::urAdapterGet
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hAdapter`
UR_APIEXPORT ur_result_t UR_APICALL
urAdapterRelease(
    ur_adapter_handle_t hAdapter ///< [in] Adapter handle to release
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the adapter handle.
///
/// @details
///     - Get a reference to the adapter handle. Increment its reference count
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hAdapter`
UR_APIEXPORT ur_result_t UR_APICALL
urAdapterRetain(
    ur_adapter_handle_t hAdapter ///< [in] Adapter handle to retain
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the last adapter specific error.
///
/// @details
/// To be used after another entry-point has returned
/// ::UR_RESULT_ERROR_ADAPTER_SPECIFIC in order to retrieve a message describing
/// the circumstances of the underlying driver error and the error code
/// returned by the failed driver entry-point.
///
/// * Implementations *must* store the message and error code in thread-local
///   storage prior to returning ::UR_RESULT_ERROR_ADAPTER_SPECIFIC.
///
/// * The message and error code storage is will only be valid if a previously
///   called entry-point returned ::UR_RESULT_ERROR_ADAPTER_SPECIFIC.
///
/// * The memory pointed to by the C string returned in `ppMessage` is owned by
///   the adapter and *must* be null terminated.
///
/// * The application *may* call this function from simultaneous threads.
///
/// * The implementation of this function *should* be lock-free.
///
/// Example usage:
///
/// ```cpp
/// if (::urQueueCreate(hContext, hDevice, nullptr, &hQueue) ==
///         ::UR_RESULT_ERROR_ADAPTER_SPECIFIC) {
///     const char* pMessage;
///     int32_t error;
///     ::urAdapterGetLastError(hAdapter, &pMessage, &error);
/// }
/// ```
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hAdapter`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMessage`
///         + `NULL == pError`
UR_APIEXPORT ur_result_t UR_APICALL
urAdapterGetLastError(
    ur_adapter_handle_t hAdapter, ///< [in] handle of the adapter instance
    const char **ppMessage,       ///< [out] pointer to a C string where the adapter specific error message
                                  ///< will be stored.
    int32_t *pError               ///< [out] pointer to an integer where the adapter specific error code will
                                  ///< be stored.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported adapter info
typedef enum ur_adapter_info_t {
    UR_ADAPTER_INFO_BACKEND = 0,         ///< [::ur_adapter_backend_t] Identifies the native backend supported by
                                         ///< the adapter.
    UR_ADAPTER_INFO_REFERENCE_COUNT = 1, ///< [uint32_t] Reference count of the adapter.
                                         ///< The reference count returned should be considered immediately stale.
                                         ///< It is unsuitable for general use in applications. This feature is
                                         ///< provided for identifying memory leaks.
    /// @cond
    UR_ADAPTER_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_adapter_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves information about the adapter
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hAdapter`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_ADAPTER_INFO_REFERENCE_COUNT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urAdapterGetInfo(
    ur_adapter_handle_t hAdapter, ///< [in] handle of the adapter
    ur_adapter_info_t propName,   ///< [in] type of the info to retrieve
    size_t propSize,              ///< [in] the number of bytes pointed to by pPropValue.
    void *pPropValue,             ///< [out][optional][typename(propName, propSize)] array of bytes holding
                                  ///< the info.
                                  ///< If Size is not equal to or greater to the real number of bytes needed
                                  ///< to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
                                  ///< returned and pPropValue is not used.
    size_t *pPropSizeRet          ///< [out][optional] pointer to the actual number of bytes being queried by pPropValue.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Identifies backend of the adapter
typedef enum ur_adapter_backend_t {
    UR_ADAPTER_BACKEND_UNKNOWN = 0,    ///< The backend is not a recognized one
    UR_ADAPTER_BACKEND_LEVEL_ZERO = 1, ///< The backend is Level Zero
    UR_ADAPTER_BACKEND_OPENCL = 2,     ///< The backend is OpenCL
    UR_ADAPTER_BACKEND_CUDA = 3,       ///< The backend is CUDA
    UR_ADAPTER_BACKEND_HIP = 4,        ///< The backend is HIP
    UR_ADAPTER_BACKEND_NATIVE_CPU = 5, ///< The backend is Native CPU
    /// @cond
    UR_ADAPTER_BACKEND_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_adapter_backend_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs for Platform
#if !defined(__GNUC__)
#pragma region platform
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves all available platforms for the given adapters
///
/// @details
///     - Multiple calls to this function will return identical platforms
///       handles, in the same order.
///     - The application may call this function from simultaneous threads, the
///       implementation must be thread-safe
///
/// @remarks
///   _Analogues_
///     - **clGetPlatformIDs**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phAdapters`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `NumEntries == 0 && phPlatforms != NULL`
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGet(
    ur_adapter_handle_t *phAdapters,   ///< [in][range(0, NumAdapters)] array of adapters to query for platforms.
    uint32_t NumAdapters,              ///< [in] number of adapters pointed to by phAdapters
    uint32_t NumEntries,               ///< [in] the number of platforms to be added to phPlatforms.
                                       ///< If phPlatforms is not NULL, then NumEntries should be greater than
                                       ///< zero, otherwise ::UR_RESULT_ERROR_INVALID_SIZE,
                                       ///< will be returned.
    ur_platform_handle_t *phPlatforms, ///< [out][optional][range(0, NumEntries)] array of handle of platforms.
                                       ///< If NumEntries is less than the number of platforms available, then
                                       ///< ::urPlatformGet shall only retrieve that number of platforms.
    uint32_t *pNumPlatforms            ///< [out][optional] returns the total number of platforms available.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported platform info
typedef enum ur_platform_info_t {
    UR_PLATFORM_INFO_NAME = 1,        ///< [char[]] The string denoting name of the platform. The size of the
                                      ///< info needs to be dynamically queried.
    UR_PLATFORM_INFO_VENDOR_NAME = 2, ///< [char[]] The string denoting name of the vendor of the platform. The
                                      ///< size of the info needs to be dynamically queried.
    UR_PLATFORM_INFO_VERSION = 3,     ///< [char[]] The string denoting the version of the platform. The size of
                                      ///< the info needs to be dynamically queried.
    UR_PLATFORM_INFO_EXTENSIONS = 4,  ///< [char[]] The string denoting extensions supported by the platform. The
                                      ///< size of the info needs to be dynamically queried.
    UR_PLATFORM_INFO_PROFILE = 5,     ///< [char[]] The string denoting profile of the platform. The size of the
                                      ///< info needs to be dynamically queried.
    UR_PLATFORM_INFO_BACKEND = 6,     ///< [::ur_platform_backend_t] The backend of the platform. Identifies the
                                      ///< native backend adapter implementing this platform.
    /// @cond
    UR_PLATFORM_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_platform_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves various information about platform
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clGetPlatformInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PLATFORM_INFO_BACKEND < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_PLATFORM
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetInfo(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform
    ur_platform_info_t propName,    ///< [in] type of the info to retrieve
    size_t propSize,                ///< [in] the number of bytes pointed to by pPlatformInfo.
    void *pPropValue,               ///< [out][optional][typename(propName, propSize)] array of bytes holding
                                    ///< the info.
                                    ///< If Size is not equal to or greater to the real number of bytes needed
                                    ///< to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
                                    ///< returned and pPlatformInfo is not used.
    size_t *pPropSizeRet            ///< [out][optional] pointer to the actual number of bytes being queried by pPlatformInfo.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported API versions
///
/// @details
///     - API versions contain major and minor attributes, use
///       ::UR_MAJOR_VERSION and ::UR_MINOR_VERSION
typedef enum ur_api_version_t {
    UR_API_VERSION_0_6 = UR_MAKE_VERSION(0, 6),     ///< version 0.6
    UR_API_VERSION_0_7 = UR_MAKE_VERSION(0, 7),     ///< version 0.7
    UR_API_VERSION_0_8 = UR_MAKE_VERSION(0, 8),     ///< version 0.8
    UR_API_VERSION_0_9 = UR_MAKE_VERSION(0, 9),     ///< version 0.9
    UR_API_VERSION_CURRENT = UR_MAKE_VERSION(0, 9), ///< latest known version
    /// @cond
    UR_API_VERSION_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_api_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns the API version supported by the specified platform
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pVersion`
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetApiVersion(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform
    ur_api_version_t *pVersion      ///< [out] api version
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Return platform native platform handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability platform extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativePlatform`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetNativeHandle(
    ur_platform_handle_t hPlatform,      ///< [in] handle of the platform.
    ur_native_handle_t *phNativePlatform ///< [out] a pointer to the native handle of the platform.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Native platform creation properties
typedef struct ur_platform_native_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_PLATFORM_NATIVE_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    bool isNativeHandleOwned;  ///< [in] Indicates UR owns the native handle or if it came from an
                               ///< interoperability operation in the application that asked to not
                               ///< transfer the ownership to the unified-runtime.

} ur_platform_native_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime platform object from native platform handle.
///
/// @details
///     - Creates runtime platform handle from native driver platform handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phPlatform`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformCreateWithNativeHandle(
    ur_native_handle_t hNativePlatform,                 ///< [in][nocheck] the native handle of the platform.
    const ur_platform_native_properties_t *pProperties, ///< [in][optional] pointer to native platform properties struct.
    ur_platform_handle_t *phPlatform                    ///< [out] pointer to the handle of the platform object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the platform specific compiler backend option from a generic
///        frontend option.
///
/// @details
///     - The string returned via the ppPlatformOption is a NULL terminated C
///       style string.
///     - The string returned via the ppPlatformOption is thread local.
///     - The memory in the string returned via the ppPlatformOption is owned by
///       the adapter.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pFrontendOption`
///         + `NULL == ppPlatformOption`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + If `pFrontendOption` is not a valid frontend option.
UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGetBackendOption(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform instance.
    const char *pFrontendOption,    ///< [in] string containing the frontend option.
    const char **ppPlatformOption   ///< [out] returns the correct platform specific compiler option based on
                                    ///< the frontend option.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Identifies native backend adapters
typedef enum ur_platform_backend_t {
    UR_PLATFORM_BACKEND_UNKNOWN = 0,    ///< The backend is not a recognized one
    UR_PLATFORM_BACKEND_LEVEL_ZERO = 1, ///< The backend is Level Zero
    UR_PLATFORM_BACKEND_OPENCL = 2,     ///< The backend is OpenCL
    UR_PLATFORM_BACKEND_CUDA = 3,       ///< The backend is CUDA
    UR_PLATFORM_BACKEND_HIP = 4,        ///< The backend is HIP
    UR_PLATFORM_BACKEND_NATIVE_CPU = 5, ///< The backend is Native CPU
    /// @cond
    UR_PLATFORM_BACKEND_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_platform_backend_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs for Device
#if !defined(__GNUC__)
#pragma region device
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef UR_DEVICE_BINARY_TARGET_UNKNOWN
/// @brief Target identification strings for
///        ::ur_device_binary_t.pDeviceTargetSpec
///        A device type represented by a particular target triple requires
///        specific
///        binary images. We need to map the image type onto the device target triple
#define UR_DEVICE_BINARY_TARGET_UNKNOWN "<unknown>"
#endif // UR_DEVICE_BINARY_TARGET_UNKNOWN

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_DEVICE_BINARY_TARGET_SPIRV32
/// @brief SPIR-V 32-bit image <-> "spir", 32-bit OpenCL device
#define UR_DEVICE_BINARY_TARGET_SPIRV32 "spir"
#endif // UR_DEVICE_BINARY_TARGET_SPIRV32

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_DEVICE_BINARY_TARGET_SPIRV64
/// @brief SPIR-V 64-bit image <-> "spir64", 64-bit OpenCL device
#define UR_DEVICE_BINARY_TARGET_SPIRV64 "spir64"
#endif // UR_DEVICE_BINARY_TARGET_SPIRV64

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_DEVICE_BINARY_TARGET_SPIRV64_X86_64
/// @brief Device-specific binary images produced from SPIR-V 64-bit <-> various
///        "spir64_*" triples for specific 64-bit OpenCL CPU devices
#define UR_DEVICE_BINARY_TARGET_SPIRV64_X86_64 "spir64_x86_64"
#endif // UR_DEVICE_BINARY_TARGET_SPIRV64_X86_64

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_DEVICE_BINARY_TARGET_SPIRV64_GEN
/// @brief Generic GPU device (64-bit OpenCL)
#define UR_DEVICE_BINARY_TARGET_SPIRV64_GEN "spir64_gen"
#endif // UR_DEVICE_BINARY_TARGET_SPIRV64_GEN

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_DEVICE_BINARY_TARGET_SPIRV64_FPGA
/// @brief 64-bit OpenCL FPGA device
#define UR_DEVICE_BINARY_TARGET_SPIRV64_FPGA "spir64_fpga"
#endif // UR_DEVICE_BINARY_TARGET_SPIRV64_FPGA

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_DEVICE_BINARY_TARGET_NVPTX64
/// @brief PTX 64-bit image <-> "nvptx64", 64-bit NVIDIA PTX device
#define UR_DEVICE_BINARY_TARGET_NVPTX64 "nvptx64"
#endif // UR_DEVICE_BINARY_TARGET_NVPTX64

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_DEVICE_BINARY_TARGET_AMDGCN
/// @brief AMD GCN
#define UR_DEVICE_BINARY_TARGET_AMDGCN "amdgcn"
#endif // UR_DEVICE_BINARY_TARGET_AMDGCN

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_DEVICE_BINARY_TARGET_NATIVE_CPU
/// @brief Native CPU
#define UR_DEVICE_BINARY_TARGET_NATIVE_CPU "native_cpu"
#endif // UR_DEVICE_BINARY_TARGET_NATIVE_CPU

///////////////////////////////////////////////////////////////////////////////
/// @brief Device Binary Type
typedef struct ur_device_binary_t {
    ur_structure_type_t stype;     ///< [in] type of this structure, must be ::UR_STRUCTURE_TYPE_DEVICE_BINARY
    const void *pNext;             ///< [in][optional] pointer to extension-specific structure
    const char *pDeviceTargetSpec; ///< [in] null-terminated string representation of the device's target architecture.
                                   ///< For example:
                                   ///< + ::UR_DEVICE_BINARY_TARGET_UNKNOWN
                                   ///< + ::UR_DEVICE_BINARY_TARGET_SPIRV32
                                   ///< + ::UR_DEVICE_BINARY_TARGET_SPIRV64
                                   ///< + ::UR_DEVICE_BINARY_TARGET_SPIRV64_X86_64
                                   ///< + ::UR_DEVICE_BINARY_TARGET_SPIRV64_GEN
                                   ///< + ::UR_DEVICE_BINARY_TARGET_SPIRV64_FPGA
                                   ///< + ::UR_DEVICE_BINARY_TARGET_NVPTX64
                                   ///< + ::UR_DEVICE_BINARY_TARGET_AMDGCN

} ur_device_binary_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device types
typedef enum ur_device_type_t {
    UR_DEVICE_TYPE_DEFAULT = 1, ///< The default device type as preferred by the runtime
    UR_DEVICE_TYPE_ALL = 2,     ///< Devices of all types
    UR_DEVICE_TYPE_GPU = 3,     ///< Graphics Processing Unit
    UR_DEVICE_TYPE_CPU = 4,     ///< Central Processing Unit
    UR_DEVICE_TYPE_FPGA = 5,    ///< Field Programmable Gate Array
    UR_DEVICE_TYPE_MCA = 6,     ///< Memory Copy Accelerator
    UR_DEVICE_TYPE_VPU = 7,     ///< Vision Processing Unit
    /// @cond
    UR_DEVICE_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves devices within a platform
///
/// @details
///     - Multiple calls to this function will return identical device handles,
///       in the same order.
///     - The number and order of handles returned from this function can be
///       affected by environment variables that filter devices exposed through
///       API.
///     - The returned devices are taken a reference of and must be released
///       with a subsequent call to ::urDeviceRelease.
///     - The application may call this function from simultaneous threads, the
///       implementation must be thread-safe
///
/// @remarks
///   _Analogues_
///     - **clGetDeviceIDs**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_DEVICE_TYPE_VPU < DeviceType`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `NumEntries == 0 && phDevices != NULL`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NumEntries > 0 && phDevices == NULL`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceGet(
    ur_platform_handle_t hPlatform, ///< [in] handle of the platform instance
    ur_device_type_t DeviceType,    ///< [in] the type of the devices.
    uint32_t NumEntries,            ///< [in] the number of devices to be added to phDevices.
                                    ///< If phDevices is not NULL, then NumEntries should be greater than zero.
                                    ///< Otherwise ::UR_RESULT_ERROR_INVALID_SIZE
                                    ///< will be returned.
    ur_device_handle_t *phDevices,  ///< [out][optional][range(0, NumEntries)] array of handle of devices.
                                    ///< If NumEntries is less than the number of devices available, then
                                    ///< platform shall only retrieve that number of devices.
    uint32_t *pNumDevices           ///< [out][optional] pointer to the number of devices.
                                    ///< pNumDevices will be updated with the total number of devices available.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device info
typedef enum ur_device_info_t {
    UR_DEVICE_INFO_TYPE = 0,                                        ///< [::ur_device_type_t] type of the device
    UR_DEVICE_INFO_VENDOR_ID = 1,                                   ///< [uint32_t] vendor Id of the device
    UR_DEVICE_INFO_DEVICE_ID = 2,                                   ///< [uint32_t] Id of the device
    UR_DEVICE_INFO_MAX_COMPUTE_UNITS = 3,                           ///< [uint32_t] the number of compute units
    UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS = 4,                    ///< [uint32_t] max work item dimensions
    UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES = 5,                         ///< [size_t[]] return an array of max work item sizes
    UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE = 6,                         ///< [size_t] max work group size
    UR_DEVICE_INFO_SINGLE_FP_CONFIG = 7,                            ///< [::ur_device_fp_capability_flags_t] single precision floating point
                                                                    ///< capability
    UR_DEVICE_INFO_HALF_FP_CONFIG = 8,                              ///< [::ur_device_fp_capability_flags_t] half precision floating point
                                                                    ///< capability
    UR_DEVICE_INFO_DOUBLE_FP_CONFIG = 9,                            ///< [::ur_device_fp_capability_flags_t] double precision floating point
                                                                    ///< capability
    UR_DEVICE_INFO_QUEUE_PROPERTIES = 10,                           ///< [::ur_queue_flags_t] command queue properties supported by the device
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR = 11,                ///< [uint32_t] preferred vector width for char
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT = 12,               ///< [uint32_t] preferred vector width for short
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT = 13,                 ///< [uint32_t] preferred vector width for int
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG = 14,                ///< [uint32_t] preferred vector width for long
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT = 15,               ///< [uint32_t] preferred vector width for float
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE = 16,              ///< [uint32_t] preferred vector width for double
    UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF = 17,                ///< [uint32_t] preferred vector width for half float
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR = 18,                   ///< [uint32_t] native vector width for char
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT = 19,                  ///< [uint32_t] native vector width for short
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT = 20,                    ///< [uint32_t] native vector width for int
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG = 21,                   ///< [uint32_t] native vector width for long
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT = 22,                  ///< [uint32_t] native vector width for float
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE = 23,                 ///< [uint32_t] native vector width for double
    UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF = 24,                   ///< [uint32_t] native vector width for half float
    UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY = 25,                        ///< [uint32_t] max clock frequency in MHz
    UR_DEVICE_INFO_MEMORY_CLOCK_RATE = 26,                          ///< [uint32_t] memory clock frequency in MHz
    UR_DEVICE_INFO_ADDRESS_BITS = 27,                               ///< [uint32_t] address bits
    UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE = 28,                         ///< [uint64_t] max memory allocation size
    UR_DEVICE_INFO_IMAGE_SUPPORTED = 29,                            ///< [::ur_bool_t] images are supported
    UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS = 30,                        ///< [uint32_t] max number of image objects arguments of a kernel declared
                                                                    ///< with the read_only qualifier
    UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS = 31,                       ///< [uint32_t] max number of image objects arguments of a kernel declared
                                                                    ///< with the write_only qualifier
    UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS = 32,                  ///< [uint32_t] max number of image objects arguments of a kernel declared
                                                                    ///< with the read_write qualifier
    UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH = 33,                          ///< [size_t] max width of Image2D object
    UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT = 34,                         ///< [size_t] max height of Image2D object
    UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH = 35,                          ///< [size_t] max width of Image3D object
    UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT = 36,                         ///< [size_t] max height of Image3D object
    UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH = 37,                          ///< [size_t] max depth of Image3D object
    UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE = 38,                      ///< [size_t] max image buffer size
    UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE = 39,                       ///< [size_t] max image array size
    UR_DEVICE_INFO_MAX_SAMPLERS = 40,                               ///< [uint32_t] max number of samplers that can be used in a kernel
    UR_DEVICE_INFO_MAX_PARAMETER_SIZE = 41,                         ///< [size_t] max size in bytes of all arguments passed to a kernel
    UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN = 42,                        ///< [uint32_t] memory base address alignment
    UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE = 43,                      ///< [::ur_device_mem_cache_type_t] global memory cache type
    UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE = 44,                  ///< [uint32_t] global memory cache line size in bytes
    UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE = 45,                      ///< [uint64_t] size of global memory cache in bytes
    UR_DEVICE_INFO_GLOBAL_MEM_SIZE = 46,                            ///< [uint64_t] size of global memory in bytes
    UR_DEVICE_INFO_GLOBAL_MEM_FREE = 47,                            ///< [uint64_t] size of global memory which is free in bytes
    UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE = 48,                   ///< [uint64_t] max constant buffer size in bytes
    UR_DEVICE_INFO_MAX_CONSTANT_ARGS = 49,                          ///< [uint32_t] max number of __const declared arguments in a kernel
    UR_DEVICE_INFO_LOCAL_MEM_TYPE = 50,                             ///< [::ur_device_local_mem_type_t] local memory type
    UR_DEVICE_INFO_LOCAL_MEM_SIZE = 51,                             ///< [uint64_t] local memory size in bytes
    UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT = 52,                   ///< [::ur_bool_t] support error correction to global and local memory
    UR_DEVICE_INFO_HOST_UNIFIED_MEMORY = 53,                        ///< [::ur_bool_t] unified host device memory
    UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION = 54,                 ///< [size_t] profiling timer resolution in nanoseconds
    UR_DEVICE_INFO_ENDIAN_LITTLE = 55,                              ///< [::ur_bool_t] little endian byte order
    UR_DEVICE_INFO_AVAILABLE = 56,                                  ///< [::ur_bool_t] device is available
    UR_DEVICE_INFO_COMPILER_AVAILABLE = 57,                         ///< [::ur_bool_t] device compiler is available
    UR_DEVICE_INFO_LINKER_AVAILABLE = 58,                           ///< [::ur_bool_t] device linker is available
    UR_DEVICE_INFO_EXECUTION_CAPABILITIES = 59,                     ///< [::ur_device_exec_capability_flags_t] device kernel execution
                                                                    ///< capability bit-field
    UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES = 60,                 ///< [::ur_queue_flags_t] device command queue property bit-field
    UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES = 61,                   ///< [::ur_queue_flags_t] host queue property bit-field
    UR_DEVICE_INFO_BUILT_IN_KERNELS = 62,                           ///< [char[]] a semi-colon separated list of built-in kernels
    UR_DEVICE_INFO_PLATFORM = 63,                                   ///< [::ur_platform_handle_t] the platform associated with the device
    UR_DEVICE_INFO_REFERENCE_COUNT = 64,                            ///< [uint32_t] Reference count of the device object.
                                                                    ///< The reference count returned should be considered immediately stale.
                                                                    ///< It is unsuitable for general use in applications. This feature is
                                                                    ///< provided for identifying memory leaks.
    UR_DEVICE_INFO_IL_VERSION = 65,                                 ///< [char[]] IL version
    UR_DEVICE_INFO_NAME = 66,                                       ///< [char[]] Device name
    UR_DEVICE_INFO_VENDOR = 67,                                     ///< [char[]] Device vendor
    UR_DEVICE_INFO_DRIVER_VERSION = 68,                             ///< [char[]] Driver version
    UR_DEVICE_INFO_PROFILE = 69,                                    ///< [char[]] Device profile
    UR_DEVICE_INFO_VERSION = 70,                                    ///< [char[]] Device version
    UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION = 71,                    ///< [char[]] Version of backend runtime
    UR_DEVICE_INFO_EXTENSIONS = 72,                                 ///< [char[]] Return a space separated list of extension names
    UR_DEVICE_INFO_PRINTF_BUFFER_SIZE = 73,                         ///< [size_t] Maximum size in bytes of internal printf buffer
    UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC = 74,                ///< [::ur_bool_t] prefer user synchronization when sharing object with
                                                                    ///< other API
    UR_DEVICE_INFO_PARENT_DEVICE = 75,                              ///< [::ur_device_handle_t] return parent device handle
    UR_DEVICE_INFO_SUPPORTED_PARTITIONS = 76,                       ///< [::ur_device_partition_t[]] Returns an array of partition types
                                                                    ///< supported by the device
    UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES = 77,                  ///< [uint32_t] maximum number of sub-devices when the device is
                                                                    ///< partitioned
    UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN = 78,                  ///< [::ur_device_affinity_domain_flags_t] Returns a bit-field of the
                                                                    ///< supported affinity domains for partitioning.
                                                                    ///< If the device does not support any affinity domains, then 0 will be returned.
    UR_DEVICE_INFO_PARTITION_TYPE = 79,                             ///< [::ur_device_partition_property_t[]] return an array of
                                                                    ///< ::ur_device_partition_property_t for properties specified in
                                                                    ///< ::urDevicePartition
    UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS = 80,                         ///< [uint32_t] max number of sub groups
    UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 81,     ///< [::ur_bool_t] support sub group independent forward progress
    UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL = 82,                      ///< [uint32_t[]] return an array of sub group sizes supported on Intel
                                                                    ///< device
    UR_DEVICE_INFO_USM_HOST_SUPPORT = 83,                           ///< [::ur_device_usm_access_capability_flags_t] support USM host memory
                                                                    ///< access
    UR_DEVICE_INFO_USM_DEVICE_SUPPORT = 84,                         ///< [::ur_device_usm_access_capability_flags_t] support USM device memory
                                                                    ///< access
    UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT = 85,                  ///< [::ur_device_usm_access_capability_flags_t] support USM single device
                                                                    ///< shared memory access
    UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT = 86,                   ///< [::ur_device_usm_access_capability_flags_t] support USM cross device
                                                                    ///< shared memory access
    UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT = 87,                  ///< [::ur_device_usm_access_capability_flags_t] support USM system wide
                                                                    ///< shared memory access
    UR_DEVICE_INFO_UUID = 88,                                       ///< [char[]] return device UUID
    UR_DEVICE_INFO_PCI_ADDRESS = 89,                                ///< [char[]] return device PCI address
    UR_DEVICE_INFO_GPU_EU_COUNT = 90,                               ///< [uint32_t] return Intel GPU EU count
    UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH = 91,                          ///< [uint32_t] return Intel GPU EU SIMD width
    UR_DEVICE_INFO_GPU_EU_SLICES = 92,                              ///< [uint32_t] return Intel GPU number of slices
    UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE = 93,                  ///< [uint32_t] return Intel GPU EU count per subslice
    UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE = 94,                    ///< [uint32_t] return Intel GPU number of subslices per slice
    UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU = 95,                      ///< [uint32_t] return Intel GPU number of threads per EU
    UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH = 96,                       ///< [uint32_t] return max memory bandwidth in Mb/s
    UR_DEVICE_INFO_IMAGE_SRGB = 97,                                 ///< [::ur_bool_t] device supports sRGB images
    UR_DEVICE_INFO_BUILD_ON_SUBDEVICE = 98,                         ///< [::ur_bool_t] Return true if sub-device should do its own program
                                                                    ///< build
    UR_DEVICE_INFO_ATOMIC_64 = 99,                                  ///< [::ur_bool_t] support 64 bit atomics
    UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES = 100,          ///< [::ur_memory_order_capability_flags_t] return a bit-field of atomic
                                                                    ///< memory order capabilities
    UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES = 101,          ///< [::ur_memory_scope_capability_flags_t] return a bit-field of atomic
                                                                    ///< memory scope capabilities
    UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES = 102,           ///< [::ur_memory_order_capability_flags_t] return a bit-field of atomic
                                                                    ///< memory fence order capabilities
    UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES = 103,           ///< [::ur_memory_scope_capability_flags_t] return a bit-field of atomic
                                                                    ///< memory fence scope capabilities
    UR_DEVICE_INFO_BFLOAT16 = 104,                                  ///< [::ur_bool_t] support for bfloat16
    UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES = 105,                 ///< [uint32_t] Returns 1 if the device doesn't have a notion of a
                                                                    ///< queue index. Otherwise, returns the number of queue indices that are
                                                                    ///< available for this device.
    UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS = 106,       ///< [::ur_bool_t] support the ::urKernelSetSpecializationConstants entry
                                                                    ///< point
    UR_DEVICE_INFO_MEMORY_BUS_WIDTH = 107,                          ///< [uint32_t] return the width in bits of the memory bus interface of the
                                                                    ///< device.
    UR_DEVICE_INFO_MAX_WORK_GROUPS_3D = 108,                        ///< [size_t[3]] return max 3D work groups
    UR_DEVICE_INFO_ASYNC_BARRIER = 109,                             ///< [::ur_bool_t] return true if Async Barrier is supported
    UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT = 110,                       ///< [::ur_bool_t] return true if specifying memory channels is supported
    UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORTED = 111,            ///< [::ur_bool_t] Return true if the device supports enqueueing commands
                                                                    ///< to read and write pipes from the host.
    UR_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP = 112,              ///< [uint32_t] The maximum number of registers available per block.
    UR_DEVICE_INFO_IP_VERSION = 113,                                ///< [uint32_t] The device IP version. The meaning of the device IP version
                                                                    ///< is implementation-defined, but newer devices should have a higher
                                                                    ///< version than older devices.
    UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT = 114,                    ///< [::ur_bool_t] return true if the device supports virtual memory.
    UR_DEVICE_INFO_ESIMD_SUPPORT = 115,                             ///< [::ur_bool_t] return true if the device supports ESIMD.
    UR_DEVICE_INFO_COMPONENT_DEVICES = 116,                         ///< [::ur_device_handle_t[]] The set of component devices contained by
                                                                    ///< this composite device.
    UR_DEVICE_INFO_COMPOSITE_DEVICE = 117,                          ///< [::ur_device_handle_t] The composite device containing this component
                                                                    ///< device.
    UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP = 0x1000,             ///< [::ur_bool_t] Returns true if the device supports the use of
                                                                    ///< command-buffers.
    UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_SUPPORT_EXP = 0x1001,      ///< [::ur_bool_t] Returns true if the device supports updating the kernel
                                                                    ///< commands in a command-buffer.
    UR_DEVICE_INFO_BINDLESS_IMAGES_SUPPORT_EXP = 0x2000,            ///< [::ur_bool_t] returns true if the device supports the creation of
                                                                    ///< bindless images
    UR_DEVICE_INFO_BINDLESS_IMAGES_SHARED_USM_SUPPORT_EXP = 0x2001, ///< [::ur_bool_t] returns true if the device supports the creation of
                                                                    ///< bindless images backed by shared USM
    UR_DEVICE_INFO_BINDLESS_IMAGES_1D_USM_SUPPORT_EXP = 0x2002,     ///< [::ur_bool_t] returns true if the device supports the creation of 1D
                                                                    ///< bindless images backed by USM
    UR_DEVICE_INFO_BINDLESS_IMAGES_2D_USM_SUPPORT_EXP = 0x2003,     ///< [::ur_bool_t] returns true if the device supports the creation of 2D
                                                                    ///< bindless images backed by USM
    UR_DEVICE_INFO_IMAGE_PITCH_ALIGN_EXP = 0x2004,                  ///< [uint32_t] returns the required alignment of the pitch between two
                                                                    ///< rows of an image in bytes
    UR_DEVICE_INFO_MAX_IMAGE_LINEAR_WIDTH_EXP = 0x2005,             ///< [size_t] returns the maximum linear width allowed for images allocated
                                                                    ///< using USM
    UR_DEVICE_INFO_MAX_IMAGE_LINEAR_HEIGHT_EXP = 0x2006,            ///< [size_t] returns the maximum linear height allowed for images
                                                                    ///< allocated using USM
    UR_DEVICE_INFO_MAX_IMAGE_LINEAR_PITCH_EXP = 0x2007,             ///< [size_t] returns the maximum linear pitch allowed for images allocated
                                                                    ///< using USM
    UR_DEVICE_INFO_MIPMAP_SUPPORT_EXP = 0x2008,                     ///< [::ur_bool_t] returns true if the device supports allocating mipmap
                                                                    ///< resources
    UR_DEVICE_INFO_MIPMAP_ANISOTROPY_SUPPORT_EXP = 0x2009,          ///< [::ur_bool_t] returns true if the device supports sampling mipmap
                                                                    ///< images with anisotropic filtering
    UR_DEVICE_INFO_MIPMAP_MAX_ANISOTROPY_EXP = 0x200A,              ///< [uint32_t] returns the maximum anisotropic ratio supported by the
                                                                    ///< device
    UR_DEVICE_INFO_MIPMAP_LEVEL_REFERENCE_SUPPORT_EXP = 0x200B,     ///< [::ur_bool_t] returns true if the device supports using images created
                                                                    ///< from individual mipmap levels
    UR_DEVICE_INFO_INTEROP_MEMORY_IMPORT_SUPPORT_EXP = 0x200C,      ///< [::ur_bool_t] returns true if the device supports importing external
                                                                    ///< memory resources
    UR_DEVICE_INFO_INTEROP_MEMORY_EXPORT_SUPPORT_EXP = 0x200D,      ///< [::ur_bool_t] returns true if the device supports exporting internal
                                                                    ///< memory resources
    UR_DEVICE_INFO_INTEROP_SEMAPHORE_IMPORT_SUPPORT_EXP = 0x200E,   ///< [::ur_bool_t] returns true if the device supports importing external
                                                                    ///< semaphore resources
    UR_DEVICE_INFO_INTEROP_SEMAPHORE_EXPORT_SUPPORT_EXP = 0x200F,   ///< [::ur_bool_t] returns true if the device supports exporting internal
                                                                    ///< event resources
    /// @cond
    UR_DEVICE_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves various information about device
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clGetDeviceInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_DEVICE_INFO_INTEROP_SEMAPHORE_EXPORT_SUPPORT_EXP < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceGetInfo(
    ur_device_handle_t hDevice, ///< [in] handle of the device instance
    ur_device_info_t propName,  ///< [in] type of the info to retrieve
    size_t propSize,            ///< [in] the number of bytes pointed to by pPropValue.
    void *pPropValue,           ///< [out][optional][typename(propName, propSize)] array of bytes holding
                                ///< the info.
                                ///< If propSize is not equal to or greater than the real number of bytes
                                ///< needed to return the info
                                ///< then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                ///< pPropValue is not used.
    size_t *pPropSizeRet        ///< [out][optional] pointer to the actual size in bytes of the queried propName.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Makes a reference of the device handle indicating it's in use until
///        paired ::urDeviceRelease is called
///
/// @details
///     - Increments the device reference count if `hDevice` is a valid
///       sub-device created by a call to `urDevicePartition`. If `hDevice` is a
///       root level device (e.g. obtained with `urDeviceGet`), the reference
///       count remains unchanged.
///     - It is not valid to use the device handle, which has all of its
///       references released.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clRetainDevice**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceRetain(
    ur_device_handle_t hDevice ///< [in] handle of the device to get a reference of.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Releases the device handle reference indicating end of its usage
///
/// @details
///     - Decrements the device reference count if `hDevice` is a valid
///       sub-device created by a call to `urDevicePartition`. If `hDevice` is a
///       root level device (e.g. obtained with `urDeviceGet`), the reference
///       count remains unchanged.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clReleaseDevice**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceRelease(
    ur_device_handle_t hDevice ///< [in] handle of the device to release.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Device affinity domain
typedef uint32_t ur_device_affinity_domain_flags_t;
typedef enum ur_device_affinity_domain_flag_t {
    UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA = UR_BIT(0),               ///< Split the device into sub devices comprised of compute units that
                                                                   ///< share a NUMA node.
    UR_DEVICE_AFFINITY_DOMAIN_FLAG_L4_CACHE = UR_BIT(1),           ///< Split the device into sub devices comprised of compute units that
                                                                   ///< share a level 4 data cache.
    UR_DEVICE_AFFINITY_DOMAIN_FLAG_L3_CACHE = UR_BIT(2),           ///< Split the device into sub devices comprised of compute units that
                                                                   ///< share a level 3 data cache.
    UR_DEVICE_AFFINITY_DOMAIN_FLAG_L2_CACHE = UR_BIT(3),           ///< Split the device into sub devices comprised of compute units that
                                                                   ///< share a level 2 data cache.
    UR_DEVICE_AFFINITY_DOMAIN_FLAG_L1_CACHE = UR_BIT(4),           ///< Split the device into sub devices comprised of compute units that
                                                                   ///< share a level 1 data cache.
    UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE = UR_BIT(5), ///< Split the device along the next partitionable affinity domain.
                                                                   ///< The implementation shall find the first level along which the device
                                                                   ///< or sub device may be further subdivided in the order:
                                                                   ///< ::UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA,
                                                                   ///< ::UR_DEVICE_AFFINITY_DOMAIN_FLAG_L4_CACHE,
                                                                   ///< ::UR_DEVICE_AFFINITY_DOMAIN_FLAG_L3_CACHE,
                                                                   ///< ::UR_DEVICE_AFFINITY_DOMAIN_FLAG_L2_CACHE,
                                                                   ///< ::UR_DEVICE_AFFINITY_DOMAIN_FLAG_L1_CACHE,
                                                                   ///< and partition the device into sub devices comprised of compute units
                                                                   ///< that share memory subsystems at this level.
    /// @cond
    UR_DEVICE_AFFINITY_DOMAIN_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_affinity_domain_flag_t;
/// @brief Bit Mask for validating ur_device_affinity_domain_flags_t
#define UR_DEVICE_AFFINITY_DOMAIN_FLAGS_MASK 0xffffffc0

///////////////////////////////////////////////////////////////////////////////
/// @brief Partition Properties
typedef enum ur_device_partition_t {
    UR_DEVICE_PARTITION_EQUALLY = 0x1086,            ///< Partition Equally
    UR_DEVICE_PARTITION_BY_COUNTS = 0x1087,          ///< Partition by counts
    UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN = 0x1088, ///< Partition by affinity domain
    UR_DEVICE_PARTITION_BY_CSLICE = 0x1089,          ///< Partition by c-slice
    /// @cond
    UR_DEVICE_PARTITION_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_partition_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device partition value.
typedef union ur_device_partition_value_t {
    uint32_t equally;                                  ///< [in] Number of compute units per sub-device when partitioning with
                                                       ///< ::UR_DEVICE_PARTITION_EQUALLY.
    uint32_t count;                                    ///< [in] Number of compute units in a sub-device when partitioning with
                                                       ///< ::UR_DEVICE_PARTITION_BY_COUNTS.
    ur_device_affinity_domain_flags_t affinity_domain; ///< [in] The affinity domain to partition for when partitioning with
                                                       ///< ::UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN.

} ur_device_partition_value_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device partition property
typedef struct ur_device_partition_property_t {
    ur_device_partition_t type;        ///< [in] The partitioning type to be used.
    ur_device_partition_value_t value; ///< [in][tagged_by(type)] The partitioning value.

} ur_device_partition_property_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device Partition Properties
typedef struct ur_device_partition_properties_t {
    ur_structure_type_t stype;                         ///< [in] type of this structure, must be
                                                       ///< ::UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES
    void *pNext;                                       ///< [in,out][optional] pointer to extension-specific structure
    const ur_device_partition_property_t *pProperties; ///< [in] Pointer to the beginning of the properties array.
    size_t PropCount;                                  ///< [in] The length of properties pointed to by `pProperties`.

} ur_device_partition_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Partition the device into sub-devices
///
/// @details
///     - Repeated calls to this function with the same inputs will produce the
///       same output in the same order.
///     - The function may be called to request a further partitioning of a
///       sub-device into sub-sub-devices, and so on.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clCreateSubDevices**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pProperties`
///         + `NULL == pProperties->pProperties`
///     - ::UR_RESULT_ERROR_DEVICE_PARTITION_FAILED
///     - ::UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT
UR_APIEXPORT ur_result_t UR_APICALL
urDevicePartition(
    ur_device_handle_t hDevice,                          ///< [in] handle of the device to partition.
    const ur_device_partition_properties_t *pProperties, ///< [in] Device partition properties.
    uint32_t NumDevices,                                 ///< [in] the number of sub-devices.
    ur_device_handle_t *phSubDevices,                    ///< [out][optional][range(0, NumDevices)] array of handle of devices.
                                                         ///< If NumDevices is less than the number of sub-devices available, then
                                                         ///< the function shall only retrieve that number of sub-devices.
    uint32_t *pNumDevicesRet                             ///< [out][optional] pointer to the number of sub-devices the device can be
                                                         ///< partitioned into according to the partitioning property.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Selects the most appropriate device binary based on runtime
///        information and the IR characteristics.
///
/// @details
///     - The input binaries are various AOT images, and possibly an IL binary
///       for JIT compilation.
///     - The selected binary will be able to be run on the target device.
///     - If no suitable binary can be found then function returns
///       ${X}_INVALID_BINARY.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pBinaries`
///         + `NULL == pSelectedBinary`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `NumBinaries == 0`
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceSelectBinary(
    ur_device_handle_t hDevice,          ///< [in] handle of the device to select binary for.
    const ur_device_binary_t *pBinaries, ///< [in] the array of binaries to select from.
    uint32_t NumBinaries,                ///< [in] the number of binaries passed in ppBinaries.
                                         ///< Must greater than or equal to zero otherwise
                                         ///< ::UR_RESULT_ERROR_INVALID_VALUE is returned.
    uint32_t *pSelectedBinary            ///< [out] the index of the selected binary in the input array of binaries.
                                         ///< If a suitable binary was not found the function returns ::UR_RESULT_ERROR_INVALID_BINARY.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief FP capabilities
typedef uint32_t ur_device_fp_capability_flags_t;
typedef enum ur_device_fp_capability_flag_t {
    UR_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT = UR_BIT(0), ///< Support correctly rounded divide and sqrt
    UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST = UR_BIT(1),              ///< Support round to nearest
    UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO = UR_BIT(2),                 ///< Support round to zero
    UR_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF = UR_BIT(3),                  ///< Support round to infinity
    UR_DEVICE_FP_CAPABILITY_FLAG_INF_NAN = UR_BIT(4),                       ///< Support INF to NAN
    UR_DEVICE_FP_CAPABILITY_FLAG_DENORM = UR_BIT(5),                        ///< Support denorm
    UR_DEVICE_FP_CAPABILITY_FLAG_FMA = UR_BIT(6),                           ///< Support FMA
    UR_DEVICE_FP_CAPABILITY_FLAG_SOFT_FLOAT = UR_BIT(7),                    ///< Basic floating point operations implemented in software.
    /// @cond
    UR_DEVICE_FP_CAPABILITY_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_fp_capability_flag_t;
/// @brief Bit Mask for validating ur_device_fp_capability_flags_t
#define UR_DEVICE_FP_CAPABILITY_FLAGS_MASK 0xffffff00

///////////////////////////////////////////////////////////////////////////////
/// @brief Device memory cache type
typedef enum ur_device_mem_cache_type_t {
    UR_DEVICE_MEM_CACHE_TYPE_NONE = 0,             ///< Has none cache
    UR_DEVICE_MEM_CACHE_TYPE_READ_ONLY_CACHE = 1,  ///< Has read only cache
    UR_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE = 2, ///< Has read write cache
    /// @cond
    UR_DEVICE_MEM_CACHE_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_mem_cache_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device local memory type
typedef enum ur_device_local_mem_type_t {
    UR_DEVICE_LOCAL_MEM_TYPE_NONE = 0,   ///< No local memory support
    UR_DEVICE_LOCAL_MEM_TYPE_LOCAL = 1,  ///< Dedicated local memory
    UR_DEVICE_LOCAL_MEM_TYPE_GLOBAL = 2, ///< Global memory
    /// @cond
    UR_DEVICE_LOCAL_MEM_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_local_mem_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device kernel execution capability
typedef uint32_t ur_device_exec_capability_flags_t;
typedef enum ur_device_exec_capability_flag_t {
    UR_DEVICE_EXEC_CAPABILITY_FLAG_KERNEL = UR_BIT(0),        ///< Support kernel execution
    UR_DEVICE_EXEC_CAPABILITY_FLAG_NATIVE_KERNEL = UR_BIT(1), ///< Support native kernel execution
    /// @cond
    UR_DEVICE_EXEC_CAPABILITY_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_exec_capability_flag_t;
/// @brief Bit Mask for validating ur_device_exec_capability_flags_t
#define UR_DEVICE_EXEC_CAPABILITY_FLAGS_MASK 0xfffffffc

///////////////////////////////////////////////////////////////////////////////
/// @brief Return platform native device handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability platform extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeDevice`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceGetNativeHandle(
    ur_device_handle_t hDevice,        ///< [in] handle of the device.
    ur_native_handle_t *phNativeDevice ///< [out] a pointer to the native handle of the device.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Native device creation properties
typedef struct ur_device_native_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_DEVICE_NATIVE_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    bool isNativeHandleOwned;  ///< [in] Indicates UR owns the native handle or if it came from an
                               ///< interoperability operation in the application that asked to not
                               ///< transfer the ownership to the unified-runtime.

} ur_device_native_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime device object from native device handle.
///
/// @details
///     - Creates runtime device handle from native driver device handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPlatform`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevice`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceCreateWithNativeHandle(
    ur_native_handle_t hNativeDevice,                 ///< [in][nocheck] the native handle of the device.
    ur_platform_handle_t hPlatform,                   ///< [in] handle of the platform instance
    const ur_device_native_properties_t *pProperties, ///< [in][optional] pointer to native device properties struct.
    ur_device_handle_t *phDevice                      ///< [out] pointer to the handle of the device object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns synchronized Host and Device global timestamps.
///
/// @details
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clGetDeviceAndHostTimer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceGetGlobalTimestamps(
    ur_device_handle_t hDevice, ///< [in] handle of the device instance
    uint64_t *pDeviceTimestamp, ///< [out][optional] pointer to the Device's global timestamp that
                                ///< correlates with the Host's global timestamp value
    uint64_t *pHostTimestamp    ///< [out][optional] pointer to the Host's global timestamp that
                                ///< correlates with the Device's global timestamp value
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory order capabilities
typedef uint32_t ur_memory_order_capability_flags_t;
typedef enum ur_memory_order_capability_flag_t {
    UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED = UR_BIT(0), ///< Relaxed memory ordering
    UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE = UR_BIT(1), ///< Acquire memory ordering
    UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE = UR_BIT(2), ///< Release memory ordering
    UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL = UR_BIT(3), ///< Acquire/release memory ordering
    UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST = UR_BIT(4), ///< Sequentially consistent memory ordering
    /// @cond
    UR_MEMORY_ORDER_CAPABILITY_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_memory_order_capability_flag_t;
/// @brief Bit Mask for validating ur_memory_order_capability_flags_t
#define UR_MEMORY_ORDER_CAPABILITY_FLAGS_MASK 0xffffffe0

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory scope capabilities
typedef uint32_t ur_memory_scope_capability_flags_t;
typedef enum ur_memory_scope_capability_flag_t {
    UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_ITEM = UR_BIT(0),  ///< Work item scope
    UR_MEMORY_SCOPE_CAPABILITY_FLAG_SUB_GROUP = UR_BIT(1),  ///< Sub group scope
    UR_MEMORY_SCOPE_CAPABILITY_FLAG_WORK_GROUP = UR_BIT(2), ///< Work group scope
    UR_MEMORY_SCOPE_CAPABILITY_FLAG_DEVICE = UR_BIT(3),     ///< Device scope
    UR_MEMORY_SCOPE_CAPABILITY_FLAG_SYSTEM = UR_BIT(4),     ///< System scope
    /// @cond
    UR_MEMORY_SCOPE_CAPABILITY_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_memory_scope_capability_flag_t;
/// @brief Bit Mask for validating ur_memory_scope_capability_flags_t
#define UR_MEMORY_SCOPE_CAPABILITY_FLAGS_MASK 0xffffffe0

///////////////////////////////////////////////////////////////////////////////
/// @brief USM access capabilities
typedef uint32_t ur_device_usm_access_capability_flags_t;
typedef enum ur_device_usm_access_capability_flag_t {
    UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS = UR_BIT(0),                   ///< Memory can be accessed
    UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_ACCESS = UR_BIT(1),            ///< Memory can be accessed atomically
    UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_CONCURRENT_ACCESS = UR_BIT(2),        ///< Memory can be accessed concurrently
    UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ATOMIC_CONCURRENT_ACCESS = UR_BIT(3), ///< Memory can be accessed atomically and concurrently
    /// @cond
    UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_device_usm_access_capability_flag_t;
/// @brief Bit Mask for validating ur_device_usm_access_capability_flags_t
#define UR_DEVICE_USM_ACCESS_CAPABILITY_FLAGS_MASK 0xfffffff0

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs for Context
#if !defined(__GNUC__)
#pragma region context
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Context property type
typedef uint32_t ur_context_flags_t;
typedef enum ur_context_flag_t {
    UR_CONTEXT_FLAG_TBD = UR_BIT(0), ///< reserved for future use
    /// @cond
    UR_CONTEXT_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_context_flag_t;
/// @brief Bit Mask for validating ur_context_flags_t
#define UR_CONTEXT_FLAGS_MASK 0xfffffffe

///////////////////////////////////////////////////////////////////////////////
/// @brief Context creation properties
typedef struct ur_context_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_CONTEXT_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    ur_context_flags_t flags;  ///< [in] context creation flags.

} ur_context_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a context with the given devices.
///
/// @details
///     - All devices should be from the same platform.
///     - Context is used for resource sharing between all the devices
///       associated with it.
///     - Context also serves for resource isolation such that resources do not
///       cross context boundaries.
///     - The returned context is a reference and must be released with a
///       subsequent call to ::urContextRelease.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clCreateContext**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevices`
///         + `NULL == phContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pProperties && ::UR_CONTEXT_FLAGS_MASK & pProperties->flags`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urContextCreate(
    uint32_t DeviceCount,                       ///< [in] the number of devices given in phDevices
    const ur_device_handle_t *phDevices,        ///< [in][range(0, DeviceCount)] array of handle of devices.
    const ur_context_properties_t *pProperties, ///< [in][optional] pointer to context creation properties.
    ur_context_handle_t *phContext              ///< [out] pointer to handle of context object created
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Makes a reference of the context handle indicating it's in use until
///        paired ::urContextRelease is called
///
/// @details
///     - It is not valid to use a context handle, which has all of its
///       references released.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clRetainContext**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
UR_APIEXPORT ur_result_t UR_APICALL
urContextRetain(
    ur_context_handle_t hContext ///< [in] handle of the context to get a reference of.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported context info
typedef enum ur_context_info_t {
    UR_CONTEXT_INFO_NUM_DEVICES = 0,                      ///< [uint32_t] The number of the devices in the context
    UR_CONTEXT_INFO_DEVICES = 1,                          ///< [::ur_device_handle_t[]] The array of the device handles in the
                                                          ///< context
    UR_CONTEXT_INFO_REFERENCE_COUNT = 2,                  ///< [uint32_t] Reference count of the context object.
                                                          ///< The reference count returned should be considered immediately stale.
                                                          ///< It is unsuitable for general use in applications. This feature is
                                                          ///< provided for identifying memory leaks.
    UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT = 3,             ///< [::ur_bool_t] to indicate if the ::urEnqueueUSMMemcpy2D entrypoint is
                                                          ///< supported.
    UR_CONTEXT_INFO_USM_FILL2D_SUPPORT = 4,               ///< [::ur_bool_t] to indicate if the ::urEnqueueUSMFill2D entrypoint is
                                                          ///< supported.
    UR_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES = 5, ///< [::ur_memory_order_capability_flags_t] return a bit-field of atomic
                                                          ///< memory order capabilities.
    UR_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES = 6, ///< [::ur_memory_scope_capability_flags_t] return a bit-field of atomic
                                                          ///< memory scope capabilities.
    UR_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES = 7,  ///< [::ur_memory_order_capability_flags_t] return a bit-field of atomic
                                                          ///< memory fence order capabilities.
                                                          ///< Zero is returned if the backend does not support context-level fences.
    UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES = 8,  ///< [::ur_memory_scope_capability_flags_t] return a bit-field of atomic
                                                          ///< memory fence scope capabilities.
                                                          ///< Zero is returned if the backend does not support context-level fences.
    /// @cond
    UR_CONTEXT_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_context_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Releases the context handle reference indicating end of its usage
///
/// @details
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clReleaseContext**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
UR_APIEXPORT ur_result_t UR_APICALL
urContextRelease(
    ur_context_handle_t hContext ///< [in] handle of the context to release.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves various information about context
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clGetContextInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urContextGetInfo(
    ur_context_handle_t hContext, ///< [in] handle of the context
    ur_context_info_t propName,   ///< [in] type of the info to retrieve
    size_t propSize,              ///< [in] the number of bytes of memory pointed to by pPropValue.
    void *pPropValue,             ///< [out][optional][typename(propName, propSize)] array of bytes holding
                                  ///< the info.
                                  ///< if propSize is not equal to or greater than the real number of bytes
                                  ///< needed to return
                                  ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                  ///< pPropValue is not used.
    size_t *pPropSizeRet          ///< [out][optional] pointer to the actual size in bytes of the queried propName.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Return platform native context handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability platform extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeContext`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urContextGetNativeHandle(
    ur_context_handle_t hContext,       ///< [in] handle of the context.
    ur_native_handle_t *phNativeContext ///< [out] a pointer to the native handle of the context.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties for for ::urContextCreateWithNativeHandle.
typedef struct ur_context_native_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_CONTEXT_NATIVE_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    bool isNativeHandleOwned;  ///< [in] Indicates UR owns the native handle or if it came from an interoperability
                               ///< operation in the application that asked to not transfer the ownership to
                               ///< the unified-runtime.

} ur_context_native_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime context object from native context handle.
///
/// @details
///     - Creates runtime context handle from native driver context handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevices`
///         + `NULL == phContext`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urContextCreateWithNativeHandle(
    ur_native_handle_t hNativeContext,                 ///< [in][nocheck] the native handle of the context.
    uint32_t numDevices,                               ///< [in] number of devices associated with the context
    const ur_device_handle_t *phDevices,               ///< [in][range(0, numDevices)] list of devices associated with the context
    const ur_context_native_properties_t *pProperties, ///< [in][optional] pointer to native context properties struct
    ur_context_handle_t *phContext                     ///< [out] pointer to the handle of the context object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Context's extended deleter callback function with user data.
typedef void (*ur_context_extended_deleter_t)(
    void *pUserData ///< [in][out] pointer to data to be passed to callback
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Call extended deleter function as callback.
///
/// @details
///     - Calls extended deleter, a user-defined callback to delete context on
///       some platforms.
///     - This is done for performance reasons.
///     - This API might be called directly by an application instead of a
///       runtime backend.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pfnDeleter`
UR_APIEXPORT ur_result_t UR_APICALL
urContextSetExtendedDeleter(
    ur_context_handle_t hContext,             ///< [in] handle of the context.
    ur_context_extended_deleter_t pfnDeleter, ///< [in] Function pointer to extended deleter.
    void *pUserData                           ///< [in][out][optional] pointer to data to be passed to callback.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs
#if !defined(__GNUC__)
#pragma region memory
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Memory flags
typedef uint32_t ur_mem_flags_t;
typedef enum ur_mem_flag_t {
    UR_MEM_FLAG_READ_WRITE = UR_BIT(0),              ///< The memory object will be read and written by a kernel. This is the
                                                     ///< default
    UR_MEM_FLAG_WRITE_ONLY = UR_BIT(1),              ///< The memory object will be written but not read by a kernel
    UR_MEM_FLAG_READ_ONLY = UR_BIT(2),               ///< The memory object is a read-only inside a kernel
    UR_MEM_FLAG_USE_HOST_POINTER = UR_BIT(3),        ///< Use memory pointed by a host pointer parameter as the storage bits for
                                                     ///< the memory object
    UR_MEM_FLAG_ALLOC_HOST_POINTER = UR_BIT(4),      ///< Allocate memory object from host accessible memory
    UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER = UR_BIT(5), ///< Allocate memory and copy the data from host pointer pointed memory
    /// @cond
    UR_MEM_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_mem_flag_t;
/// @brief Bit Mask for validating ur_mem_flags_t
#define UR_MEM_FLAGS_MASK 0xffffffc0

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory types
typedef enum ur_mem_type_t {
    UR_MEM_TYPE_BUFFER = 0,         ///< Buffer object
    UR_MEM_TYPE_IMAGE2D = 1,        ///< 2D image object
    UR_MEM_TYPE_IMAGE3D = 2,        ///< 3D image object
    UR_MEM_TYPE_IMAGE2D_ARRAY = 3,  ///< 2D image array object
    UR_MEM_TYPE_IMAGE1D = 4,        ///< 1D image object
    UR_MEM_TYPE_IMAGE1D_ARRAY = 5,  ///< 1D image array object
    UR_MEM_TYPE_IMAGE1D_BUFFER = 6, ///< 1D image buffer object
    /// @cond
    UR_MEM_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_mem_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory Information type
typedef enum ur_mem_info_t {
    UR_MEM_INFO_SIZE = 0,    ///< [size_t] actual size of of memory object in bytes
    UR_MEM_INFO_CONTEXT = 1, ///< [::ur_context_handle_t] context in which the memory object was created
    /// @cond
    UR_MEM_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_mem_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image channel order info: number of channels and the channel layout
typedef enum ur_image_channel_order_t {
    UR_IMAGE_CHANNEL_ORDER_A = 0,          ///< channel order A
    UR_IMAGE_CHANNEL_ORDER_R = 1,          ///< channel order R
    UR_IMAGE_CHANNEL_ORDER_RG = 2,         ///< channel order RG
    UR_IMAGE_CHANNEL_ORDER_RA = 3,         ///< channel order RA
    UR_IMAGE_CHANNEL_ORDER_RGB = 4,        ///< channel order RGB
    UR_IMAGE_CHANNEL_ORDER_RGBA = 5,       ///< channel order RGBA
    UR_IMAGE_CHANNEL_ORDER_BGRA = 6,       ///< channel order BGRA
    UR_IMAGE_CHANNEL_ORDER_ARGB = 7,       ///< channel order ARGB
    UR_IMAGE_CHANNEL_ORDER_ABGR = 8,       ///< channel order ABGR
    UR_IMAGE_CHANNEL_ORDER_INTENSITY = 9,  ///< channel order intensity
    UR_IMAGE_CHANNEL_ORDER_LUMINANCE = 10, ///< channel order luminance
    UR_IMAGE_CHANNEL_ORDER_RX = 11,        ///< channel order Rx
    UR_IMAGE_CHANNEL_ORDER_RGX = 12,       ///< channel order RGx
    UR_IMAGE_CHANNEL_ORDER_RGBX = 13,      ///< channel order RGBx
    UR_IMAGE_CHANNEL_ORDER_SRGBA = 14,     ///< channel order sRGBA
    /// @cond
    UR_IMAGE_CHANNEL_ORDER_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_image_channel_order_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image channel type info: describe the size of the channel data type
typedef enum ur_image_channel_type_t {
    UR_IMAGE_CHANNEL_TYPE_SNORM_INT8 = 0,      ///< channel type snorm int8
    UR_IMAGE_CHANNEL_TYPE_SNORM_INT16 = 1,     ///< channel type snorm int16
    UR_IMAGE_CHANNEL_TYPE_UNORM_INT8 = 2,      ///< channel type unorm int8
    UR_IMAGE_CHANNEL_TYPE_UNORM_INT16 = 3,     ///< channel type unorm int16
    UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565 = 4, ///< channel type unorm short 565
    UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555 = 5, ///< channel type unorm short 555
    UR_IMAGE_CHANNEL_TYPE_INT_101010 = 6,      ///< channel type int 101010
    UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8 = 7,     ///< channel type signed int8
    UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16 = 8,    ///< channel type signed int16
    UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32 = 9,    ///< channel type signed int32
    UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8 = 10,  ///< channel type unsigned int8
    UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16 = 11, ///< channel type unsigned int16
    UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32 = 12, ///< channel type unsigned int32
    UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT = 13,     ///< channel type half float
    UR_IMAGE_CHANNEL_TYPE_FLOAT = 14,          ///< channel type float
    /// @cond
    UR_IMAGE_CHANNEL_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_image_channel_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image information types
typedef enum ur_image_info_t {
    UR_IMAGE_INFO_FORMAT = 0,       ///< [::ur_image_format_t] image format
    UR_IMAGE_INFO_ELEMENT_SIZE = 1, ///< [size_t] element size
    UR_IMAGE_INFO_ROW_PITCH = 2,    ///< [size_t] row pitch
    UR_IMAGE_INFO_SLICE_PITCH = 3,  ///< [size_t] slice pitch
    UR_IMAGE_INFO_WIDTH = 4,        ///< [size_t] image width
    UR_IMAGE_INFO_HEIGHT = 5,       ///< [size_t] image height
    UR_IMAGE_INFO_DEPTH = 6,        ///< [size_t] image depth
    /// @cond
    UR_IMAGE_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_image_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image format including channel layout and data type
typedef struct ur_image_format_t {
    ur_image_channel_order_t channelOrder; ///< [in] image channel order
    ur_image_channel_type_t channelType;   ///< [in] image channel type

} ur_image_format_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image descriptor type.
typedef struct ur_image_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be ::UR_STRUCTURE_TYPE_IMAGE_DESC
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    ur_mem_type_t type;        ///< [in][nocheck] memory object type
    size_t width;              ///< [in] image width
    size_t height;             ///< [in] image height
    size_t depth;              ///< [in] image depth
    size_t arraySize;          ///< [in] image array size
    size_t rowPitch;           ///< [in] image row pitch
    size_t slicePitch;         ///< [in] image slice pitch
    uint32_t numMipLevel;      ///< [in] number of MIP levels
    uint32_t numSamples;       ///< [in] number of samples

} ur_image_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create an image object
///
/// @remarks
///   _Analogues_
///     - **clCreateImage**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MEM_FLAGS_MASK & flags`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == phMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///         + `pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///     - ::UR_RESULT_ERROR_INVALID_HOST_PTR
///         + `pHost == NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) != 0`
///         + `pHost != NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urMemImageCreate(
    ur_context_handle_t hContext,          ///< [in] handle of the context object
    ur_mem_flags_t flags,                  ///< [in] allocation and usage information flags
    const ur_image_format_t *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc,     ///< [in] pointer to image description
    void *pHost,                           ///< [in][optional] pointer to the buffer data
    ur_mem_handle_t *phMem                 ///< [out] pointer to handle of image object created
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer creation properties
typedef struct ur_buffer_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_BUFFER_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    void *pHost;               ///< [in][optional] pointer to the buffer data

} ur_buffer_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer memory channel creation properties
///
/// @details
///     - Specify these properties in ::urMemBufferCreate via
///       ::ur_buffer_properties_t as part of a `pNext` chain.
///
/// @remarks
///   _Analogues_
///     - cl_intel_mem_channel_property
typedef struct ur_buffer_channel_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_BUFFER_CHANNEL_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    uint32_t channel;          ///< [in] Identifies the channel/region to which the buffer should be mapped.

} ur_buffer_channel_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer allocation location creation properties
///
/// @details
///     - Specify these properties in ::urMemBufferCreate via
///       ::ur_buffer_properties_t as part of a `pNext` chain.
///
/// @remarks
///   _Analogues_
///     - cl_intel_mem_alloc_buffer_location
typedef struct ur_buffer_alloc_location_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_BUFFER_ALLOC_LOCATION_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    uint32_t location;         ///< [in] Identifies the ID of global memory partition to which the memory
                               ///< should be allocated.

} ur_buffer_alloc_location_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a memory buffer
///
/// @details
///     - See also ::ur_buffer_channel_properties_t.
///     - See also ::ur_buffer_alloc_location_properties_t.
///
/// @remarks
///   _Analogues_
///     - **clCreateBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MEM_FLAGS_MASK & flags`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phBuffer`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_BUFFER_SIZE
///     - ::UR_RESULT_ERROR_INVALID_HOST_PTR
///         + `pProperties == NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) != 0`
///         + `pProperties != NULL && pProperties->pHost == NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) != 0`
///         + `pProperties != NULL && pProperties->pHost != NULL && (flags & (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urMemBufferCreate(
    ur_context_handle_t hContext,              ///< [in] handle of the context object
    ur_mem_flags_t flags,                      ///< [in] allocation and usage information flags
    size_t size,                               ///< [in] size in bytes of the memory object to be allocated
    const ur_buffer_properties_t *pProperties, ///< [in][optional] pointer to buffer creation properties
    ur_mem_handle_t *phBuffer                  ///< [out] pointer to handle of the memory buffer created
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference the memory object. Increment the memory object's
///        reference count
///
/// @details
///     - Useful in library function to retain access to the memory object after
///       the caller released the object
///
/// @remarks
///   _Analogues_
///     - **clRetainMemoryObject**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urMemRetain(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to get access
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the memory object's reference count and delete the object if
///        the reference count becomes zero.
///
/// @remarks
///   _Analogues_
///     - **clReleaseMemoryObject**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urMemRelease(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to release
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer region type, used to describe a sub buffer
typedef struct ur_buffer_region_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be ::UR_STRUCTURE_TYPE_BUFFER_REGION
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    size_t origin;             ///< [in] buffer origin offset
    size_t size;               ///< [in] size of the buffer region

} ur_buffer_region_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer creation type
typedef enum ur_buffer_create_type_t {
    UR_BUFFER_CREATE_TYPE_REGION = 0, ///< buffer create type is region
    /// @cond
    UR_BUFFER_CREATE_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_buffer_create_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a sub buffer representing a region in an existing buffer
///
/// @remarks
///   _Analogues_
///     - **clCreateSubBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MEM_FLAGS_MASK & flags`
///         + `::UR_BUFFER_CREATE_TYPE_REGION < bufferCreateType`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pRegion`
///         + `NULL == phMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_BUFFER_SIZE
///     - ::UR_RESULT_ERROR_INVALID_HOST_PTR
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urMemBufferPartition(
    ur_mem_handle_t hBuffer,                  ///< [in] handle of the buffer object to allocate from
    ur_mem_flags_t flags,                     ///< [in] allocation and usage information flags
    ur_buffer_create_type_t bufferCreateType, ///< [in] buffer creation type
    const ur_buffer_region_t *pRegion,        ///< [in] pointer to buffer create region information
    ur_mem_handle_t *phMem                    ///< [out] pointer to the handle of sub buffer created
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Return platform native mem handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability platform extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMem`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeMem`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urMemGetNativeHandle(
    ur_mem_handle_t hMem,           ///< [in] handle of the mem.
    ur_device_handle_t hDevice,     ///< [in] handle of the device that the native handle will be resident on.
    ur_native_handle_t *phNativeMem ///< [out] a pointer to the native handle of the mem.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Native memory object creation properties
typedef struct ur_mem_native_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_MEM_NATIVE_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    bool isNativeHandleOwned;  ///< [in] Indicates UR owns the native handle or if it came from an
                               ///< interoperability operation in the application that asked to not
                               ///< transfer the ownership to the unified-runtime.

} ur_mem_native_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime buffer memory object from native memory handle.
///
/// @details
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phMem`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urMemBufferCreateWithNativeHandle(
    ur_native_handle_t hNativeMem,                 ///< [in][nocheck] the native handle to the memory.
    ur_context_handle_t hContext,                  ///< [in] handle of the context object.
    const ur_mem_native_properties_t *pProperties, ///< [in][optional] pointer to native memory creation properties.
    ur_mem_handle_t *phMem                         ///< [out] pointer to handle of buffer memory object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime image memory object from native memory handle.
///
/// @details
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == phMem`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urMemImageCreateWithNativeHandle(
    ur_native_handle_t hNativeMem,                 ///< [in][nocheck] the native handle to the memory.
    ur_context_handle_t hContext,                  ///< [in] handle of the context object.
    const ur_image_format_t *pImageFormat,         ///< [in] pointer to image format specification.
    const ur_image_desc_t *pImageDesc,             ///< [in] pointer to image description.
    const ur_mem_native_properties_t *pProperties, ///< [in][optional] pointer to native memory creation properties.
    ur_mem_handle_t *phMem                         ///< [out] pointer to handle of image memory object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve information about a memory object.
///
/// @details
///     - Query information that is common to all memory objects.
///
/// @remarks
///   _Analogues_
///     - **clGetMemObjectInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMemory`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MEM_INFO_CONTEXT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urMemGetInfo(
    ur_mem_handle_t hMemory, ///< [in] handle to the memory object being queried.
    ur_mem_info_t propName,  ///< [in] type of the info to retrieve.
    size_t propSize,         ///< [in] the number of bytes of memory pointed to by pPropValue.
    void *pPropValue,        ///< [out][optional][typename(propName, propSize)] array of bytes holding
                             ///< the info.
                             ///< If propSize is less than the real number of bytes needed to return
                             ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                             ///< pPropValue is not used.
    size_t *pPropSizeRet     ///< [out][optional] pointer to the actual size in bytes of the queried propName.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve information about an image object.
///
/// @details
///     - Query information specific to an image object.
///
/// @remarks
///   _Analogues_
///     - **clGetImageInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hMemory`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_IMAGE_INFO_DEPTH < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urMemImageGetInfo(
    ur_mem_handle_t hMemory,  ///< [in] handle to the image object being queried.
    ur_image_info_t propName, ///< [in] type of image info to retrieve.
    size_t propSize,          ///< [in] the number of bytes of memory pointer to by pPropValue.
    void *pPropValue,         ///< [out][optional][typename(propName, propSize)] array of bytes holding
                              ///< the info.
                              ///< If propSize is less than the real number of bytes needed to return
                              ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                              ///< pPropValue is not used.
    size_t *pPropSizeRet      ///< [out][optional] pointer to the actual size in bytes of the queried propName.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs
#if !defined(__GNUC__)
#pragma region sampler
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Sampler Filter Mode
typedef enum ur_sampler_filter_mode_t {
    UR_SAMPLER_FILTER_MODE_NEAREST = 0, ///< Filter mode nearest.
    UR_SAMPLER_FILTER_MODE_LINEAR = 1,  ///< Filter mode linear.
    /// @cond
    UR_SAMPLER_FILTER_MODE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_sampler_filter_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sampler addressing mode
typedef enum ur_sampler_addressing_mode_t {
    UR_SAMPLER_ADDRESSING_MODE_NONE = 0,            ///< None
    UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE = 1,   ///< Clamp to edge
    UR_SAMPLER_ADDRESSING_MODE_CLAMP = 2,           ///< Clamp
    UR_SAMPLER_ADDRESSING_MODE_REPEAT = 3,          ///< Repeat
    UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT = 4, ///< Mirrored Repeat
    /// @cond
    UR_SAMPLER_ADDRESSING_MODE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_sampler_addressing_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get sample object information
typedef enum ur_sampler_info_t {
    UR_SAMPLER_INFO_REFERENCE_COUNT = 0,   ///< [uint32_t] Reference count of the sampler object.
                                           ///< The reference count returned should be considered immediately stale.
                                           ///< It is unsuitable for general use in applications. This feature is
                                           ///< provided for identifying memory leaks.
    UR_SAMPLER_INFO_CONTEXT = 1,           ///< [::ur_context_handle_t] Sampler context info
    UR_SAMPLER_INFO_NORMALIZED_COORDS = 2, ///< [::ur_bool_t] Sampler normalized coordinate setting
    UR_SAMPLER_INFO_ADDRESSING_MODE = 3,   ///< [::ur_sampler_addressing_mode_t] Sampler addressing mode setting
    UR_SAMPLER_INFO_FILTER_MODE = 4,       ///< [::ur_sampler_filter_mode_t] Sampler filter mode setting
    /// @cond
    UR_SAMPLER_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_sampler_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sampler description.
typedef struct ur_sampler_desc_t {
    ur_structure_type_t stype;                   ///< [in] type of this structure, must be ::UR_STRUCTURE_TYPE_SAMPLER_DESC
    const void *pNext;                           ///< [in][optional] pointer to extension-specific structure
    bool normalizedCoords;                       ///< [in] Specify if image coordinates are normalized (true) or not (false)
    ur_sampler_addressing_mode_t addressingMode; ///< [in] Specify the address mode of the sampler
    ur_sampler_filter_mode_t filterMode;         ///< [in] Specify the filter mode of the sampler

} ur_sampler_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a sampler object in a context
///
/// @details
///     - The props parameter specifies a list of sampler property names and
///       their corresponding values.
///     - The list is terminated with 0. If the list is NULL, default values
///       will be used.
///
/// @remarks
///   _Analogues_
///     - **clCreateSamplerWithProperties**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDesc`
///         + `NULL == phSampler`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT < pDesc->addressingMode`
///         + `::UR_SAMPLER_FILTER_MODE_LINEAR < pDesc->filterMode`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urSamplerCreate(
    ur_context_handle_t hContext,   ///< [in] handle of the context object
    const ur_sampler_desc_t *pDesc, ///< [in] pointer to the sampler description
    ur_sampler_handle_t *phSampler  ///< [out] pointer to handle of sampler object created
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the sampler object handle. Increment its reference
///        count
///
/// @remarks
///   _Analogues_
///     - **clRetainSampler**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_SAMPLER
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRetain(
    ur_sampler_handle_t hSampler ///< [in] handle of the sampler object to get access
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the sampler's reference count and delete the sampler if the
///        reference count becomes zero.
///
/// @remarks
///   _Analogues_
///     - **clReleaseSampler**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_SAMPLER
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urSamplerRelease(
    ur_sampler_handle_t hSampler ///< [in] handle of the sampler object to release
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a sampler object
///
/// @remarks
///   _Analogues_
///     - **clGetSamplerInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_SAMPLER_INFO_FILTER_MODE < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_SAMPLER
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urSamplerGetInfo(
    ur_sampler_handle_t hSampler, ///< [in] handle of the sampler object
    ur_sampler_info_t propName,   ///< [in] name of the sampler property to query
    size_t propSize,              ///< [in] size in bytes of the sampler property value provided
    void *pPropValue,             ///< [out][typename(propName, propSize)][optional] value of the sampler
                                  ///< property
    size_t *pPropSizeRet          ///< [out][optional] size in bytes returned in sampler property value
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Return sampler native sampler handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability sampler extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeSampler`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urSamplerGetNativeHandle(
    ur_sampler_handle_t hSampler,       ///< [in] handle of the sampler.
    ur_native_handle_t *phNativeSampler ///< [out] a pointer to the native handle of the sampler.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Native sampler creation properties
typedef struct ur_sampler_native_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_SAMPLER_NATIVE_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    bool isNativeHandleOwned;  ///< [in] Indicates UR owns the native handle or if it came from an
                               ///< interoperability operation in the application that asked to not
                               ///< transfer the ownership to the unified-runtime.

} ur_sampler_native_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime sampler object from native sampler handle.
///
/// @details
///     - Creates runtime sampler handle from native driver sampler handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phSampler`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urSamplerCreateWithNativeHandle(
    ur_native_handle_t hNativeSampler,                 ///< [in][nocheck] the native handle of the sampler.
    ur_context_handle_t hContext,                      ///< [in] handle of the context object
    const ur_sampler_native_properties_t *pProperties, ///< [in][optional] pointer to native sampler properties struct.
    ur_sampler_handle_t *phSampler                     ///< [out] pointer to the handle of the sampler object created.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs
#if !defined(__GNUC__)
#pragma region usm
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief USM host memory property flags
typedef uint32_t ur_usm_host_mem_flags_t;
typedef enum ur_usm_host_mem_flag_t {
    UR_USM_HOST_MEM_FLAG_INITIAL_PLACEMENT = UR_BIT(0), ///< Optimize shared allocation for first access on the host
    /// @cond
    UR_USM_HOST_MEM_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_usm_host_mem_flag_t;
/// @brief Bit Mask for validating ur_usm_host_mem_flags_t
#define UR_USM_HOST_MEM_FLAGS_MASK 0xfffffffe

///////////////////////////////////////////////////////////////////////////////
/// @brief USM device memory property flags
typedef uint32_t ur_usm_device_mem_flags_t;
typedef enum ur_usm_device_mem_flag_t {
    UR_USM_DEVICE_MEM_FLAG_WRITE_COMBINED = UR_BIT(0),    ///< Memory should be allocated write-combined (WC)
    UR_USM_DEVICE_MEM_FLAG_INITIAL_PLACEMENT = UR_BIT(1), ///< Optimize shared allocation for first access on the device
    UR_USM_DEVICE_MEM_FLAG_DEVICE_READ_ONLY = UR_BIT(2),  ///< Memory is only possibly modified from the host, but read-only in all
                                                          ///< device code
    /// @cond
    UR_USM_DEVICE_MEM_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_usm_device_mem_flag_t;
/// @brief Bit Mask for validating ur_usm_device_mem_flags_t
#define UR_USM_DEVICE_MEM_FLAGS_MASK 0xfffffff8

///////////////////////////////////////////////////////////////////////////////
/// @brief USM memory property flags
typedef uint32_t ur_usm_pool_flags_t;
typedef enum ur_usm_pool_flag_t {
    UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK = UR_BIT(0), ///< All coarse-grain allocations (allocations from the driver) will be
                                                        ///< zero-initialized.
    /// @cond
    UR_USM_POOL_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_usm_pool_flag_t;
/// @brief Bit Mask for validating ur_usm_pool_flags_t
#define UR_USM_POOL_FLAGS_MASK 0xfffffffe

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocation type
typedef enum ur_usm_type_t {
    UR_USM_TYPE_UNKNOWN = 0, ///< Unknown USM type
    UR_USM_TYPE_HOST = 1,    ///< Host USM type
    UR_USM_TYPE_DEVICE = 2,  ///< Device USM type
    UR_USM_TYPE_SHARED = 3,  ///< Shared USM type
    /// @cond
    UR_USM_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_usm_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM memory allocation information type
typedef enum ur_usm_alloc_info_t {
    UR_USM_ALLOC_INFO_TYPE = 0,     ///< [::ur_usm_type_t] Memory allocation type info
    UR_USM_ALLOC_INFO_BASE_PTR = 1, ///< [void *] Memory allocation base pointer info
    UR_USM_ALLOC_INFO_SIZE = 2,     ///< [size_t] Memory allocation size info
    UR_USM_ALLOC_INFO_DEVICE = 3,   ///< [::ur_device_handle_t] Memory allocation device info
    UR_USM_ALLOC_INFO_POOL = 4,     ///< [::ur_usm_pool_handle_t] Memory allocation pool info
    /// @cond
    UR_USM_ALLOC_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_usm_alloc_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM memory advice
typedef uint32_t ur_usm_advice_flags_t;
typedef enum ur_usm_advice_flag_t {
    UR_USM_ADVICE_FLAG_DEFAULT = UR_BIT(0),                        ///< The USM memory advice is default
    UR_USM_ADVICE_FLAG_SET_READ_MOSTLY = UR_BIT(1),                ///< Hint that memory will be read from frequently and written to rarely
    UR_USM_ADVICE_FLAG_CLEAR_READ_MOSTLY = UR_BIT(2),              ///< Removes the affect of ::UR_USM_ADVICE_FLAG_SET_READ_MOSTLY
    UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION = UR_BIT(3),         ///< Hint that the preferred memory location is the specified device
    UR_USM_ADVICE_FLAG_CLEAR_PREFERRED_LOCATION = UR_BIT(4),       ///< Removes the affect of ::UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION
    UR_USM_ADVICE_FLAG_SET_NON_ATOMIC_MOSTLY = UR_BIT(5),          ///< Hint that memory will mostly be accessed non-atomically
    UR_USM_ADVICE_FLAG_CLEAR_NON_ATOMIC_MOSTLY = UR_BIT(6),        ///< Removes the affect of ::UR_USM_ADVICE_FLAG_SET_NON_ATOMIC_MOSTLY
    UR_USM_ADVICE_FLAG_BIAS_CACHED = UR_BIT(7),                    ///< Hint that memory should be cached
    UR_USM_ADVICE_FLAG_BIAS_UNCACHED = UR_BIT(8),                  ///< Hint that memory should be not be cached
    UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE = UR_BIT(9),         ///< Hint that memory will be mostly accessed by the specified device
    UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_DEVICE = UR_BIT(10),      ///< Removes the affect of ::UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE
    UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_HOST = UR_BIT(11),          ///< Hint that memory will be mostly accessed by the host
    UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_HOST = UR_BIT(12),        ///< Removes the affect of ::UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_HOST
    UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION_HOST = UR_BIT(13),   ///< Hint that the preferred memory location is the host
    UR_USM_ADVICE_FLAG_CLEAR_PREFERRED_LOCATION_HOST = UR_BIT(14), ///< Removes the affect of ::UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION_HOST
    UR_USM_ADVICE_FLAG_SET_NON_COHERENT_MEMORY = UR_BIT(15),       ///< Hint that memory coherence will be coarse-grained (up-to-date only at
                                                                   ///< kernel boundaries)
    UR_USM_ADVICE_FLAG_CLEAR_NON_COHERENT_MEMORY = UR_BIT(16),     ///< Removes the effect of ::UR_USM_ADVICE_FLAG_SET_NON_COHERENT_MEMORY
    /// @cond
    UR_USM_ADVICE_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_usm_advice_flag_t;
/// @brief Bit Mask for validating ur_usm_advice_flags_t
#define UR_USM_ADVICE_FLAGS_MASK 0xfffe0000

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of USM pool
typedef struct ur_usm_pool_handle_t_ *ur_usm_pool_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocation descriptor type.
typedef struct ur_usm_desc_t {
    ur_structure_type_t stype;   ///< [in] type of this structure, must be ::UR_STRUCTURE_TYPE_USM_DESC
    const void *pNext;           ///< [in][optional] pointer to extension-specific structure
    ur_usm_advice_flags_t hints; ///< [in] Memory advice hints
    uint32_t align;              ///< [in] alignment of the USM memory object
                                 ///< Must be zero or a power of 2.
                                 ///< Must be equal to or smaller than the size of the largest data type
                                 ///< supported by `hDevice`.

} ur_usm_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM host allocation descriptor type.
///
/// @details
///     - Specify these properties in ::urUSMHostAlloc and ::urUSMSharedAlloc
///       via ::ur_usm_desc_t as part of a `pNext` chain.
typedef struct ur_usm_host_desc_t {
    ur_structure_type_t stype;     ///< [in] type of this structure, must be ::UR_STRUCTURE_TYPE_USM_HOST_DESC
    const void *pNext;             ///< [in][optional] pointer to extension-specific structure
    ur_usm_host_mem_flags_t flags; ///< [in] host memory allocation flags

} ur_usm_host_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM device allocation descriptor type.
///
/// @details
///     - Specify these properties in ::urUSMDeviceAlloc and ::urUSMSharedAlloc
///       via ::ur_usm_desc_t as part of a `pNext` chain.
typedef struct ur_usm_device_desc_t {
    ur_structure_type_t stype;       ///< [in] type of this structure, must be
                                     ///< ::UR_STRUCTURE_TYPE_USM_DEVICE_DESC
    const void *pNext;               ///< [in][optional] pointer to extension-specific structure
    ur_usm_device_mem_flags_t flags; ///< [in] device memory allocation flags.

} ur_usm_device_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocation location desc
///
/// @details
///     - Specify these properties in ::urUSMHostAlloc, ::urUSMDeviceAlloc and
///       ::urUSMSharedAlloc via ::ur_usm_desc_t as part of a `pNext` chain.
///
/// @remarks
///   _Analogues_
///     - cl_intel_mem_alloc_buffer_location
typedef struct ur_usm_alloc_location_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_USM_ALLOC_LOCATION_DESC
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    uint32_t location;         ///< [in] Identifies the ID of global memory partition to which the memory
                               ///< should be allocated.

} ur_usm_alloc_location_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM pool descriptor type
typedef struct ur_usm_pool_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be ::UR_STRUCTURE_TYPE_USM_POOL_DESC
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    ur_usm_pool_flags_t flags; ///< [in] memory allocation flags

} ur_usm_pool_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM pool limits descriptor type
///
/// @details
///     - Specify these properties in ::urUSMPoolCreate via ::ur_usm_pool_desc_t
///       as part of a `pNext` chain.
typedef struct ur_usm_pool_limits_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_USM_POOL_LIMITS_DESC
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    size_t maxPoolableSize;    ///< [in] Allocations up to this limit will be subject to pooling
    size_t minDriverAllocSize; ///< [in] Minimum allocation size that will be requested from the driver

} ur_usm_pool_limits_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate host memory
///
/// @details
///     - This function must support memory pooling.
///     - If pUSMDesc is not NULL and pUSMDesc->pool is not NULL the allocation
///       will be served from a specified memory pool.
///     - Otherwise, the behavior is implementation-defined.
///     - Allocations served from different memory pools must be isolated and
///       must not reside on the same page.
///     - Any flags/hints passed through pUSMDesc only affect the single
///       allocation.
///     - See also ::ur_usm_host_desc_t.
///     - See also ::ur_usm_alloc_location_desc_t.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pUSMDesc && ::UR_USM_ADVICE_FLAGS_MASK & pUSMDesc->hints`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///         + If ::UR_DEVICE_INFO_USM_HOST_SUPPORT is false.
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `pUSMDesc && pUSMDesc->align != 0 && ((pUSMDesc->align & (pUSMDesc->align-1)) != 0)`
///         + If `align` is greater that the size of the largest data type supported by `hDevice`.
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///         + `size == 0`
///         + `size` is greater than ::UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urUSMHostAlloc(
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    const ur_usm_desc_t *pUSMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t pool,     ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t size,                   ///< [in] size in bytes of the USM memory object to be allocated
    void **ppMem                   ///< [out] pointer to USM host memory object
);

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate device memory
///
/// @details
///     - This function must support memory pooling.
///     - If pUSMDesc is not NULL and pUSMDesc->pool is not NULL the allocation
///       will be served from a specified memory pool.
///     - Otherwise, the behavior is implementation-defined.
///     - Allocations served from different memory pools must be isolated and
///       must not reside on the same page.
///     - Any flags/hints passed through pUSMDesc only affect the single
///       allocation.
///     - See also ::ur_usm_device_desc_t.
///     - See also ::ur_usm_alloc_location_desc_t.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pUSMDesc && ::UR_USM_ADVICE_FLAGS_MASK & pUSMDesc->hints`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///         + If ::UR_DEVICE_INFO_USM_HOST_SUPPORT is false.
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `pUSMDesc && pUSMDesc->align != 0 && ((pUSMDesc->align & (pUSMDesc->align-1)) != 0)`
///         + If `align` is greater that the size of the largest data type supported by `hDevice`.
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///         + `size == 0`
///         + `size` is greater than ::UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urUSMDeviceAlloc(
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    ur_device_handle_t hDevice,    ///< [in] handle of the device object
    const ur_usm_desc_t *pUSMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t pool,     ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t size,                   ///< [in] size in bytes of the USM memory object to be allocated
    void **ppMem                   ///< [out] pointer to USM device memory object
);

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate shared memory
///
/// @details
///     - This function must support memory pooling.
///     - If pUSMDesc is not NULL and pUSMDesc->pool is not NULL the allocation
///       will be served from a specified memory pool.
///     - Otherwise, the behavior is implementation-defined.
///     - Allocations served from different memory pools must be isolated and
///       must not reside on the same page.
///     - Any flags/hints passed through pUSMDesc only affect the single
///       allocation.
///     - See also ::ur_usm_host_desc_t.
///     - See also ::ur_usm_device_desc_t.
///     - See also ::ur_usm_alloc_location_desc_t.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pUSMDesc && ::UR_USM_ADVICE_FLAGS_MASK & pUSMDesc->hints`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `pUSMDesc && pUSMDesc->align != 0 && ((pUSMDesc->align & (pUSMDesc->align-1)) != 0)`
///         + If `align` is greater that the size of the largest data type supported by `hDevice`.
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///         + `size == 0`
///         + `size` is greater than ::UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE.
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///         + If `UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT` and `UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT` are both false.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urUSMSharedAlloc(
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    ur_device_handle_t hDevice,    ///< [in] handle of the device object
    const ur_usm_desc_t *pUSMDesc, ///< [in][optional] Pointer to USM memory allocation descriptor.
    ur_usm_pool_handle_t pool,     ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t size,                   ///< [in] size in bytes of the USM memory object to be allocated
    void **ppMem                   ///< [out] pointer to USM shared memory object
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Free the USM memory object
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urUSMFree(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to USM memory object
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get USM memory object allocation information
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_ALLOC_INFO_POOL < propName`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urUSMGetMemAllocInfo(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const void *pMem,             ///< [in] pointer to USM memory object
    ur_usm_alloc_info_t propName, ///< [in] the name of the USM allocation property to query
    size_t propSize,              ///< [in] size in bytes of the USM allocation property value
    void *pPropValue,             ///< [out][optional][typename(propName, propSize)] value of the USM
                                  ///< allocation property
    size_t *pPropSizeRet          ///< [out][optional] bytes returned in USM allocation property
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create USM memory pool with desired properties.
///
/// @details
///     - UR can create multiple instances of the pool depending on allocation
///       requests.
///     - See also ::ur_usm_pool_limits_desc_t.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPoolDesc`
///         + `NULL == ppPool`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_POOL_FLAGS_MASK & pPoolDesc->flags`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolCreate(
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    ur_usm_pool_desc_t *pPoolDesc, ///< [in] pointer to USM pool descriptor. Can be chained with
                                   ///< ::ur_usm_pool_limits_desc_t
    ur_usm_pool_handle_t *ppPool   ///< [out] pointer to USM memory pool
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the pool handle. Increment its reference count
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == pPool`
UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolRetain(
    ur_usm_pool_handle_t pPool ///< [in] pointer to USM memory pool
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the pool's reference count and delete the pool if the
///        reference count becomes zero.
///
/// @details
///     - All allocation belonging to the pool must be freed prior to the the
///       reference count becoming zero.
///     - If the pool is deleted, this function returns all its reserved memory
///       to the driver.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == pPool`
UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolRelease(
    ur_usm_pool_handle_t pPool ///< [in] pointer to USM memory pool
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get USM memory pool information
typedef enum ur_usm_pool_info_t {
    UR_USM_POOL_INFO_REFERENCE_COUNT = 0, ///< [uint32_t] Reference count of the pool object.
                                          ///< The reference count returned should be considered immediately stale.
                                          ///< It is unsuitable for general use in applications. This feature is
                                          ///< provided for identifying memory leaks.
    UR_USM_POOL_INFO_CONTEXT = 1,         ///< [::ur_context_handle_t] USM memory pool context info
    /// @cond
    UR_USM_POOL_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_usm_pool_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a USM memory pool
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPool`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_POOL_INFO_CONTEXT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolGetInfo(
    ur_usm_pool_handle_t hPool,  ///< [in] handle of the USM memory pool
    ur_usm_pool_info_t propName, ///< [in] name of the pool property to query
    size_t propSize,             ///< [in] size in bytes of the pool property value provided
    void *pPropValue,            ///< [out][optional][typename(propName, propSize)] value of the pool
                                 ///< property
    size_t *pPropSizeRet         ///< [out][optional] size in bytes returned in pool property value
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs
#if !defined(__GNUC__)
#pragma region virtual memory
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Virtual memory granularity info
typedef enum ur_virtual_mem_granularity_info_t {
    UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM = 0x30100,     ///< [size_t] size in bytes of the minimum virtual memory granularity.
    UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED = 0x30101, ///< [size_t] size in bytes of the recommended virtual memory granularity.
    /// @cond
    UR_VIRTUAL_MEM_GRANULARITY_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_virtual_mem_granularity_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get information about the minimum and recommended granularity of
///        physical and virtual memory.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemGranularityGetInfo(
    ur_context_handle_t hContext,               ///< [in] handle of the context object.
    ur_device_handle_t hDevice,                 ///< [in][optional] is the device to get the granularity from, if the
                                                ///< device is null then the granularity is suitable for all devices in context.
    ur_virtual_mem_granularity_info_t propName, ///< [in] type of the info to query.
    size_t propSize,                            ///< [in] size in bytes of the memory pointed to by pPropValue.
    void *pPropValue,                           ///< [out][optional][typename(propName, propSize)] array of bytes holding
                                                ///< the info. If propSize is less than the real number of bytes needed to
                                                ///< return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
                                                ///< returned and pPropValue is not used.
    size_t *pPropSizeRet                        ///< [out][optional] pointer to the actual size in bytes of the queried propName."
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Reserve a virtual memory range.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppStart`
UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemReserve(
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    const void *pStart,           ///< [in][optional] pointer to the start of the virtual memory region to
                                  ///< reserve, specifying a null value causes the implementation to select a
                                  ///< start address.
    size_t size,                  ///< [in] size in bytes of the virtual address range to reserve.
    void **ppStart                ///< [out] pointer to the returned address at the start of reserved virtual
                                  ///< memory range.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Free a virtual memory range.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pStart`
UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemFree(
    ur_context_handle_t hContext, ///< [in] handle of the context object.
    const void *pStart,           ///< [in] pointer to the start of the virtual memory range to free.
    size_t size                   ///< [in] size in bytes of the virtual memory range to free.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Virtual memory access mode flags.
typedef uint32_t ur_virtual_mem_access_flags_t;
typedef enum ur_virtual_mem_access_flag_t {
    UR_VIRTUAL_MEM_ACCESS_FLAG_NONE = UR_BIT(0),       ///< Virtual memory has no access.
    UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE = UR_BIT(1), ///< Virtual memory has both read and write access.
    UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY = UR_BIT(2),  ///< Virtual memory has read only access.
    /// @cond
    UR_VIRTUAL_MEM_ACCESS_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_virtual_mem_access_flag_t;
/// @brief Bit Mask for validating ur_virtual_mem_access_flags_t
#define UR_VIRTUAL_MEM_ACCESS_FLAGS_MASK 0xfffffff8

///////////////////////////////////////////////////////////////////////////////
/// @brief Map a virtual memory range to a physical memory handle.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hPhysicalMem`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pStart`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_VIRTUAL_MEM_ACCESS_FLAGS_MASK & flags`
UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemMap(
    ur_context_handle_t hContext,          ///< [in] handle to the context object.
    const void *pStart,                    ///< [in] pointer to the start of the virtual memory range.
    size_t size,                           ///< [in] size in bytes of the virtual memory range to map.
    ur_physical_mem_handle_t hPhysicalMem, ///< [in] handle of the physical memory to map pStart to.
    size_t offset,                         ///< [in] offset in bytes into the physical memory to map pStart to.
    ur_virtual_mem_access_flags_t flags    ///< [in] access flags for the physical memory mapping.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Unmap a virtual memory range previously mapped in a context.
///
/// @details
///     - After a call to this function, the virtual memory range is left in a
///       state ready to be remapped.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pStart`
UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemUnmap(
    ur_context_handle_t hContext, ///< [in] handle to the context object.
    const void *pStart,           ///< [in] pointer to the start of the mapped virtual memory range
    size_t size                   ///< [in] size in bytes of the virtual memory range.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the access mode of a mapped virtual memory range.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pStart`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_VIRTUAL_MEM_ACCESS_FLAGS_MASK & flags`
UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemSetAccess(
    ur_context_handle_t hContext,       ///< [in] handle to the context object.
    const void *pStart,                 ///< [in] pointer to the start of the virtual memory range.
    size_t size,                        ///< [in] size in bytes of the virtual memory range.
    ur_virtual_mem_access_flags_t flags ///< [in] access flags to set for the mapped virtual memory range.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Virtual memory range info queries.
typedef enum ur_virtual_mem_info_t {
    UR_VIRTUAL_MEM_INFO_ACCESS_MODE = 0, ///< [::ur_virtual_mem_access_flags_t] access flags of a mapped virtual
                                         ///< memory range.
    /// @cond
    UR_VIRTUAL_MEM_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_virtual_mem_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get information about a mapped virtual memory range.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pStart`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_VIRTUAL_MEM_INFO_ACCESS_MODE < propName`
UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemGetInfo(
    ur_context_handle_t hContext,   ///< [in] handle to the context object.
    const void *pStart,             ///< [in] pointer to the start of the virtual memory range.
    size_t size,                    ///< [in] size in bytes of the virtual memory range.
    ur_virtual_mem_info_t propName, ///< [in] type of the info to query.
    size_t propSize,                ///< [in] size in bytes of the memory pointed to by pPropValue.
    void *pPropValue,               ///< [out][optional][typename(propName, propSize)] array of bytes holding
                                    ///< the info. If propSize is less than the real number of bytes needed to
                                    ///< return the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is
                                    ///< returned and pPropValue is not used.
    size_t *pPropSizeRet            ///< [out][optional] pointer to the actual size in bytes of the queried propName."
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Physical memory creation properties.
typedef uint32_t ur_physical_mem_flags_t;
typedef enum ur_physical_mem_flag_t {
    UR_PHYSICAL_MEM_FLAG_TBD = UR_BIT(0), ///< reserved for future use.
    /// @cond
    UR_PHYSICAL_MEM_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_physical_mem_flag_t;
/// @brief Bit Mask for validating ur_physical_mem_flags_t
#define UR_PHYSICAL_MEM_FLAGS_MASK 0xfffffffe

///////////////////////////////////////////////////////////////////////////////
/// @brief Physical memory creation properties.
typedef struct ur_physical_mem_properties_t {
    ur_structure_type_t stype;     ///< [in] type of this structure, must be
                                   ///< ::UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES
    void *pNext;                   ///< [in,out][optional] pointer to extension-specific structure
    ur_physical_mem_flags_t flags; ///< [in] physical memory creation flags

} ur_physical_mem_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a physical memory handle that virtual memory can be mapped to.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pProperties && ::UR_PHYSICAL_MEM_FLAGS_MASK & pProperties->flags`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phPhysicalMem`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If size is not a multiple of ::UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM.
UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemCreate(
    ur_context_handle_t hContext,                    ///< [in] handle of the context object.
    ur_device_handle_t hDevice,                      ///< [in] handle of the device object.
    size_t size,                                     ///< [in] size in bytes of physical memory to allocate, must be a multiple
                                                     ///< of ::UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM.
    const ur_physical_mem_properties_t *pProperties, ///< [in][optional] pointer to physical memory creation properties.
    ur_physical_mem_handle_t *phPhysicalMem          ///< [out] pointer to handle of physical memory object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retain a physical memory handle, increment its reference count.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPhysicalMem`
UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRetain(
    ur_physical_mem_handle_t hPhysicalMem ///< [in] handle of the physical memory object to retain.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Release a physical memory handle, decrement its reference count.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hPhysicalMem`
UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRelease(
    ur_physical_mem_handle_t hPhysicalMem ///< [in] handle of the physical memory object to release.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime Runtime APIs for Program
#if !defined(__GNUC__)
#pragma region program
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Program metadata property type.
typedef enum ur_program_metadata_type_t {
    UR_PROGRAM_METADATA_TYPE_UINT32 = 0,     ///< type is a 32-bit integer.
    UR_PROGRAM_METADATA_TYPE_UINT64 = 1,     ///< type is a 64-bit integer.
    UR_PROGRAM_METADATA_TYPE_BYTE_ARRAY = 2, ///< type is a byte array.
    UR_PROGRAM_METADATA_TYPE_STRING = 3,     ///< type is a null-terminated string.
    /// @cond
    UR_PROGRAM_METADATA_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_program_metadata_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Program metadata value union.
typedef union ur_program_metadata_value_t {
    uint32_t data32; ///< [in] inline storage for the 32-bit data, type
                     ///< ::UR_PROGRAM_METADATA_TYPE_UINT32.
    uint64_t data64; ///< [in] inline storage for the 64-bit data, type
                     ///< ::UR_PROGRAM_METADATA_TYPE_UINT64.
    char *pString;   ///< [in] pointer to null-terminated string data, type
                     ///< ::UR_PROGRAM_METADATA_TYPE_STRING.
    void *pData;     ///< [in] pointer to binary data, type
                     ///< ::UR_PROGRAM_METADATA_TYPE_BYTE_ARRAY.

} ur_program_metadata_value_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Program metadata property.
typedef struct ur_program_metadata_t {
    const char *pName;                 ///< [in] null-terminated metadata name.
    ur_program_metadata_type_t type;   ///< [in] the type of metadata value.
    size_t size;                       ///< [in] size in bytes of the data pointed to by value.pData, or 0 when
                                       ///< value size is less than 64-bits and is stored directly in value.data.
    ur_program_metadata_value_t value; ///< [in][tagged_by(type)] the metadata value storage.

} ur_program_metadata_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Program creation properties.
typedef struct ur_program_properties_t {
    ur_structure_type_t stype;               ///< [in] type of this structure, must be
                                             ///< ::UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES
    void *pNext;                             ///< [in,out][optional] pointer to extension-specific structure
    uint32_t count;                          ///< [in] the number of entries in pMetadatas, if count is greater than
                                             ///< zero then pMetadatas must not be null.
    const ur_program_metadata_t *pMetadatas; ///< [in][optional][range(0,count)] pointer to array of metadata entries.

} ur_program_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a program object from input intermediate language.
///
/// @details
///     - The application may call this function from simultaneous threads.
///
/// @remarks
///   _Analogues_
///     - **clCreateProgramWithIL**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pIL`
///         + `NULL == phProgram`
///         + `NULL != pProperties && pProperties->count > 0 && NULL == pProperties->pMetadatas`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `NULL != pProperties && NULL != pProperties->pMetadatas && pProperties->count == 0`
///         + `length == 0`
///     - ::UR_RESULT_ERROR_INVALID_BINARY
///         + If `pIL` is not a valid IL binary for devices in `hContext`.
///     - ::UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE
///         + If devices in `hContext` don't have the capability to compile an IL binary at runtime.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCreateWithIL(
    ur_context_handle_t hContext,               ///< [in] handle of the context instance
    const void *pIL,                            ///< [in] pointer to IL binary.
    size_t length,                              ///< [in] length of `pIL` in bytes.
    const ur_program_properties_t *pProperties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t *phProgram              ///< [out] pointer to handle of program object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a program object from device native binary.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point, `phProgram` will
///       contain a binary of type ::UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT or
///       ::UR_PROGRAM_BINARY_TYPE_LIBRARY for `hDevice`.
///
/// @remarks
///   _Analogues_
///     - **clCreateProgramWithBinary**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pBinary`
///         + `NULL == phProgram`
///         + `NULL != pProperties && pProperties->count > 0 && NULL == pProperties->pMetadatas`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `NULL != pProperties && NULL != pProperties->pMetadatas && pProperties->count == 0`
///     - ::UR_RESULT_ERROR_INVALID_NATIVE_BINARY
///         + If `pBinary` isn't a valid binary for `hDevice.`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCreateWithBinary(
    ur_context_handle_t hContext,               ///< [in] handle of the context instance
    ur_device_handle_t hDevice,                 ///< [in] handle to device associated with binary.
    size_t size,                                ///< [in] size in bytes.
    const uint8_t *pBinary,                     ///< [in] pointer to binary.
    const ur_program_properties_t *pProperties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t *phProgram              ///< [out] pointer to handle of Program object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one program, negates need for the
///        linking step.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point, the program passed
///       will contain a binary of the ::UR_PROGRAM_BINARY_TYPE_EXECUTABLE type
///       for each device in `hContext`.
///
/// @remarks
///   _Analogues_
///     - **clBuildProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If `hProgram` isn't a valid program object.
///     - ::UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE
///         + If an error occurred when building `hProgram`.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramBuild(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    ur_program_handle_t hProgram, ///< [in] Handle of the program to build.
    const char *pOptions          ///< [in][optional] pointer to build options null-terminated string.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one or more programs.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point `hProgram` will
///       contain a binary of the ::UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT type
///       for each device in `hContext`.
///
/// @remarks
///   _Analogues_
///     - **clCompileProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If `hProgram` isn't a valid program object.
///     - ::UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE
///         + If an error occurred while compiling `hProgram`.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCompile(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    ur_program_handle_t hProgram, ///< [in][out] handle of the program to compile.
    const char *pOptions          ///< [in][optional] pointer to build options null-terminated string.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one or more programs.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point the program returned
///       in `phProgram` will contain a binary of the
///       ::UR_PROGRAM_BINARY_TYPE_EXECUTABLE type for each device in
///       `hContext`.
///
/// @remarks
///   _Analogues_
///     - **clLinkProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phPrograms`
///         + `NULL == phProgram`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If one of the programs in `phPrograms` isn't a valid program object.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `count == 0`
///     - ::UR_RESULT_ERROR_PROGRAM_LINK_FAILURE
///         + If an error occurred while linking `phPrograms`.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramLink(
    ur_context_handle_t hContext,          ///< [in] handle of the context instance.
    uint32_t count,                        ///< [in] number of program handles in `phPrograms`.
    const ur_program_handle_t *phPrograms, ///< [in][range(0, count)] pointer to array of program handles.
    const char *pOptions,                  ///< [in][optional] pointer to linker options null-terminated string.
    ur_program_handle_t *phProgram         ///< [out] pointer to handle of program object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the Program object.
///
/// @details
///     - Get a reference to the Program object handle. Increment its reference
///       count
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clRetainProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramRetain(
    ur_program_handle_t hProgram ///< [in] handle for the Program to retain
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Release Program.
///
/// @details
///     - Decrement reference count and destroy the Program if reference count
///       becomes zero.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clReleaseProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramRelease(
    ur_program_handle_t hProgram ///< [in] handle for the Program to release
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves a device function pointer to a user-defined function.
///
/// @details
///     - Retrieves a pointer to the functions with the given name and defined
///       in the given program.
///     - ::UR_RESULT_ERROR_INVALID_FUNCTION_NAME is returned if the function
///       can not be obtained.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @remarks
///   _Analogues_
///     - **clGetDeviceFunctionPointerINTEL**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hDevice`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pFunctionName`
///         + `NULL == ppFunctionPointer`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetFunctionPointer(
    ur_device_handle_t hDevice,   ///< [in] handle of the device to retrieve pointer for.
    ur_program_handle_t hProgram, ///< [in] handle of the program to search for function in.
                                  ///< The program must already be built to the specified device, or
                                  ///< otherwise ::UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE is returned.
    const char *pFunctionName,    ///< [in] A null-terminates string denoting the mangled function name.
    void **ppFunctionPointer      ///< [out] Returns the pointer to the function if it is found in the program.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Program object information
typedef enum ur_program_info_t {
    UR_PROGRAM_INFO_REFERENCE_COUNT = 0, ///< [uint32_t] Reference count of the program object.
                                         ///< The reference count returned should be considered immediately stale.
                                         ///< It is unsuitable for general use in applications. This feature is
                                         ///< provided for identifying memory leaks.
    UR_PROGRAM_INFO_CONTEXT = 1,         ///< [::ur_context_handle_t] Program context info.
    UR_PROGRAM_INFO_NUM_DEVICES = 2,     ///< [uint32_t] Return number of devices associated with Program.
    UR_PROGRAM_INFO_DEVICES = 3,         ///< [::ur_device_handle_t[]] Return list of devices associated with
                                         ///< Program.
    UR_PROGRAM_INFO_SOURCE = 4,          ///< [char[]] Return program source associated with Program.
    UR_PROGRAM_INFO_BINARY_SIZES = 5,    ///< [size_t[]] Return program binary sizes for each device.
    UR_PROGRAM_INFO_BINARIES = 6,        ///< [unsigned char[]] Return program binaries for all devices for this
                                         ///< Program.
    UR_PROGRAM_INFO_NUM_KERNELS = 7,     ///< [size_t] Number of kernels in Program, return type size_t.
    UR_PROGRAM_INFO_KERNEL_NAMES = 8,    ///< [char[]] Return a null-terminated, semi-colon separated list of kernel
                                         ///< names in Program.
    /// @cond
    UR_PROGRAM_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_program_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a Program object
///
/// @remarks
///   _Analogues_
///     - **clGetProgramInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PROGRAM_INFO_KERNEL_NAMES < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetInfo(
    ur_program_handle_t hProgram, ///< [in] handle of the Program object
    ur_program_info_t propName,   ///< [in] name of the Program property to query
    size_t propSize,              ///< [in] the size of the Program property.
    void *pPropValue,             ///< [in,out][optional][typename(propName, propSize)] array of bytes of
                                  ///< holding the program info property.
                                  ///< If propSize is not equal to or greater than the real number of bytes
                                  ///< needed to return
                                  ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                  ///< pPropValue is not used.
    size_t *pPropSizeRet          ///< [out][optional] pointer to the actual size in bytes of the queried propName.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Program object build status
typedef enum ur_program_build_status_t {
    UR_PROGRAM_BUILD_STATUS_NONE = 0,        ///< Program build status none
    UR_PROGRAM_BUILD_STATUS_ERROR = 1,       ///< Program build error
    UR_PROGRAM_BUILD_STATUS_SUCCESS = 2,     ///< Program build success
    UR_PROGRAM_BUILD_STATUS_IN_PROGRESS = 3, ///< Program build in progress
    /// @cond
    UR_PROGRAM_BUILD_STATUS_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_program_build_status_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Program object binary type
typedef enum ur_program_binary_type_t {
    UR_PROGRAM_BINARY_TYPE_NONE = 0,            ///< No program binary is associated with device
    UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT = 1, ///< Program binary is compiled object
    UR_PROGRAM_BINARY_TYPE_LIBRARY = 2,         ///< Program binary is library object
    UR_PROGRAM_BINARY_TYPE_EXECUTABLE = 3,      ///< Program binary is executable
    /// @cond
    UR_PROGRAM_BINARY_TYPE_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_program_binary_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Program object build information
typedef enum ur_program_build_info_t {
    UR_PROGRAM_BUILD_INFO_STATUS = 0,      ///< [::ur_program_build_status_t] Program build status.
    UR_PROGRAM_BUILD_INFO_OPTIONS = 1,     ///< [char[]] Null-terminated options string specified by last build,
                                           ///< compile or link operation performed on the program.
    UR_PROGRAM_BUILD_INFO_LOG = 2,         ///< [char[]] Null-terminated program build log.
    UR_PROGRAM_BUILD_INFO_BINARY_TYPE = 3, ///< [::ur_program_binary_type_t] Program binary type.
    /// @cond
    UR_PROGRAM_BUILD_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_program_build_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query build information about a Program object for a Device
///
/// @remarks
///   _Analogues_
///     - **clGetProgramBuildInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PROGRAM_BUILD_INFO_BINARY_TYPE < propName`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetBuildInfo(
    ur_program_handle_t hProgram,     ///< [in] handle of the Program object
    ur_device_handle_t hDevice,       ///< [in] handle of the Device object
    ur_program_build_info_t propName, ///< [in] name of the Program build info to query
    size_t propSize,                  ///< [in] size of the Program build info property.
    void *pPropValue,                 ///< [in,out][optional][typename(propName, propSize)] value of the Program
                                      ///< build property.
                                      ///< If propSize is not equal to or greater than the real number of bytes
                                      ///< needed to return the info then the ::UR_RESULT_ERROR_INVALID_SIZE
                                      ///< error is returned and pPropValue is not used.
    size_t *pPropSizeRet              ///< [out][optional] pointer to the actual size in bytes of data being
                                      ///< queried by propName.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Specialization constant information
typedef struct ur_specialization_constant_info_t {
    uint32_t id;        ///< [in] specialization constant Id
    size_t size;        ///< [in] size of the specialization constant value
    const void *pValue; ///< [in] pointer to the specialization constant value bytes

} ur_specialization_constant_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Set an array of specialization constants on a Program.
///
/// @details
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///     - `hProgram` must have been created with the ::urProgramCreateWithIL
///       entry point.
///     - Any spec constants set with this entry point will apply only to
///       subsequent calls to ::urProgramBuild or ::urProgramCompile.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSpecConstants`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `count == 0`
UR_APIEXPORT ur_result_t UR_APICALL
urProgramSetSpecializationConstants(
    ur_program_handle_t hProgram,                           ///< [in] handle of the Program object
    uint32_t count,                                         ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t *pSpecConstants ///< [in][range(0, count)] array of specialization constant value
                                                            ///< descriptions
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Return program native program handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability program extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeProgram`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetNativeHandle(
    ur_program_handle_t hProgram,       ///< [in] handle of the program.
    ur_native_handle_t *phNativeProgram ///< [out] a pointer to the native handle of the program.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Native program creation properties
typedef struct ur_program_native_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_PROGRAM_NATIVE_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    bool isNativeHandleOwned;  ///< [in] Indicates UR owns the native handle or if it came from an
                               ///< interoperability operation in the application that asked to not
                               ///< transfer the ownership to the unified-runtime.

} ur_program_native_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime program object from native program handle.
///
/// @details
///     - Creates runtime program handle from native driver program handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phProgram`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCreateWithNativeHandle(
    ur_native_handle_t hNativeProgram,                 ///< [in][nocheck] the native handle of the program.
    ur_context_handle_t hContext,                      ///< [in] handle of the context instance
    const ur_program_native_properties_t *pProperties, ///< [in][optional] pointer to native program properties struct.
    ur_program_handle_t *phProgram                     ///< [out] pointer to the handle of the program object created.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs for Program
#if !defined(__GNUC__)
#pragma region kernel
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Create kernel object from a program.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pKernelName`
///         + `NULL == phKernel`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_NAME
///         + If `pKernelName` wasn't found in `hProgram`.
UR_APIEXPORT ur_result_t UR_APICALL
urKernelCreate(
    ur_program_handle_t hProgram, ///< [in] handle of the program instance
    const char *pKernelName,      ///< [in] pointer to null-terminated string.
    ur_kernel_handle_t *phKernel  ///< [out] pointer to handle of kernel object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties for for ::urKernelSetArgValue.
typedef struct ur_kernel_arg_value_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_KERNEL_ARG_VALUE_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure

} ur_kernel_arg_value_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Set kernel argument to a value.
///
/// @details
///     - The application may call this function from simultaneous threads with
///       the same kernel handle.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pArgValue`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgValue(
    ur_kernel_handle_t hKernel,                          ///< [in] handle of the kernel object
    uint32_t argIndex,                                   ///< [in] argument index in range [0, num args - 1]
    size_t argSize,                                      ///< [in] size of argument type
    const ur_kernel_arg_value_properties_t *pProperties, ///< [in][optional] pointer to value properties.
    const void *pArgValue                                ///< [in] argument value represented as matching arg type.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties for for ::urKernelSetArgLocal.
typedef struct ur_kernel_arg_local_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_KERNEL_ARG_LOCAL_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure

} ur_kernel_arg_local_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Set kernel argument to a local buffer.
///
/// @details
///     - The application may call this function from simultaneous threads with
///       the same kernel handle.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgLocal(
    ur_kernel_handle_t hKernel,                         ///< [in] handle of the kernel object
    uint32_t argIndex,                                  ///< [in] argument index in range [0, num args - 1]
    size_t argSize,                                     ///< [in] size of the local buffer to be allocated by the runtime
    const ur_kernel_arg_local_properties_t *pProperties ///< [in][optional] pointer to local buffer properties.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Kernel object information
typedef enum ur_kernel_info_t {
    UR_KERNEL_INFO_FUNCTION_NAME = 0,   ///< [char[]] Return null-terminated kernel function name.
    UR_KERNEL_INFO_NUM_ARGS = 1,        ///< [size_t] Return Kernel number of arguments.
    UR_KERNEL_INFO_REFERENCE_COUNT = 2, ///< [uint32_t] Reference count of the kernel object.
                                        ///< The reference count returned should be considered immediately stale.
                                        ///< It is unsuitable for general use in applications. This feature is
                                        ///< provided for identifying memory leaks.
    UR_KERNEL_INFO_CONTEXT = 3,         ///< [::ur_context_handle_t] Return Context object associated with Kernel.
    UR_KERNEL_INFO_PROGRAM = 4,         ///< [::ur_program_handle_t] Return Program object associated with Kernel.
    UR_KERNEL_INFO_ATTRIBUTES = 5,      ///< [char[]] Return null-terminated kernel attributes string.
    UR_KERNEL_INFO_NUM_REGS = 6,        ///< [uint32_t] Return the number of registers used by the compiled kernel
                                        ///< (device specific).
    /// @cond
    UR_KERNEL_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_kernel_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Kernel Work Group information
typedef enum ur_kernel_group_info_t {
    UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE = 0,                   ///< [size_t[3]] Return Work Group maximum global size
    UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE = 1,                    ///< [size_t] Return maximum Work Group size
    UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE = 2,            ///< [size_t[3]] Return Work Group size required by the source code, such
                                                                 ///< as __attribute__((required_work_group_size(X,Y,Z))
    UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE = 3,                     ///< [size_t] Return local memory required by the Kernel
    UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 4, ///< [size_t] Return preferred multiple of Work Group size for launch
    UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE = 5,                   ///< [size_t] Return minimum amount of private memory in bytes used by each
                                                                 ///< work item in the Kernel
    /// @cond
    UR_KERNEL_GROUP_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_kernel_group_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Kernel SubGroup information
typedef enum ur_kernel_sub_group_info_t {
    UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE = 0,     ///< [uint32_t] Return maximum SubGroup size
    UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS = 1,     ///< [uint32_t] Return maximum number of SubGroup
    UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS = 2, ///< [uint32_t] Return number of SubGroup required by the source code
    UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL = 3,   ///< [uint32_t] Return SubGroup size required by Intel
    /// @cond
    UR_KERNEL_SUB_GROUP_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_kernel_sub_group_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Kernel Cache Configuration.
typedef enum ur_kernel_cache_config_t {
    UR_KERNEL_CACHE_CONFIG_DEFAULT = 0,    ///< No preference for SLM or data cache.
    UR_KERNEL_CACHE_CONFIG_LARGE_SLM = 1,  ///< Large Shared Local Memory (SLM) size.
    UR_KERNEL_CACHE_CONFIG_LARGE_DATA = 2, ///< Large General Data size.
    /// @cond
    UR_KERNEL_CACHE_CONFIG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_kernel_cache_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Set additional Kernel execution information
typedef enum ur_kernel_exec_info_t {
    UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS = 0, ///< [::ur_bool_t] Kernel might access data through USM pointer.
    UR_KERNEL_EXEC_INFO_USM_PTRS = 1,            ///< [void *[]] Provide an explicit array of USM pointers that the kernel
                                                 ///< will access.
    UR_KERNEL_EXEC_INFO_CACHE_CONFIG = 2,        ///< [::ur_kernel_cache_config_t] Provide the preferred cache configuration
    /// @cond
    UR_KERNEL_EXEC_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_kernel_exec_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a Kernel object
///
/// @remarks
///   _Analogues_
///     - **clGetKernelInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_INFO_NUM_REGS < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetInfo(
    ur_kernel_handle_t hKernel, ///< [in] handle of the Kernel object
    ur_kernel_info_t propName,  ///< [in] name of the Kernel property to query
    size_t propSize,            ///< [in] the size of the Kernel property value.
    void *pPropValue,           ///< [in,out][optional][typename(propName, propSize)] array of bytes
                                ///< holding the kernel info property.
                                ///< If propSize is not equal to or greater than the real number of bytes
                                ///< needed to return
                                ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                ///< pPropValue is not used.
    size_t *pPropSizeRet        ///< [out][optional] pointer to the actual size in bytes of data being
                                ///< queried by propName.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Query work Group information about a Kernel object
///
/// @remarks
///   _Analogues_
///     - **clGetKernelWorkGroupInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE < propName`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetGroupInfo(
    ur_kernel_handle_t hKernel,      ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice,      ///< [in] handle of the Device object
    ur_kernel_group_info_t propName, ///< [in] name of the work Group property to query
    size_t propSize,                 ///< [in] size of the Kernel Work Group property value
    void *pPropValue,                ///< [in,out][optional][typename(propName, propSize)] value of the Kernel
                                     ///< Work Group property.
    size_t *pPropSizeRet             ///< [out][optional] pointer to the actual size in bytes of data being
                                     ///< queried by propName.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Query SubGroup information about a Kernel object
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL < propName`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetSubGroupInfo(
    ur_kernel_handle_t hKernel,          ///< [in] handle of the Kernel object
    ur_device_handle_t hDevice,          ///< [in] handle of the Device object
    ur_kernel_sub_group_info_t propName, ///< [in] name of the SubGroup property to query
    size_t propSize,                     ///< [in] size of the Kernel SubGroup property value
    void *pPropValue,                    ///< [in,out][optional][typename(propName, propSize)] value of the Kernel
                                         ///< SubGroup property.
    size_t *pPropSizeRet                 ///< [out][optional] pointer to the actual size in bytes of data being
                                         ///< queried by propName.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the Kernel object.
///
/// @details
///     - Get a reference to the Kernel object handle. Increment its reference
///       count
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clRetainKernel**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelRetain(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to retain
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Release Kernel.
///
/// @details
///     - Decrement reference count and destroy the Kernel if reference count
///       becomes zero.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clReleaseKernel**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelRelease(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to release
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties for for ::urKernelSetArgPointer.
typedef struct ur_kernel_arg_pointer_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_KERNEL_ARG_POINTER_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure

} ur_kernel_arg_pointer_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Set a USM pointer as the argument value of a Kernel.
///
/// @details
///     - The application may call this function from simultaneous threads with
///       the same kernel handle.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clSetKernelArgSVMPointer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgPointer(
    ur_kernel_handle_t hKernel,                            ///< [in] handle of the kernel object
    uint32_t argIndex,                                     ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_pointer_properties_t *pProperties, ///< [in][optional] pointer to USM pointer properties.
    const void *pArgValue                                  ///< [in][optional] USM pointer to memory location holding the argument
                                                           ///< value. If null then argument value is considered null.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties for for ::urKernelSetExecInfo.
typedef struct ur_kernel_exec_info_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_KERNEL_EXEC_INFO_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure

} ur_kernel_exec_info_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Set additional Kernel execution attributes.
///
/// @details
///     - The application must **not** call this function from simultaneous
///       threads with the same kernel handle.
///     - The implementation of this function should be lock-free.
///
/// @remarks
///   _Analogues_
///     - **clSetKernelExecInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_KERNEL_EXEC_INFO_CACHE_CONFIG < propName`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPropValue`
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetExecInfo(
    ur_kernel_handle_t hKernel,                          ///< [in] handle of the kernel object
    ur_kernel_exec_info_t propName,                      ///< [in] name of the execution attribute
    size_t propSize,                                     ///< [in] size in byte the attribute value
    const ur_kernel_exec_info_properties_t *pProperties, ///< [in][optional] pointer to execution info properties.
    const void *pPropValue                               ///< [in][typename(propName, propSize)] pointer to memory location holding
                                                         ///< the property value.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties for for ::urKernelSetArgSampler.
typedef struct ur_kernel_arg_sampler_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_KERNEL_ARG_SAMPLER_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure

} ur_kernel_arg_sampler_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Set a Sampler object as the argument value of a Kernel.
///
/// @details
///     - The application may call this function from simultaneous threads with
///       the same kernel handle.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///         + `NULL == hArgValue`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgSampler(
    ur_kernel_handle_t hKernel,                            ///< [in] handle of the kernel object
    uint32_t argIndex,                                     ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_sampler_properties_t *pProperties, ///< [in][optional] pointer to sampler properties.
    ur_sampler_handle_t hArgValue                          ///< [in] handle of Sampler object.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties for for ::urKernelSetArgMemObj.
typedef struct ur_kernel_arg_mem_obj_properties_t {
    ur_structure_type_t stype;   ///< [in] type of this structure, must be
                                 ///< ::UR_STRUCTURE_TYPE_KERNEL_ARG_MEM_OBJ_PROPERTIES
    void *pNext;                 ///< [in,out][optional] pointer to extension-specific structure
    ur_mem_flags_t memoryAccess; ///< [in] Memory access flag. Allowed values are: ::UR_MEM_FLAG_READ_WRITE,
                                 ///< ::UR_MEM_FLAG_WRITE_ONLY, ::UR_MEM_FLAG_READ_ONLY.

} ur_kernel_arg_mem_obj_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Set a Memory object as the argument value of a Kernel.
///
/// @details
///     - The application may call this function from simultaneous threads with
///       the same kernel handle.
///     - The implementation of this function should be lock-free.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pProperties && ::UR_MEM_FLAGS_MASK & pProperties->memoryAccess`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgMemObj(
    ur_kernel_handle_t hKernel,                            ///< [in] handle of the kernel object
    uint32_t argIndex,                                     ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_mem_obj_properties_t *pProperties, ///< [in][optional] pointer to Memory object properties.
    ur_mem_handle_t hArgValue                              ///< [in][optional] handle of Memory object.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Set an array of specialization constants on a Kernel.
///
/// @details
///     - This entry point is optional, the application should query for support
///       with device query ::UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS
///       passed to ::urDeviceGetInfo.
///     - Adapters which are capable of setting specialization constants
///       immediately prior to ::urEnqueueKernelLaunch with low overhead should
///       implement this entry point.
///     - Otherwise, if setting specialization constants late requires
///       recompiling or linking a program, adapters should not implement this
///       entry point.
///     - The application may call this function from simultaneous threads for
///       the same device.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSpecConstants`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `count == 0`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If ::UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS query is false
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetSpecializationConstants(
    ur_kernel_handle_t hKernel,                             ///< [in] handle of the kernel object
    uint32_t count,                                         ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t *pSpecConstants ///< [in] array of specialization constant value descriptions
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Return platform native kernel handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability platform extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeKernel`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetNativeHandle(
    ur_kernel_handle_t hKernel,        ///< [in] handle of the kernel.
    ur_native_handle_t *phNativeKernel ///< [out] a pointer to the native handle of the kernel.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties for for ::urKernelCreateWithNativeHandle.
typedef struct ur_kernel_native_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_KERNEL_NATIVE_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    bool isNativeHandleOwned;  ///< [in] Indicates UR owns the native handle or if it came from an interoperability
                               ///< operation in the application that asked to not transfer the ownership to
                               ///< the unified-runtime.

} ur_kernel_native_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime kernel object from native kernel handle.
///
/// @details
///     - Creates runtime kernel handle from native driver kernel handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phKernel`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urKernelCreateWithNativeHandle(
    ur_native_handle_t hNativeKernel,                 ///< [in][nocheck] the native handle of the kernel.
    ur_context_handle_t hContext,                     ///< [in] handle of the context object
    ur_program_handle_t hProgram,                     ///< [in] handle of the program associated with the kernel
    const ur_kernel_native_properties_t *pProperties, ///< [in][optional] pointer to native kernel properties struct
    ur_kernel_handle_t *phKernel                      ///< [out] pointer to the handle of the kernel object created.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs
#if !defined(__GNUC__)
#pragma region queue
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Query queue info
typedef enum ur_queue_info_t {
    UR_QUEUE_INFO_CONTEXT = 0,         ///< [::ur_context_handle_t] context associated with this queue.
    UR_QUEUE_INFO_DEVICE = 1,          ///< [::ur_device_handle_t] device associated with this queue.
    UR_QUEUE_INFO_DEVICE_DEFAULT = 2,  ///< [::ur_queue_handle_t] the current default queue of the underlying
                                       ///< device.
    UR_QUEUE_INFO_FLAGS = 3,           ///< [::ur_queue_flags_t] the properties associated with
                                       ///< ::ur_queue_properties_t::flags.
    UR_QUEUE_INFO_REFERENCE_COUNT = 4, ///< [uint32_t] Reference count of the queue object.
                                       ///< The reference count returned should be considered immediately stale.
                                       ///< It is unsuitable for general use in applications. This feature is
                                       ///< provided for identifying memory leaks.
    UR_QUEUE_INFO_SIZE = 5,            ///< [uint32_t] The size of the queue
    UR_QUEUE_INFO_EMPTY = 6,           ///< [::ur_bool_t] return true if the queue was empty at the time of the
                                       ///< query
    /// @cond
    UR_QUEUE_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_queue_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Queue property flags
typedef uint32_t ur_queue_flags_t;
typedef enum ur_queue_flag_t {
    UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE = UR_BIT(0), ///< Enable/disable out of order execution
    UR_QUEUE_FLAG_PROFILING_ENABLE = UR_BIT(1),              ///< Enable/disable profiling
    UR_QUEUE_FLAG_ON_DEVICE = UR_BIT(2),                     ///< Is a device queue
    UR_QUEUE_FLAG_ON_DEVICE_DEFAULT = UR_BIT(3),             ///< Is the default queue for a device
    UR_QUEUE_FLAG_DISCARD_EVENTS = UR_BIT(4),                ///< Events will be discarded
    UR_QUEUE_FLAG_PRIORITY_LOW = UR_BIT(5),                  ///< Low priority queue
    UR_QUEUE_FLAG_PRIORITY_HIGH = UR_BIT(6),                 ///< High priority queue
    UR_QUEUE_FLAG_SUBMISSION_BATCHED = UR_BIT(7),            ///< Hint: enqueue and submit in a batch later. No change in queue
                                                             ///< semantics. Implementation chooses submission mode.
    UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE = UR_BIT(8),          ///< Hint: enqueue and submit immediately. No change in queue semantics.
                                                             ///< Implementation chooses submission mode.
    UR_QUEUE_FLAG_USE_DEFAULT_STREAM = UR_BIT(9),            ///< Use the default stream. Only meaningful for CUDA. Other platforms may
                                                             ///< ignore this flag.
    UR_QUEUE_FLAG_SYNC_WITH_DEFAULT_STREAM = UR_BIT(10),     ///< Synchronize with the default stream. Only meaningful for CUDA. Other
                                                             ///< platforms may ignore this flag.
    /// @cond
    UR_QUEUE_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_queue_flag_t;
/// @brief Bit Mask for validating ur_queue_flags_t
#define UR_QUEUE_FLAGS_MASK 0xfffff800

///////////////////////////////////////////////////////////////////////////////
/// @brief Query information about a command queue
///
/// @remarks
///   _Analogues_
///     - **clGetCommandQueueInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_QUEUE_INFO_EMPTY < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urQueueGetInfo(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_queue_info_t propName, ///< [in] name of the queue property to query
    size_t propSize,          ///< [in] size in bytes of the queue property value provided
    void *pPropValue,         ///< [out][optional][typename(propName, propSize)] value of the queue
                              ///< property
    size_t *pPropSizeRet      ///< [out][optional] size in bytes returned in queue property value
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Queue creation properties
typedef struct ur_queue_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_QUEUE_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    ur_queue_flags_t flags;    ///< [in] Bitfield of queue creation flags

} ur_queue_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Queue index creation properties
///
/// @details
///     - Specify these properties in ::urQueueCreate via
///       ::ur_queue_properties_t as part of a `pNext` chain.
typedef struct ur_queue_index_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_QUEUE_INDEX_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    uint32_t computeIndex;     ///< [in] Specifies the compute index as described in the
                               ///< sycl_ext_intel_queue_index extension.

} ur_queue_index_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a command queue for a device in a context
///
/// @details
///     - See also ::ur_queue_index_properties_t.
///
/// @remarks
///   _Analogues_
///     - **clCreateCommandQueueWithProperties**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pProperties && ::UR_QUEUE_FLAGS_MASK & pProperties->flags`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phQueue`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES
///         + `pProperties != NULL && pProperties->flags & UR_QUEUE_FLAG_PRIORITY_HIGH && pProperties->flags & UR_QUEUE_FLAG_PRIORITY_LOW`
///         + `pProperties != NULL && pProperties->flags & UR_QUEUE_FLAG_SUBMISSION_BATCHED && pProperties->flags & UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urQueueCreate(
    ur_context_handle_t hContext,             ///< [in] handle of the context object
    ur_device_handle_t hDevice,               ///< [in] handle of the device object
    const ur_queue_properties_t *pProperties, ///< [in][optional] pointer to queue creation properties.
    ur_queue_handle_t *phQueue                ///< [out] pointer to handle of queue object created
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to the command queue handle. Increment the command
///        queue's reference count
///
/// @details
///     - Useful in library function to retain access to the command queue after
///       the caller released the queue.
///
/// @remarks
///   _Analogues_
///     - **clRetainCommandQueue**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urQueueRetain(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to get access
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the command queue's reference count and delete the command
///        queue if the reference count becomes zero.
///
/// @details
///     - After the command queue reference count becomes zero and all queued
///       commands in the queue have finished, the queue is deleted.
///     - It also performs an implicit flush to issue all previously queued
///       commands in the queue.
///
/// @remarks
///   _Analogues_
///     - **clReleaseCommandQueue**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urQueueRelease(
    ur_queue_handle_t hQueue ///< [in] handle of the queue object to release
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Descriptor for ::urQueueGetNativeHandle and
///        ::urQueueCreateWithNativeHandle.
///
/// @details
///     - Specify this descriptor in ::urQueueGetNativeHandle directly or
///       ::urQueueCreateWithNativeHandle via ::ur_queue_native_properties_t as
///       part of a `pNext` chain.
typedef struct ur_queue_native_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_QUEUE_NATIVE_DESC
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    void *pNativeData;         ///< [in][optional] Adapter-specific metadata needed to create the handle.

} ur_queue_native_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Return queue native queue handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability queue extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeQueue`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urQueueGetNativeHandle(
    ur_queue_handle_t hQueue,         ///< [in] handle of the queue.
    ur_queue_native_desc_t *pDesc,    ///< [in][optional] pointer to native descriptor
    ur_native_handle_t *phNativeQueue ///< [out] a pointer to the native handle of the queue.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties for for ::urQueueCreateWithNativeHandle.
typedef struct ur_queue_native_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_QUEUE_NATIVE_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    bool isNativeHandleOwned;  ///< [in] Indicates UR owns the native handle or if it came from an interoperability
                               ///< operation in the application that asked to not transfer the ownership to
                               ///< the unified-runtime.

} ur_queue_native_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime queue object from native queue handle.
///
/// @details
///     - Creates runtime queue handle from native driver queue handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phQueue`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue,                 ///< [in][nocheck] the native handle of the queue.
    ur_context_handle_t hContext,                    ///< [in] handle of the context object
    ur_device_handle_t hDevice,                      ///< [in] handle of the device object
    const ur_queue_native_properties_t *pProperties, ///< [in][optional] pointer to native queue properties struct
    ur_queue_handle_t *phQueue                       ///< [out] pointer to the handle of the queue object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Blocks until all previously issued commands to the command queue are
///        finished.
///
/// @details
///     - Blocks until all previously issued commands to the command queue are
///       issued and completed.
///     - ::urQueueFinish does not return until all enqueued commands have been
///       processed and finished.
///     - ::urQueueFinish acts as a synchronization point.
///
/// @remarks
///   _Analogues_
///     - **clFinish**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urQueueFinish(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be finished.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Issues all previously enqueued commands in a command queue to the
///        device.
///
/// @details
///     - Guarantees that all enqueued commands will be issued to the
///       appropriate device.
///     - There is no guarantee that they will be completed after ::urQueueFlush
///       returns.
///
/// @remarks
///   _Analogues_
///     - **clFlush**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urQueueFlush(
    ur_queue_handle_t hQueue ///< [in] handle of the queue to be flushed.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs
#if !defined(__GNUC__)
#pragma region event
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Command type
typedef enum ur_command_t {
    UR_COMMAND_KERNEL_LAUNCH = 0,                     ///< Event created by ::urEnqueueKernelLaunch
    UR_COMMAND_EVENTS_WAIT = 1,                       ///< Event created by ::urEnqueueEventsWait
    UR_COMMAND_EVENTS_WAIT_WITH_BARRIER = 2,          ///< Event created by ::urEnqueueEventsWaitWithBarrier
    UR_COMMAND_MEM_BUFFER_READ = 3,                   ///< Event created by ::urEnqueueMemBufferRead
    UR_COMMAND_MEM_BUFFER_WRITE = 4,                  ///< Event created by ::urEnqueueMemBufferWrite
    UR_COMMAND_MEM_BUFFER_READ_RECT = 5,              ///< Event created by ::urEnqueueMemBufferReadRect
    UR_COMMAND_MEM_BUFFER_WRITE_RECT = 6,             ///< Event created by ::urEnqueueMemBufferWriteRect
    UR_COMMAND_MEM_BUFFER_COPY = 7,                   ///< Event created by ::urEnqueueMemBufferCopy
    UR_COMMAND_MEM_BUFFER_COPY_RECT = 8,              ///< Event created by ::urEnqueueMemBufferCopyRect
    UR_COMMAND_MEM_BUFFER_FILL = 9,                   ///< Event created by ::urEnqueueMemBufferFill
    UR_COMMAND_MEM_IMAGE_READ = 10,                   ///< Event created by ::urEnqueueMemImageRead
    UR_COMMAND_MEM_IMAGE_WRITE = 11,                  ///< Event created by ::urEnqueueMemImageWrite
    UR_COMMAND_MEM_IMAGE_COPY = 12,                   ///< Event created by ::urEnqueueMemImageCopy
    UR_COMMAND_MEM_BUFFER_MAP = 14,                   ///< Event created by ::urEnqueueMemBufferMap
    UR_COMMAND_MEM_UNMAP = 16,                        ///< Event created by ::urEnqueueMemUnmap
    UR_COMMAND_USM_FILL = 17,                         ///< Event created by ::urEnqueueUSMFill
    UR_COMMAND_USM_MEMCPY = 18,                       ///< Event created by ::urEnqueueUSMMemcpy
    UR_COMMAND_USM_PREFETCH = 19,                     ///< Event created by ::urEnqueueUSMPrefetch
    UR_COMMAND_USM_ADVISE = 20,                       ///< Event created by ::urEnqueueUSMAdvise
    UR_COMMAND_USM_FILL_2D = 21,                      ///< Event created by ::urEnqueueUSMFill2D
    UR_COMMAND_USM_MEMCPY_2D = 22,                    ///< Event created by ::urEnqueueUSMMemcpy2D
    UR_COMMAND_DEVICE_GLOBAL_VARIABLE_WRITE = 23,     ///< Event created by ::urEnqueueDeviceGlobalVariableWrite
    UR_COMMAND_DEVICE_GLOBAL_VARIABLE_READ = 24,      ///< Event created by ::urEnqueueDeviceGlobalVariableRead
    UR_COMMAND_READ_HOST_PIPE = 25,                   ///< Event created by ::urEnqueueReadHostPipe
    UR_COMMAND_WRITE_HOST_PIPE = 26,                  ///< Event created by ::urEnqueueWriteHostPipe
    UR_COMMAND_COMMAND_BUFFER_ENQUEUE_EXP = 0x1000,   ///< Event created by ::urCommandBufferEnqueueExp
    UR_COMMAND_INTEROP_SEMAPHORE_WAIT_EXP = 0x2000,   ///< Event created by ::urBindlessImagesWaitExternalSemaphoreExp
    UR_COMMAND_INTEROP_SEMAPHORE_SIGNAL_EXP = 0x2001, ///< Event created by ::urBindlessImagesSignalExternalSemaphoreExp
    /// @cond
    UR_COMMAND_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_command_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event Status
typedef enum ur_event_status_t {
    UR_EVENT_STATUS_COMPLETE = 0,  ///< Command is complete
    UR_EVENT_STATUS_RUNNING = 1,   ///< Command is running
    UR_EVENT_STATUS_SUBMITTED = 2, ///< Command is submitted
    UR_EVENT_STATUS_QUEUED = 3,    ///< Command is queued
    /// @cond
    UR_EVENT_STATUS_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_event_status_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event query information type
typedef enum ur_event_info_t {
    UR_EVENT_INFO_COMMAND_QUEUE = 0,            ///< [::ur_queue_handle_t] Command queue information of an event object
    UR_EVENT_INFO_CONTEXT = 1,                  ///< [::ur_context_handle_t] Context information of an event object
    UR_EVENT_INFO_COMMAND_TYPE = 2,             ///< [::ur_command_t] Command type information of an event object
    UR_EVENT_INFO_COMMAND_EXECUTION_STATUS = 3, ///< [::ur_event_status_t] Command execution status of an event object
    UR_EVENT_INFO_REFERENCE_COUNT = 4,          ///< [uint32_t] Reference count of the event object.
                                                ///< The reference count returned should be considered immediately stale.
                                                ///< It is unsuitable for general use in applications. This feature is
                                                ///< provided for identifying memory leaks.
    /// @cond
    UR_EVENT_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_event_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Profiling query information type
typedef enum ur_profiling_info_t {
    UR_PROFILING_INFO_COMMAND_QUEUED = 0,   ///< [uint64_t] A 64-bit value of current device counter in nanoseconds
                                            ///< when the event is enqueued
    UR_PROFILING_INFO_COMMAND_SUBMIT = 1,   ///< [uint64_t] A 64-bit value of current device counter in nanoseconds
                                            ///< when the event is submitted
    UR_PROFILING_INFO_COMMAND_START = 2,    ///< [uint64_t] A 64-bit value of current device counter in nanoseconds
                                            ///< when the event starts execution
    UR_PROFILING_INFO_COMMAND_END = 3,      ///< [uint64_t] A 64-bit value of current device counter in nanoseconds
                                            ///< when the event has finished execution
    UR_PROFILING_INFO_COMMAND_COMPLETE = 4, ///< [uint64_t] A 64-bit value of current device counter in nanoseconds
                                            ///< when the event and any child events enqueued by this event on the
                                            ///< device have finished execution
    /// @cond
    UR_PROFILING_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_profiling_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get event object information
///
/// @remarks
///   _Analogues_
///     - **clGetEventInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EVENT_INFO_REFERENCE_COUNT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urEventGetInfo(
    ur_event_handle_t hEvent, ///< [in] handle of the event object
    ur_event_info_t propName, ///< [in] the name of the event property to query
    size_t propSize,          ///< [in] size in bytes of the event property value
    void *pPropValue,         ///< [out][optional][typename(propName, propSize)] value of the event
                              ///< property
    size_t *pPropSizeRet      ///< [out][optional] bytes returned in event property
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get profiling information for the command associated with an event
///        object
///
/// @remarks
///   _Analogues_
///     - **clGetEventProfilingInfo**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_PROFILING_INFO_COMMAND_COMPLETE < propName`
///     - ::UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE
///         + If `hEvent`s associated queue was not created with `UR_QUEUE_FLAG_PROFILING_ENABLE`.
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `pPropValue && propSize == 0`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urEventGetProfilingInfo(
    ur_event_handle_t hEvent,     ///< [in] handle of the event object
    ur_profiling_info_t propName, ///< [in] the name of the profiling property to query
    size_t propSize,              ///< [in] size in bytes of the profiling property value
    void *pPropValue,             ///< [out][optional][typename(propName, propSize)] value of the profiling
                                  ///< property
    size_t *pPropSizeRet          ///< [out][optional] pointer to the actual size in bytes returned in
                                  ///< propValue
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Wait for a list of events to finish.
///
/// @remarks
///   _Analogues_
///     - **clWaitForEvent**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phEventWaitList`
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `numEvents == 0`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEventWait(
    uint32_t numEvents,                      ///< [in] number of events in the event list
    const ur_event_handle_t *phEventWaitList ///< [in][range(0, numEvents)] pointer to a list of events to wait for
                                             ///< completion
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get a reference to an event handle. Increment the event object's
///        reference count.
///
/// @remarks
///   _Analogues_
///     - **clRetainEvent**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urEventRetain(
    ur_event_handle_t hEvent ///< [in] handle of the event object
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the event object's reference count and delete the event
///        object if the reference count becomes zero.
///
/// @remarks
///   _Analogues_
///     - **clReleaseEvent**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urEventRelease(
    ur_event_handle_t hEvent ///< [in] handle of the event object
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Return platform native event handle.
///
/// @details
///     - Retrieved native handle can be used for direct interaction with the
///       native platform driver.
///     - Use interoperability platform extensions to convert native handle to
///       native type.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phNativeEvent`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urEventGetNativeHandle(
    ur_event_handle_t hEvent,         ///< [in] handle of the event.
    ur_native_handle_t *phNativeEvent ///< [out] a pointer to the native handle of the event.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties for for ::urEventCreateWithNativeHandle.
typedef struct ur_event_native_properties_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_EVENT_NATIVE_PROPERTIES
    void *pNext;               ///< [in,out][optional] pointer to extension-specific structure
    bool isNativeHandleOwned;  ///< [in] Indicates UR owns the native handle or if it came from an interoperability
                               ///< operation in the application that asked to not transfer the ownership to
                               ///< the unified-runtime.

} ur_event_native_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create runtime event object from native event handle.
///
/// @details
///     - Creates runtime event handle from native driver event handle.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phEvent`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If the adapter has no underlying equivalent handle.
UR_APIEXPORT ur_result_t UR_APICALL
urEventCreateWithNativeHandle(
    ur_native_handle_t hNativeEvent,                 ///< [in][nocheck] the native handle of the event.
    ur_context_handle_t hContext,                    ///< [in] handle of the context object
    const ur_event_native_properties_t *pProperties, ///< [in][optional] pointer to native event properties struct
    ur_event_handle_t *phEvent                       ///< [out] pointer to the handle of the event object created.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Event states for all events.
typedef enum ur_execution_info_t {
    UR_EXECUTION_INFO_COMPLETE = 0,  ///< Indicates that the event has completed.
    UR_EXECUTION_INFO_RUNNING = 1,   ///< Indicates that the device has started processing this event.
    UR_EXECUTION_INFO_SUBMITTED = 2, ///< Indicates that the event has been submitted by the host to the device.
    UR_EXECUTION_INFO_QUEUED = 3,    ///< Indicates that the event has been queued, this is the initial state of
                                     ///< events.
    /// @cond
    UR_EXECUTION_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_execution_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event callback function that can be registered by the application.
typedef void (*ur_event_callback_t)(
    ur_event_handle_t hEvent,       ///< [in] handle to event
    ur_execution_info_t execStatus, ///< [in] execution status of the event
    void *pUserData                 ///< [in][out] pointer to data to be passed to callback
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Register a user callback function for a specific command execution
///        status.
///
/// @details
///     - The registered callback function will be called when the execution
///       status of command associated with event changes to an execution status
///       equal to or past the status specified by command_exec_status.
///     - `execStatus` must not be `UR_EXECUTION_INFO_QUEUED` as this is the
///       initial state of all events.
///     - The application may call this function from simultaneous threads for
///       the same context.
///     - The implementation of this function should be thread-safe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hEvent`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EXECUTION_INFO_QUEUED < execStatus`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pfnNotify`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + `execStatus == UR_EXECUTION_INFO_QUEUED`
UR_APIEXPORT ur_result_t UR_APICALL
urEventSetCallback(
    ur_event_handle_t hEvent,       ///< [in] handle of the event object
    ur_execution_info_t execStatus, ///< [in] execution status of the event
    ur_event_callback_t pfnNotify,  ///< [in] execution status of the event
    void *pUserData                 ///< [in][out][optional] pointer to data to be passed to callback.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime APIs
#if !defined(__GNUC__)
#pragma region enqueue
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to execute a kernel
///
/// @remarks
///   _Analogues_
///     - **clEnqueueNDRangeKernel**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pGlobalWorkOffset`
///         + `NULL == pGlobalWorkSize`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_KERNEL
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_WORK_DIMENSION
///     - ::UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_kernel_handle_t hKernel,               ///< [in] handle of the kernel object
    uint32_t workDim,                         ///< [in] number of dimensions, from 1 to 3, to specify the global and
                                              ///< work-group work-items
    const size_t *pGlobalWorkOffset,          ///< [in] pointer to an array of workDim unsigned values that specify the
                                              ///< offset used to calculate the global ID of a work-item
    const size_t *pGlobalWorkSize,            ///< [in] pointer to an array of workDim unsigned values that specify the
                                              ///< number of global work-items in workDim that will execute the kernel
                                              ///< function
    const size_t *pLocalWorkSize,             ///< [in][optional] pointer to an array of workDim unsigned values that
                                              ///< specify the number of local work-items forming a work-group that will
                                              ///< execute the kernel function.
                                              ///< If nullptr, the runtime implementation will choose the work-group
                                              ///< size.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< kernel execution instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command which waits a list of events to complete before it
///        completes
///
/// @details
///     - If the event list is empty, it waits for all previously enqueued
///       commands to complete.
///     - It returns an event which can be waited on.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueMarkerWithWaitList**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueEventsWait(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
                                              ///< previously enqueued commands
                                              ///< must be complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a barrier command which waits a list of events to complete
///        before it completes
///
/// @details
///     - If the event list is empty, it waits for all previously enqueued
///       commands to complete.
///     - It blocks command execution - any following commands enqueued after it
///       do not execute until it completes.
///     - It returns an event which can be waited on.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueBarrierWithWaitList**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
                                              ///< previously enqueued commands
                                              ///< must be complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to read from a buffer object to host memory
///
/// @details
///     - Input parameter blockingRead indicates if the read is blocking or
///       non-blocking.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueReadBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If `offset + size` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in][bounds(offset, size)] handle of the buffer object
    bool blockingRead,                        ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,                            ///< [in] offset in bytes in the buffer object
    size_t size,                              ///< [in] size in bytes of data being read
    void *pDst,                               ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to write into a buffer object from host memory
///
/// @details
///     - Input parameter blockingWrite indicates if the write is blocking or
///       non-blocking.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueWriteBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If `offset + size` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in][bounds(offset, size)] handle of the buffer object
    bool blockingWrite,                       ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,                            ///< [in] offset in bytes in the buffer object
    size_t size,                              ///< [in] size in bytes of data being written
    const void *pSrc,                         ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to read a 2D or 3D rectangular region from a buffer
///        object to host memory
///
/// @details
///     - Input parameter blockingRead indicates if the read is blocking or
///       non-blocking.
///     - The buffer and host 2D or 3D rectangular regions can have different
///       shapes.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueReadBufferRect**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `region.width == 0 || region.height == 0 || region.width == 0`
///         + `bufferRowPitch != 0 && bufferRowPitch < region.width`
///         + `hostRowPitch != 0 && hostRowPitch < region.width`
///         + `bufferSlicePitch != 0 && bufferSlicePitch < region.height * (bufferRowPitch != 0 ? bufferRowPitch : region.width)`
///         + `bufferSlicePitch != 0 && bufferSlicePitch % (bufferRowPitch != 0 ? bufferRowPitch : region.width) != 0`
///         + `hostSlicePitch != 0 && hostSlicePitch < region.height * (hostRowPitch != 0 ? hostRowPitch : region.width)`
///         + `hostSlicePitch != 0 && hostSlicePitch % (hostRowPitch != 0 ? hostRowPitch : region.width) != 0`
///         + If the combination of `bufferOrigin`, `region`, `bufferRowPitch`, and `bufferSlicePitch` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in][bounds(bufferOrigin, region)] handle of the buffer object
    bool blockingRead,                        ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOrigin,            ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOrigin,              ///< [in] 3D offset in the host region
    ur_rect_region_t region,                  ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t bufferRowPitch,                    ///< [in] length of each row in bytes in the buffer object
    size_t bufferSlicePitch,                  ///< [in] length of each 2D slice in bytes in the buffer object being read
    size_t hostRowPitch,                      ///< [in] length of each row in bytes in the host memory region pointed by
                                              ///< dst
    size_t hostSlicePitch,                    ///< [in] length of each 2D slice in bytes in the host memory region
                                              ///< pointed by dst
    void *pDst,                               ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to write a 2D or 3D rectangular region in a buffer
///        object from host memory
///
/// @details
///     - Input parameter blockingWrite indicates if the write is blocking or
///       non-blocking.
///     - The buffer and host 2D or 3D rectangular regions can have different
///       shapes.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueWriteBufferRect**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `region.width == 0 || region.height == 0 || region.width == 0`
///         + `bufferRowPitch != 0 && bufferRowPitch < region.width`
///         + `hostRowPitch != 0 && hostRowPitch < region.width`
///         + `bufferSlicePitch != 0 && bufferSlicePitch < region.height * (bufferRowPitch != 0 ? bufferRowPitch : region.width)`
///         + `bufferSlicePitch != 0 && bufferSlicePitch % (bufferRowPitch != 0 ? bufferRowPitch : region.width) != 0`
///         + `hostSlicePitch != 0 && hostSlicePitch < region.height * (hostRowPitch != 0 ? hostRowPitch : region.width)`
///         + `hostSlicePitch != 0 && hostSlicePitch % (hostRowPitch != 0 ? hostRowPitch : region.width) != 0`
///         + If the combination of `bufferOrigin`, `region`, `bufferRowPitch`, and `bufferSlicePitch` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in][bounds(bufferOrigin, region)] handle of the buffer object
    bool blockingWrite,                       ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOrigin,            ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOrigin,              ///< [in] 3D offset in the host region
    ur_rect_region_t region,                  ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t bufferRowPitch,                    ///< [in] length of each row in bytes in the buffer object
    size_t bufferSlicePitch,                  ///< [in] length of each 2D slice in bytes in the buffer object being
                                              ///< written
    size_t hostRowPitch,                      ///< [in] length of each row in bytes in the host memory region pointed by
                                              ///< src
    size_t hostSlicePitch,                    ///< [in] length of each 2D slice in bytes in the host memory region
                                              ///< pointed by src
    void *pSrc,                               ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] points to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy from a buffer object to another
///
/// @remarks
///   _Analogues_
///     - **clEnqueueCopyBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBufferSrc`
///         + `NULL == hBufferDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If `srcOffset + size` results in an out-of-bounds access.
///         + If `dstOffset + size` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferCopy(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBufferSrc,               ///< [in][bounds(srcOffset, size)] handle of the src buffer object
    ur_mem_handle_t hBufferDst,               ///< [in][bounds(dstOffset, size)] handle of the dest buffer object
    size_t srcOffset,                         ///< [in] offset into hBufferSrc to begin copying from
    size_t dstOffset,                         ///< [in] offset info hBufferDst to begin copying into
    size_t size,                              ///< [in] size in bytes of data being copied
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy a 2D or 3D rectangular region from one
///        buffer object to another
///
/// @remarks
///   _Analogues_
///     - **clEnqueueCopyBufferRect**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBufferSrc`
///         + `NULL == hBufferDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `region.width == 0 || region.height == 0 || region.depth == 0`
///         + `srcRowPitch != 0 && srcRowPitch < region.width`
///         + `dstRowPitch != 0 && dstRowPitch < region.width`
///         + `srcSlicePitch != 0 && srcSlicePitch < region.height * (srcRowPitch != 0 ? srcRowPitch : region.width)`
///         + `srcSlicePitch != 0 && srcSlicePitch % (srcRowPitch != 0 ? srcRowPitch : region.width) != 0`
///         + `dstSlicePitch != 0 && dstSlicePitch < region.height * (dstRowPitch != 0 ? dstRowPitch : region.width)`
///         + `dstSlicePitch != 0 && dstSlicePitch % (dstRowPitch != 0 ? dstRowPitch : region.width) != 0`
///         + If the combination of `srcOrigin`, `region`, `srcRowPitch`, and `srcSlicePitch` results in an out-of-bounds access.
///         + If the combination of `dstOrigin`, `region`, `dstRowPitch`, and `dstSlicePitch` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBufferSrc,               ///< [in][bounds(srcOrigin, region)] handle of the source buffer object
    ur_mem_handle_t hBufferDst,               ///< [in][bounds(dstOrigin, region)] handle of the dest buffer object
    ur_rect_offset_t srcOrigin,               ///< [in] 3D offset in the source buffer
    ur_rect_offset_t dstOrigin,               ///< [in] 3D offset in the destination buffer
    ur_rect_region_t region,                  ///< [in] source 3D rectangular region descriptor: width, height, depth
    size_t srcRowPitch,                       ///< [in] length of each row in bytes in the source buffer object
    size_t srcSlicePitch,                     ///< [in] length of each 2D slice in bytes in the source buffer object
    size_t dstRowPitch,                       ///< [in] length of each row in bytes in the destination buffer object
    size_t dstSlicePitch,                     ///< [in] length of each 2D slice in bytes in the destination buffer object
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to fill a buffer object with a pattern of a given
///        size
///
/// @remarks
///   _Analogues_
///     - **clEnqueueFillBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPattern`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `patternSize == 0 || size == 0`
///         + `patternSize > size`
///         + `(patternSize & (patternSize - 1)) != 0`
///         + `size % patternSize != 0`
///         + `offset % patternSize != 0`
///         + If `offset + size` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in][bounds(offset, size)] handle of the buffer object
    const void *pPattern,                     ///< [in] pointer to the fill pattern
    size_t patternSize,                       ///< [in] size in bytes of the pattern
    size_t offset,                            ///< [in] offset into the buffer
    size_t size,                              ///< [in] fill size in bytes, must be a multiple of patternSize
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to read from an image or image array object to host
///        memory
///
/// @details
///     - Input parameter blockingRead indicates if the read is blocking or
///       non-blocking.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueReadImage**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hImage`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `region.width == 0 || region.height == 0 || region.depth == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemImageRead(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hImage,                   ///< [in][bounds(origin, region)] handle of the image object
    bool blockingRead,                        ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t origin,                  ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_region_t region,                  ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                              ///< image
    size_t rowPitch,                          ///< [in] length of each row in bytes
    size_t slicePitch,                        ///< [in] length of each 2D slice of the 3D image
    void *pDst,                               ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to write an image or image array object from host
///        memory
///
/// @details
///     - Input parameter blockingWrite indicates if the write is blocking or
///       non-blocking.
///
/// @remarks
///   _Analogues_
///     - **clEnqueueWriteImage**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hImage`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `region.width == 0 || region.height == 0 || region.depth == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hImage,                   ///< [in][bounds(origin, region)] handle of the image object
    bool blockingWrite,                       ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t origin,                  ///< [in] defines the (x,y,z) offset in pixels in the 1D, 2D, or 3D image
    ur_rect_region_t region,                  ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                              ///< image
    size_t rowPitch,                          ///< [in] length of each row in bytes
    size_t slicePitch,                        ///< [in] length of each 2D slice of the 3D image
    void *pSrc,                               ///< [in] pointer to host memory where image is to be read into
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy from an image object to another
///
/// @remarks
///   _Analogues_
///     - **clEnqueueCopyImage**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hImageSrc`
///         + `NULL == hImageDst`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `region.width == 0 || region.height == 0 || region.depth == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemImageCopy(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hImageSrc,                ///< [in][bounds(srcOrigin, region)] handle of the src image object
    ur_mem_handle_t hImageDst,                ///< [in][bounds(dstOrigin, region)] handle of the dest image object
    ur_rect_offset_t srcOrigin,               ///< [in] defines the (x,y,z) offset in pixels in the source 1D, 2D, or 3D
                                              ///< image
    ur_rect_offset_t dstOrigin,               ///< [in] defines the (x,y,z) offset in pixels in the destination 1D, 2D,
                                              ///< or 3D image
    ur_rect_region_t region,                  ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                              ///< image
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Map flags
typedef uint32_t ur_map_flags_t;
typedef enum ur_map_flag_t {
    UR_MAP_FLAG_READ = UR_BIT(0),                    ///< Map for read access
    UR_MAP_FLAG_WRITE = UR_BIT(1),                   ///< Map for write access
    UR_MAP_FLAG_WRITE_INVALIDATE_REGION = UR_BIT(2), ///< Map for discard_write access
    /// @cond
    UR_MAP_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_map_flag_t;
/// @brief Bit Mask for validating ur_map_flags_t
#define UR_MAP_FLAGS_MASK 0xfffffff8

///////////////////////////////////////////////////////////////////////////////
/// @brief Map flags
typedef uint32_t ur_usm_migration_flags_t;
typedef enum ur_usm_migration_flag_t {
    UR_USM_MIGRATION_FLAG_DEFAULT = UR_BIT(0), ///< Default migration TODO: Add more enums!
    /// @cond
    UR_USM_MIGRATION_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_usm_migration_flag_t;
/// @brief Bit Mask for validating ur_usm_migration_flags_t
#define UR_USM_MIGRATION_FLAGS_MASK 0xfffffffe

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to map a region of the buffer object into the host
///        address space and return a pointer to the mapped region
///
/// @details
///     - Input parameter blockingMap indicates if the map is blocking or
///       non-blocking.
///     - Currently, no direct support in Level Zero. Implemented as a shared
///       allocation followed by copying on discrete GPU
///     - TODO: add a driver function in Level Zero?
///
/// @remarks
///   _Analogues_
///     - **clEnqueueMapBuffer**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_MAP_FLAGS_MASK & mapFlags`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppRetMap`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If `offset + size` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hBuffer,                  ///< [in][bounds(offset, size)] handle of the buffer object
    bool blockingMap,                         ///< [in] indicates blocking (true), non-blocking (false)
    ur_map_flags_t mapFlags,                  ///< [in] flags for read, write, readwrite mapping
    size_t offset,                            ///< [in] offset in bytes of the buffer region being mapped
    size_t size,                              ///< [in] size in bytes of the buffer region being mapped
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent,               ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
    void **ppRetMap                           ///< [out] return mapped pointer.  TODO: move it before
                                              ///< numEventsInWaitList?
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to unmap a previously mapped region of a memory
///        object
///
/// @remarks
///   _Analogues_
///     - **clEnqueueUnmapMemObject**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMappedPtr`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueMemUnmap(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_mem_handle_t hMem,                     ///< [in] handle of the memory (buffer or image) object
    void *pMappedPtr,                         ///< [in] mapped host address
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to fill USM memory.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///         + `NULL == pPattern`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `patternSize == 0 || size == 0`
///         + `patternSize > size`
///         + `(patternSize & (patternSize - 1)) != 0`
///         + `size % patternSize != 0`
///         + If `size` is higher than the allocation size of `ptr`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMFill(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    void *pMem,                               ///< [in][bounds(0, size)] pointer to USM memory object
    size_t patternSize,                       ///< [in] the size in bytes of the pattern. Must be a power of 2 and less
                                              ///< than or equal to width.
    const void *pPattern,                     ///< [in] pointer with the bytes of the pattern to set.
    size_t size,                              ///< [in] size in bytes to be set. Must be a multiple of patternSize.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy USM memory
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + If `size` is higher than the allocation size of `pSrc` or `pDst`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMMemcpy(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    bool blocking,                            ///< [in] blocking or non-blocking copy
    void *pDst,                               ///< [in][bounds(0, size)] pointer to the destination USM memory object
    const void *pSrc,                         ///< [in][bounds(0, size)] pointer to the source USM memory object
    size_t size,                              ///< [in] size in bytes to be copied
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to prefetch USM memory
///
/// @details
///     - Prefetching may not be supported for all devices or allocation types.
///       If memory prefetching is not supported, the prefetch hint will be
///       ignored.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_MIGRATION_FLAGS_MASK & flags`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + If `size` is higher than the allocation size of `pMem`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMPrefetch(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    const void *pMem,                         ///< [in][bounds(0, size)] pointer to the USM memory object
    size_t size,                              ///< [in] size in bytes to be fetched
    ur_usm_migration_flags_t flags,           ///< [in] USM prefetch flags
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
                                              ///< command does not wait on any event to complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to set USM memory advice
///
/// @details
///     - Not all memory advice hints may be supported for all devices or
///       allocation types. If a memory advice hint is not supported, it will be
///       ignored.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_ADVICE_FLAGS_MASK & advice`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + If `size` is higher than the allocation size of `pMem`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMAdvise(
    ur_queue_handle_t hQueue,     ///< [in] handle of the queue object
    const void *pMem,             ///< [in][bounds(0, size)] pointer to the USM memory object
    size_t size,                  ///< [in] size in bytes to be advised
    ur_usm_advice_flags_t advice, ///< [in] USM memory advice
    ur_event_handle_t *phEvent    ///< [out][optional] return an event object that identifies this particular
                                  ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to fill 2D USM memory.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///         + `NULL == pPattern`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `pitch == 0`
///         + `pitch < width`
///         + `patternSize == 0`
///         + `patternSize > width * height`
///         + `patternSize != 0 && ((patternSize & (patternSize - 1)) != 0)`
///         + `width == 0`
///         + `height == 0`
///         + `width * height % patternSize != 0`
///         + If `pitch * height` is higher than the allocation size of `pMem`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMFill2D(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue to submit to.
    void *pMem,                               ///< [in][bounds(0, pitch * height)] pointer to memory to be filled.
    size_t pitch,                             ///< [in] the total width of the destination memory including padding.
    size_t patternSize,                       ///< [in] the size in bytes of the pattern. Must be a power of 2 and less
                                              ///< than or equal to width.
    const void *pPattern,                     ///< [in] pointer with the bytes of the pattern to set.
    size_t width,                             ///< [in] the width in bytes of each row to fill. Must be a multiple of
                                              ///< patternSize.
    size_t height,                            ///< [in] the height of the columns to fill.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< kernel execution instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to copy 2D USM memory.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `srcPitch == 0`
///         + `dstPitch == 0`
///         + `srcPitch < width`
///         + `dstPitch < width`
///         + `height == 0`
///         + If `srcPitch * height` is higher than the allocation size of `pSrc`
///         + If `dstPitch * height` is higher than the allocation size of `pDst`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMMemcpy2D(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue to submit to.
    bool blocking,                            ///< [in] indicates if this operation should block the host.
    void *pDst,                               ///< [in][bounds(0, dstPitch * height)] pointer to memory where data will
                                              ///< be copied.
    size_t dstPitch,                          ///< [in] the total width of the source memory including padding.
    const void *pSrc,                         ///< [in][bounds(0, srcPitch * height)] pointer to memory to be copied.
    size_t srcPitch,                          ///< [in] the total width of the source memory including padding.
    size_t width,                             ///< [in] the width in bytes of each row to be copied.
    size_t height,                            ///< [in] the height of columns to be copied.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< kernel execution instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to write data from the host to device global
///        variable.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == name`
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue to submit to.
    ur_program_handle_t hProgram,             ///< [in] handle of the program containing the device global variable.
    const char *name,                         ///< [in] the unique identifier for the device global variable.
    bool blockingWrite,                       ///< [in] indicates if this operation should block.
    size_t count,                             ///< [in] the number of bytes to copy.
    size_t offset,                            ///< [in] the byte offset into the device global variable to start copying.
    const void *pSrc,                         ///< [in] pointer to where the data must be copied from.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list.
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< kernel execution instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to read data from a device global variable to the
///        host.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == name`
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue to submit to.
    ur_program_handle_t hProgram,             ///< [in] handle of the program containing the device global variable.
    const char *name,                         ///< [in] the unique identifier for the device global variable.
    bool blockingRead,                        ///< [in] indicates if this operation should block.
    size_t count,                             ///< [in] the number of bytes to copy.
    size_t offset,                            ///< [in] the byte offset into the device global variable to start copying.
    void *pDst,                               ///< [in] pointer to where the data must be copied to.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list.
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< kernel execution instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to read from a pipe to the host.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pipe_symbol`
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueReadHostPipe(
    ur_queue_handle_t hQueue,                 ///< [in] a valid host command-queue in which the read command
                                              ///< will be queued. hQueue and hProgram must be created with the same
                                              ///< UR context.
    ur_program_handle_t hProgram,             ///< [in] a program object with a successfully built executable.
    const char *pipe_symbol,                  ///< [in] the name of the program scope pipe global variable.
    bool blocking,                            ///< [in] indicate if the read operation is blocking or non-blocking.
    void *pDst,                               ///< [in] a pointer to buffer in host memory that will hold resulting data
                                              ///< from pipe.
    size_t size,                              ///< [in] size of the memory region to read, in bytes.
    uint32_t numEventsInWaitList,             ///< [in] number of events in the wait list.
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the host pipe read.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait event.
    ur_event_handle_t *phEvent                ///< [out][optional] returns an event object that identifies this read
                                              ///< command
                                              ///< and can be used to query or queue a wait for this command to complete.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to write data from the host to a pipe.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pipe_symbol`
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueWriteHostPipe(
    ur_queue_handle_t hQueue,                 ///< [in] a valid host command-queue in which the write command
                                              ///< will be queued. hQueue and hProgram must be created with the same
                                              ///< UR context.
    ur_program_handle_t hProgram,             ///< [in] a program object with a successfully built executable.
    const char *pipe_symbol,                  ///< [in] the name of the program scope pipe global variable.
    bool blocking,                            ///< [in] indicate if the read and write operations are blocking or
                                              ///< non-blocking.
    void *pSrc,                               ///< [in] a pointer to buffer in host memory that holds data to be written
                                              ///< to the host pipe.
    size_t size,                              ///< [in] size of the memory region to read or write, in bytes.
    uint32_t numEventsInWaitList,             ///< [in] number of events in the wait list.
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the host pipe write.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait event.
    ur_event_handle_t *phEvent                ///< [out][optional] returns an event object that identifies this write command
                                              ///< and can be used to query or queue a wait for this command to complete.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Bindless Images Extension APIs
#if !defined(__GNUC__)
#pragma region bindless images(experimental)
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of bindless image
typedef struct ur_exp_image_handle_t_ *ur_exp_image_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of bindless image memory
typedef struct ur_exp_image_mem_handle_t_ *ur_exp_image_mem_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of interop memory
typedef struct ur_exp_interop_mem_handle_t_ *ur_exp_interop_mem_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of interop semaphore
typedef struct ur_exp_interop_semaphore_handle_t_ *ur_exp_interop_semaphore_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Dictates the type of memory copy.
typedef uint32_t ur_exp_image_copy_flags_t;
typedef enum ur_exp_image_copy_flag_t {
    UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE = UR_BIT(0),   ///< Host to device
    UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST = UR_BIT(1),   ///< Device to host
    UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE = UR_BIT(2), ///< Device to device
    /// @cond
    UR_EXP_IMAGE_COPY_FLAG_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_exp_image_copy_flag_t;
/// @brief Bit Mask for validating ur_exp_image_copy_flags_t
#define UR_EXP_IMAGE_COPY_FLAGS_MASK 0xfffffff8

///////////////////////////////////////////////////////////////////////////////
/// @brief File descriptor
typedef struct ur_exp_file_descriptor_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    int fd;                    ///< [in] A file descriptor used for Linux and & MacOS operating systems.

} ur_exp_file_descriptor_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Windows specific file handle
typedef struct ur_exp_win32_handle_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    void *handle;              ///< [in] A win32 file handle.

} ur_exp_win32_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Describes mipmap sampler properties
///
/// @details
///     - Specify these properties in ::urSamplerCreate via ::ur_sampler_desc_t
///       as part of a `pNext` chain.
typedef struct ur_exp_sampler_mip_properties_t {
    ur_structure_type_t stype;              ///< [in] type of this structure, must be
                                            ///< ::UR_STRUCTURE_TYPE_EXP_SAMPLER_MIP_PROPERTIES
    void *pNext;                            ///< [in,out][optional] pointer to extension-specific structure
    float minMipmapLevelClamp;              ///< [in] minimum mipmap level from which we can sample, minimum value
                                            ///< being 0
    float maxMipmapLevelClamp;              ///< [in] maximum mipmap level from which we can sample, maximum value
                                            ///< being the number of levels
    float maxAnisotropy;                    ///< [in] anisotropic ratio used when samplling the mipmap with anisotropic
                                            ///< filtering
    ur_sampler_filter_mode_t mipFilterMode; ///< [in] mipmap filter mode used for filtering between mipmap levels

} ur_exp_sampler_mip_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Describes unique sampler addressing mode per dimension
///
/// @details
///     - Specify these properties in ::urSamplerCreate via ::ur_sampler_desc_t
///       as part of a `pNext` chain.
typedef struct ur_exp_sampler_addr_modes_t {
    ur_structure_type_t stype;                 ///< [in] type of this structure, must be
                                               ///< ::UR_STRUCTURE_TYPE_EXP_SAMPLER_ADDR_MODES
    void *pNext;                               ///< [in,out][optional] pointer to extension-specific structure
    ur_sampler_addressing_mode_t addrModes[3]; ///< [in] Specify the address mode of the sampler per dimension

} ur_exp_sampler_addr_modes_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Describes an interop memory resource descriptor
typedef struct ur_exp_interop_mem_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_EXP_INTEROP_MEM_DESC
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure

} ur_exp_interop_mem_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Describes an interop semaphore resource descriptor
typedef struct ur_exp_interop_semaphore_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_EXP_INTEROP_SEMAPHORE_DESC
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure

} ur_exp_interop_semaphore_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief USM allocate pitched memory
///
/// @details
///     - This function must support memory pooling.
///     - If pUSMDesc is not NULL and pUSMDesc->pool is not NULL the allocation
///       will be served from a specified memory pool.
///     - Otherwise, the behavior is implementation-defined.
///     - Allocations served from different memory pools must be isolated and
///       must not reside on the same page.
///     - Any flags/hints passed through pUSMDesc only affect the single
///       allocation.
///     - See also ::ur_usm_host_desc_t.
///     - See also ::ur_usm_device_desc_t.
///
/// @remarks
///   _Analogues_
///     - **cuMemAllocPitch**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `NULL != pUSMDesc && ::UR_USM_ADVICE_FLAGS_MASK & pUSMDesc->hints`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == ppMem`
///         + `NULL == pResultPitch`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///         + `pUSMDesc && pUSMDesc->align != 0 && ((pUSMDesc->align & (pUSMDesc->align-1)) != 0)`
///         + If `align` is greater that the size of the largest data type supported by `hDevice`.
///     - ::UR_RESULT_ERROR_INVALID_USM_SIZE
///         + `widthInBytes == 0`
///         + `size` is greater than ::UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE.
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///         + If `UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT` and `UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT` are both false.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urUSMPitchedAllocExp(
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    ur_device_handle_t hDevice,    ///< [in] handle of the device object
    const ur_usm_desc_t *pUSMDesc, ///< [in][optional] Pointer to USM memory allocation descriptor.
    ur_usm_pool_handle_t pool,     ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t widthInBytes,           ///< [in] width in bytes of the USM memory object to be allocated
    size_t height,                 ///< [in] height of the USM memory object to be allocated
    size_t elementSizeBytes,       ///< [in] size in bytes of an element in the allocation
    void **ppMem,                  ///< [out] pointer to USM shared memory object
    size_t *pResultPitch           ///< [out] pitch of the allocation
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy bindless unsampled image handles
///
/// @remarks
///   _Analogues_
///     - **cuSurfObjectDestroy**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hImage`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesUnsampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_handle_t hImage  ///< [in] pointer to handle of image object to destroy
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy bindless sampled image handles
///
/// @remarks
///   _Analogues_
///     - **cuTexObjectDestroy**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hImage`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesSampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    ur_exp_image_handle_t hImage  ///< [in] pointer to handle of image object to destroy
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Allocate memory for bindless images
///
/// @remarks
///   _Analogues_
///     - **cuArray3DCreate**
///     - **cuMipmappedArrayCreate**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == phImageMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///         + `pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesImageAllocateExp(
    ur_context_handle_t hContext,          ///< [in] handle of the context object
    ur_device_handle_t hDevice,            ///< [in] handle of the device object
    const ur_image_format_t *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc,     ///< [in] pointer to image description
    ur_exp_image_mem_handle_t *phImageMem  ///< [out] pointer to handle of image memory allocated
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Free memory for bindless images
///
/// @remarks
///   _Analogues_
///     - **cuArrayDestroy**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hImageMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesImageFreeExp(
    ur_context_handle_t hContext,       ///< [in] handle of the context object
    ur_device_handle_t hDevice,         ///< [in] handle of the device object
    ur_exp_image_mem_handle_t hImageMem ///< [in] handle of image memory to be freed
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a bindless unsampled image handle
///
/// @remarks
///   _Analogues_
///     - **cuSurfObjectCreate**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hImageMem`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == phMem`
///         + `NULL == phImage`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///         + `pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesUnsampledImageCreateExp(
    ur_context_handle_t hContext,          ///< [in] handle of the context object
    ur_device_handle_t hDevice,            ///< [in] handle of the device object
    ur_exp_image_mem_handle_t hImageMem,   ///< [in] handle to memory from which to create the image
    const ur_image_format_t *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc,     ///< [in] pointer to image description
    ur_mem_handle_t *phMem,                ///< [out] pointer to handle of image object created
    ur_exp_image_handle_t *phImage         ///< [out] pointer to handle of image object created
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a bindless sampled image handle
///
/// @remarks
///   _Analogues_
///     - **cuTexObjectCreate**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hImageMem`
///         + `NULL == hSampler`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == phMem`
///         + `NULL == phImage`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///         + `pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_SAMPLER
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesSampledImageCreateExp(
    ur_context_handle_t hContext,          ///< [in] handle of the context object
    ur_device_handle_t hDevice,            ///< [in] handle of the device object
    ur_exp_image_mem_handle_t hImageMem,   ///< [in] handle to memory from which to create the image
    const ur_image_format_t *pImageFormat, ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc,     ///< [in] pointer to image description
    ur_sampler_handle_t hSampler,          ///< [in] sampler to be used
    ur_mem_handle_t *phMem,                ///< [out] pointer to handle of image object created
    ur_exp_image_handle_t *phImage         ///< [out] pointer to handle of image object created
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Copy image data Host to Device or Device to Host
///
/// @remarks
///   _Analogues_
///     - **cuMemcpyHtoAAsync**
///     - **cuMemcpyAtoHAsync**
///     - **cuMemcpy2DAsync**
///     - **cuMemcpy3DAsync**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///         + `NULL == pSrc`
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EXP_IMAGE_COPY_FLAGS_MASK & imageCopyFlags`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///         + `pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesImageCopyExp(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    void *pDst,                               ///< [in] location the data will be copied to
    void *pSrc,                               ///< [in] location the data will be copied from
    const ur_image_format_t *pImageFormat,    ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc,        ///< [in] pointer to image description
    ur_exp_image_copy_flags_t imageCopyFlags, ///< [in] flags describing copy direction e.g. H2D or D2H
    ur_rect_offset_t srcOffset,               ///< [in] defines the (x,y,z) source offset in pixels in the 1D, 2D, or 3D
                                              ///< image
    ur_rect_offset_t dstOffset,               ///< [in] defines the (x,y,z) destination offset in pixels in the 1D, 2D,
                                              ///< or 3D image
    ur_rect_region_t copyExtent,              ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                              ///< region to copy
    ur_rect_region_t hostExtent,              ///< [in] defines the (width, height, depth) in pixels of the 1D, 2D, or 3D
                                              ///< region on the host
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before this command can be executed.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
                                              ///< previously enqueued commands
                                              ///< must be complete.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Query an image memory handle for specific properties
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hImageMem`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_IMAGE_INFO_DEPTH < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesImageGetInfoExp(
    ur_exp_image_mem_handle_t hImageMem, ///< [in] handle to the image memory
    ur_image_info_t propName,            ///< [in] queried info name
    void *pPropValue,                    ///< [out][optional] returned query value
    size_t *pPropSizeRet                 ///< [out][optional] returned query value size
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve individual image from mipmap
///
/// @remarks
///   _Analogues_
///     - **cuMipmappedArrayGetLevel**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hImageMem`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phImageMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesMipmapGetLevelExp(
    ur_context_handle_t hContext,         ///< [in] handle of the context object
    ur_device_handle_t hDevice,           ///< [in] handle of the device object
    ur_exp_image_mem_handle_t hImageMem,  ///< [in] memory handle to the mipmap image
    uint32_t mipmapLevel,                 ///< [in] requested level of the mipmap
    ur_exp_image_mem_handle_t *phImageMem ///< [out] returning memory handle to the individual image
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Free mipmap memory for bindless images
///
/// @remarks
///   _Analogues_
///     - **cuMipmappedArrayDestroy**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesMipmapFreeExp(
    ur_context_handle_t hContext,  ///< [in] handle of the context object
    ur_device_handle_t hDevice,    ///< [in] handle of the device object
    ur_exp_image_mem_handle_t hMem ///< [in] handle of image memory to be freed
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Import external memory in the form of a file descriptor
///
/// @remarks
///   _Analogues_
///     - **cuImportExternalMemory**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pInteropMemDesc`
///         + `NULL == phInteropMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesImportOpaqueFDExp(
    ur_context_handle_t hContext,               ///< [in] handle of the context object
    ur_device_handle_t hDevice,                 ///< [in] handle of the device object
    size_t size,                                ///< [in] size of the external memory
    ur_exp_interop_mem_desc_t *pInteropMemDesc, ///< [in] the interop memory descriptor
    ur_exp_interop_mem_handle_t *phInteropMem   ///< [out] interop memory handle to the external memory
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Map an interop memory handle to an image memory handle
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hInteropMem`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pImageFormat`
///         + `NULL == pImageDesc`
///         + `NULL == phImageMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR
///         + `pImageDesc && UR_MEM_TYPE_IMAGE1D_BUFFER < pImageDesc->type`
///     - ::UR_RESULT_ERROR_INVALID_IMAGE_SIZE
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesMapExternalArrayExp(
    ur_context_handle_t hContext,            ///< [in] handle of the context object
    ur_device_handle_t hDevice,              ///< [in] handle of the device object
    const ur_image_format_t *pImageFormat,   ///< [in] pointer to image format specification
    const ur_image_desc_t *pImageDesc,       ///< [in] pointer to image description
    ur_exp_interop_mem_handle_t hInteropMem, ///< [in] interop memory handle to the external memory
    ur_exp_image_mem_handle_t *phImageMem    ///< [out] image memory handle to the externally allocated memory
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy interop memory
///
/// @remarks
///   _Analogues_
///     - **cuDestroyExternalMemory**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hInteropMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesReleaseInteropExp(
    ur_context_handle_t hContext,           ///< [in] handle of the context object
    ur_device_handle_t hDevice,             ///< [in] handle of the device object
    ur_exp_interop_mem_handle_t hInteropMem ///< [in] handle of interop memory to be freed
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Import an external semaphore in the form of a file descriptor
///
/// @remarks
///   _Analogues_
///     - **cuImportExternalSemaphore**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pInteropSemaphoreDesc`
///         + `NULL == phInteropSemaphore`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesImportExternalSemaphoreOpaqueFDExp(
    ur_context_handle_t hContext,                           ///< [in] handle of the context object
    ur_device_handle_t hDevice,                             ///< [in] handle of the device object
    ur_exp_interop_semaphore_desc_t *pInteropSemaphoreDesc, ///< [in] the interop semaphore descriptor
    ur_exp_interop_semaphore_handle_t *phInteropSemaphore   ///< [out] interop semaphore handle to the external semaphore
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy the external semaphore handle
///
/// @remarks
///   _Analogues_
///     - **cuDestroyExternalSemaphore**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///         + `NULL == hInteropSemaphore`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesDestroyExternalSemaphoreExp(
    ur_context_handle_t hContext,                       ///< [in] handle of the context object
    ur_device_handle_t hDevice,                         ///< [in] handle of the device object
    ur_exp_interop_semaphore_handle_t hInteropSemaphore ///< [in] handle of interop semaphore to be destroyed
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Instruct the queue with a non-blocking wait on an external semaphore
///
/// @remarks
///   _Analogues_
///     - **cuWaitExternalSemaphoresAsync**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hSemaphore`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesWaitExternalSemaphoreExp(
    ur_queue_handle_t hQueue,                     ///< [in] handle of the queue object
    ur_exp_interop_semaphore_handle_t hSemaphore, ///< [in] interop semaphore handle
    uint32_t numEventsInWaitList,                 ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList,     ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                  ///< events that must be complete before this command can be executed.
                                                  ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
                                                  ///< previously enqueued commands
                                                  ///< must be complete.
    ur_event_handle_t *phEvent                    ///< [out][optional] return an event object that identifies this particular
                                                  ///< command instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Instruct the queue to signal the external semaphore handle once all
///        previous commands have completed execution
///
/// @remarks
///   _Analogues_
///     - **cuSignalExternalSemaphoresAsync**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hSemaphore`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesSignalExternalSemaphoreExp(
    ur_queue_handle_t hQueue,                     ///< [in] handle of the queue object
    ur_exp_interop_semaphore_handle_t hSemaphore, ///< [in] interop semaphore handle
    uint32_t numEventsInWaitList,                 ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList,     ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                  ///< events that must be complete before this command can be executed.
                                                  ///< If nullptr, the numEventsInWaitList must be 0, indicating that all
                                                  ///< previously enqueued commands
                                                  ///< must be complete.
    ur_event_handle_t *phEvent                    ///< [out][optional] return an event object that identifies this particular
                                                  ///< command instance.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime Experimental APIs for Command-Buffers
#if !defined(__GNUC__)
#pragma region command buffer(experimental)
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Command-buffer query information type
typedef enum ur_exp_command_buffer_info_t {
    UR_EXP_COMMAND_BUFFER_INFO_REFERENCE_COUNT = 0, ///< [uint32_t] Reference count of the command-buffer object.
                                                    ///< The reference count returned should be considered immediately stale.
                                                    ///< It is unsuitable for general use in applications. This feature is
                                                    ///< provided for identifying memory leaks.
    /// @cond
    UR_EXP_COMMAND_BUFFER_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_exp_command_buffer_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Command-buffer command query information type
typedef enum ur_exp_command_buffer_command_info_t {
    UR_EXP_COMMAND_BUFFER_COMMAND_INFO_REFERENCE_COUNT = 0, ///< [uint32_t] Reference count of the command-buffer object.
                                                            ///< The reference count returned should be considered immediately stale.
                                                            ///< It is unsuitable for general use in applications. This feature is
                                                            ///< provided for identifying memory leaks.
    /// @cond
    UR_EXP_COMMAND_BUFFER_COMMAND_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_exp_command_buffer_command_info_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef UR_COMMAND_BUFFER_EXTENSION_STRING_EXP
/// @brief The extension string which defines support for command-buffers which
///        is returned when querying device extensions.
#define UR_COMMAND_BUFFER_EXTENSION_STRING_EXP "ur_exp_command_buffer"
#endif // UR_COMMAND_BUFFER_EXTENSION_STRING_EXP

///////////////////////////////////////////////////////////////////////////////
/// @brief Command-Buffer Descriptor Type
typedef struct ur_exp_command_buffer_desc_t {
    ur_structure_type_t stype; ///< [in] type of this structure, must be
                               ///< ::UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC
    const void *pNext;         ///< [in][optional] pointer to extension-specific structure
    ur_bool_t isUpdatable;     ///< [in] Commands in a finalized command-buffer can be updated.

} ur_exp_command_buffer_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Descriptor type for updating a kernel command memobj argument.
typedef struct ur_exp_command_buffer_update_memobj_arg_desc_t {
    ur_structure_type_t stype;                             ///< [in] type of this structure, must be
                                                           ///< ::UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_MEMOBJ_ARG_DESC
    const void *pNext;                                     ///< [in][optional] pointer to extension-specific structure
    uint32_t argIndex;                                     ///< [in] Argument index.
    const ur_kernel_arg_mem_obj_properties_t *pProperties; ///< [in][optinal] Pointer to memory object properties.
    ur_mem_handle_t hNewMemObjArg;                         ///< [in][optional] Handle of memory object to set at argument index.

} ur_exp_command_buffer_update_memobj_arg_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Descriptor type for updating a kernel command pointer argument.
typedef struct ur_exp_command_buffer_update_pointer_arg_desc_t {
    ur_structure_type_t stype;                             ///< [in] type of this structure, must be
                                                           ///< ::UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC
    const void *pNext;                                     ///< [in][optional] pointer to extension-specific structure
    uint32_t argIndex;                                     ///< [in] Argument index.
    const ur_kernel_arg_pointer_properties_t *pProperties; ///< [in][optinal] Pointer to USM pointer properties.
    const void *pNewPointerArg;                            ///< [in][optional] USM pointer to memory location holding the argument
                                                           ///< value to set at argument index.

} ur_exp_command_buffer_update_pointer_arg_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Descriptor type for updating a kernel command value argument.
typedef struct ur_exp_command_buffer_update_value_arg_desc_t {
    ur_structure_type_t stype;                           ///< [in] type of this structure, must be
                                                         ///< ::UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC
    const void *pNext;                                   ///< [in][optional] pointer to extension-specific structure
    uint32_t argIndex;                                   ///< [in] Argument index.
    uint32_t argSize;                                    ///< [in] Argument size.
    const ur_kernel_arg_value_properties_t *pProperties; ///< [in][optinal] Pointer to value properties.
    const void *pNewValueArg;                            ///< [in][optional] Argument value representing matching kernel arg type to
                                                         ///< set at argument index.

} ur_exp_command_buffer_update_value_arg_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Descriptor type for updating kernel command execution info.
typedef struct ur_exp_command_buffer_update_exec_info_desc_t {
    ur_structure_type_t stype;                           ///< [in] type of this structure, must be
                                                         ///< ::UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_EXEC_INFO_DESC
    const void *pNext;                                   ///< [in][optional] pointer to extension-specific structure
    ur_kernel_exec_info_t propName;                      ///< [in] Name of execution attribute.
    size_t propSize;                                     ///< [in] Size of execution attribute.
    const ur_kernel_exec_info_properties_t *pProperties; ///< [in][optional] Pointer to execution info properties.
    const void *pNewExecInfo;                            ///< [in] Pointer to memory location holding the execution info value.

} ur_exp_command_buffer_update_exec_info_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Descriptor type for updating a kernel launch command.
typedef struct ur_exp_command_buffer_update_kernel_launch_desc_t {
    ur_structure_type_t stype;                                                 ///< [in] type of this structure, must be
                                                                               ///< ::UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC
    const void *pNext;                                                         ///< [in][optional] pointer to extension-specific structure
    uint32_t numNewMemObjArgs;                                                 ///< [in] Length of pNewMemObjArgList.
    uint32_t numNewPointerArgs;                                                ///< [in] Length of pNewPointerArgList.
    uint32_t numNewValueArgs;                                                  ///< [in] Length of pNewValueArgList.
    uint32_t numNewExecInfos;                                                  ///< [in] Length of pNewExecInfoList.
    uint32_t newWorkDim;                                                       ///< [in] Number of work dimensions in the kernel ND-range, from 1-3.
    const ur_exp_command_buffer_update_memobj_arg_desc_t *pNewMemObjArgList;   ///< [in][optional][range(0, numNewMemObjArgs)] An array describing the new
                                                                               ///< kernel mem obj arguments for the command.
    const ur_exp_command_buffer_update_pointer_arg_desc_t *pNewPointerArgList; ///< [in][optional][range(0, numNewPointerArgs)] An array describing the
                                                                               ///< new kernel pointer arguments for the command.
    const ur_exp_command_buffer_update_value_arg_desc_t *pNewValueArgList;     ///< [in][optional][range(0, numNewValueArgs)] An array describing the new
                                                                               ///< kernel value arguments for the command.
    const ur_exp_command_buffer_update_exec_info_desc_t *pNewExecInfoList;     ///< [in][optional][range(0, numNewExecInfos)] An array describing the
                                                                               ///< execution info objects for the command.
    size_t *pNewGlobalWorkOffset;                                              ///< [in][optional][range(0, newWorkDim)] Array of newWorkDim unsigned
                                                                               ///< values that describe the offset used to calculate the global ID.
    size_t *pNewGlobalWorkSize;                                                ///< [in][optional][range(0, newWorkDim)] Array of newWorkDim unsigned
                                                                               ///< values that describe the number of global work-items.
    size_t *pNewLocalWorkSize;                                                 ///< [in][optional][range(0, newWorkDim)] Array of newWorkDim unsigned
                                                                               ///< values that describe the number of work-items that make up a
                                                                               ///< work-group. If nullptr, the runtime implementation will choose the
                                                                               ///< work-group size.

} ur_exp_command_buffer_update_kernel_launch_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief A value that identifies a command inside of a command-buffer, used for
///        defining dependencies between commands in the same command-buffer.
typedef uint32_t ur_exp_command_buffer_sync_point_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of Command-Buffer object
typedef struct ur_exp_command_buffer_handle_t_ *ur_exp_command_buffer_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of a Command-Buffer command
typedef struct ur_exp_command_buffer_command_handle_t_ *ur_exp_command_buffer_command_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a Command-Buffer object
///
/// @details
///     - Create a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///         + `NULL == hDevice`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_DEVICE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferCreateExp(
    ur_context_handle_t hContext,                           ///< [in] Handle of the context object.
    ur_device_handle_t hDevice,                             ///< [in] Handle of the device object.
    const ur_exp_command_buffer_desc_t *pCommandBufferDesc, ///< [in][optional] command-buffer descriptor.
    ur_exp_command_buffer_handle_t *phCommandBuffer         ///< [out] Pointer to command-Buffer handle.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Increment the command-buffer object's reference count.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferRetainExp(
    ur_exp_command_buffer_handle_t hCommandBuffer ///< [in] Handle of the command-buffer object.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the command-buffer object's reference count and delete the
///        command-buffer object if the reference count becomes zero.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferReleaseExp(
    ur_exp_command_buffer_handle_t hCommandBuffer ///< [in] Handle of the command-buffer object.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Stop recording on a command-buffer object such that no more commands
///        can be appended and make it ready to enqueue.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferFinalizeExp(
    ur_exp_command_buffer_handle_t hCommandBuffer ///< [in] Handle of the command-buffer object.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a kernel execution command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pGlobalWorkOffset`
///         + `NULL == pGlobalWorkSize`
///         + `NULL == pLocalWorkSize`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_KERNEL
///     - ::UR_RESULT_ERROR_INVALID_WORK_DIMENSION
///     - ::UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,                ///< [in] Handle of the command-buffer object.
    ur_kernel_handle_t hKernel,                                   ///< [in] Kernel to append.
    uint32_t workDim,                                             ///< [in] Dimension of the kernel execution.
    const size_t *pGlobalWorkOffset,                              ///< [in] Offset to use when executing kernel.
    const size_t *pGlobalWorkSize,                                ///< [in] Global work size to use when executing kernel.
    const size_t *pLocalWorkSize,                                 ///< [in] Local work size to use when executing kernel.
    uint32_t numSyncPointsInWaitList,                             ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *pSyncPoint,               ///< [out][optional] Sync point associated with this command.
    ur_exp_command_buffer_command_handle_t *phCommand             ///< [out][optional] Handle to this command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a USM memcpy command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + If `size` is higher than the allocation size of `pSrc` or `pDst`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferAppendUSMMemcpyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,                ///< [in] Handle of the command-buffer object.
    void *pDst,                                                   ///< [in] Location the data will be copied to.
    const void *pSrc,                                             ///< [in] The data to be copied.
    size_t size,                                                  ///< [in] The number of bytes to copy.
    uint32_t numSyncPointsInWaitList,                             ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *pSyncPoint                ///< [out][optional] Sync point associated with this command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a USM fill command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMemory`
///         + `NULL == pPattern`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `patternSize == 0 || size == 0`
///         + `patternSize > size`
///         + `(patternSize & (patternSize - 1)) != 0`
///         + `size % patternSize != 0`
///         + If `size` is higher than the allocation size of `ptr`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferAppendUSMFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,                ///< [in] handle of the command-buffer object.
    void *pMemory,                                                ///< [in] pointer to USM allocated memory to fill.
    const void *pPattern,                                         ///< [in] pointer to the fill pattern.
    size_t patternSize,                                           ///< [in] size in bytes of the pattern.
    size_t size,                                                  ///< [in] fill size in bytes, must be a multiple of patternSize.
    uint32_t numSyncPointsInWaitList,                             ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *pSyncPoint                ///< [out][optional] sync point associated with this command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a memory copy command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hSrcMem`
///         + `NULL == hDstMem`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,                ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hSrcMem,                                      ///< [in] The data to be copied.
    ur_mem_handle_t hDstMem,                                      ///< [in] The location the data will be copied to.
    size_t srcOffset,                                             ///< [in] Offset into the source memory.
    size_t dstOffset,                                             ///< [in] Offset into the destination memory
    size_t size,                                                  ///< [in] The number of bytes to be copied.
    uint32_t numSyncPointsInWaitList,                             ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *pSyncPoint                ///< [out][optional] Sync point associated with this command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a memory write command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferAppendMemBufferWriteExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,                ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hBuffer,                                      ///< [in] Handle of the buffer object.
    size_t offset,                                                ///< [in] Offset in bytes in the buffer object.
    size_t size,                                                  ///< [in] Size in bytes of data being written.
    const void *pSrc,                                             ///< [in] Pointer to host memory where data is to be written from.
    uint32_t numSyncPointsInWaitList,                             ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *pSyncPoint                ///< [out][optional] Sync point associated with this command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a memory read command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferAppendMemBufferReadExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,                ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hBuffer,                                      ///< [in] Handle of the buffer object.
    size_t offset,                                                ///< [in] Offset in bytes in the buffer object.
    size_t size,                                                  ///< [in] Size in bytes of data being written.
    void *pDst,                                                   ///< [in] Pointer to host memory where data is to be written to.
    uint32_t numSyncPointsInWaitList,                             ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *pSyncPoint                ///< [out][optional] Sync point associated with this command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a rectangular memory copy command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hSrcMem`
///         + `NULL == hDstMem`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferAppendMemBufferCopyRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,                ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hSrcMem,                                      ///< [in] The data to be copied.
    ur_mem_handle_t hDstMem,                                      ///< [in] The location the data will be copied to.
    ur_rect_offset_t srcOrigin,                                   ///< [in] Origin for the region of data to be copied from the source.
    ur_rect_offset_t dstOrigin,                                   ///< [in] Origin for the region of data to be copied to in the destination.
    ur_rect_region_t region,                                      ///< [in] The extents describing the region to be copied.
    size_t srcRowPitch,                                           ///< [in] Row pitch of the source memory.
    size_t srcSlicePitch,                                         ///< [in] Slice pitch of the source memory.
    size_t dstRowPitch,                                           ///< [in] Row pitch of the destination memory.
    size_t dstSlicePitch,                                         ///< [in] Slice pitch of the destination memory.
    uint32_t numSyncPointsInWaitList,                             ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *pSyncPoint                ///< [out][optional] Sync point associated with this command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a rectangular memory write command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pSrc`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferAppendMemBufferWriteRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,                ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hBuffer,                                      ///< [in] Handle of the buffer object.
    ur_rect_offset_t bufferOffset,                                ///< [in] 3D offset in the buffer.
    ur_rect_offset_t hostOffset,                                  ///< [in] 3D offset in the host region.
    ur_rect_region_t region,                                      ///< [in] 3D rectangular region descriptor: width, height, depth.
    size_t bufferRowPitch,                                        ///< [in] Length of each row in bytes in the buffer object.
    size_t bufferSlicePitch,                                      ///< [in] Length of each 2D slice in bytes in the buffer object being
                                                                  ///< written.
    size_t hostRowPitch,                                          ///< [in] Length of each row in bytes in the host memory region pointed to
                                                                  ///< by pSrc.
    size_t hostSlicePitch,                                        ///< [in] Length of each 2D slice in bytes in the host memory region
                                                                  ///< pointed to by pSrc.
    void *pSrc,                                                   ///< [in] Pointer to host memory where data is to be written from.
    uint32_t numSyncPointsInWaitList,                             ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *pSyncPoint                ///< [out][optional] Sync point associated with this command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a rectangular memory read command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pDst`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferAppendMemBufferReadRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,                ///< [in] Handle of the command-buffer object.
    ur_mem_handle_t hBuffer,                                      ///< [in] Handle of the buffer object.
    ur_rect_offset_t bufferOffset,                                ///< [in] 3D offset in the buffer.
    ur_rect_offset_t hostOffset,                                  ///< [in] 3D offset in the host region.
    ur_rect_region_t region,                                      ///< [in] 3D rectangular region descriptor: width, height, depth.
    size_t bufferRowPitch,                                        ///< [in] Length of each row in bytes in the buffer object.
    size_t bufferSlicePitch,                                      ///< [in] Length of each 2D slice in bytes in the buffer object being read.
    size_t hostRowPitch,                                          ///< [in] Length of each row in bytes in the host memory region pointed to
                                                                  ///< by pDst.
    size_t hostSlicePitch,                                        ///< [in] Length of each 2D slice in bytes in the host memory region
                                                                  ///< pointed to by pDst.
    void *pDst,                                                   ///< [in] Pointer to host memory where data is to be read into.
    uint32_t numSyncPointsInWaitList,                             ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *pSyncPoint                ///< [out][optional] Sync point associated with this command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a memory fill command to a command-buffer object.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pPattern`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + If `offset + size` results in an out-of-bounds access.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferAppendMemBufferFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,                ///< [in] handle of the command-buffer object.
    ur_mem_handle_t hBuffer,                                      ///< [in] handle of the buffer object.
    const void *pPattern,                                         ///< [in] pointer to the fill pattern.
    size_t patternSize,                                           ///< [in] size in bytes of the pattern.
    size_t offset,                                                ///< [in] offset into the buffer.
    size_t size,                                                  ///< [in] fill size in bytes, must be a multiple of patternSize.
    uint32_t numSyncPointsInWaitList,                             ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *pSyncPoint                ///< [out][optional] sync point associated with this command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a USM Prefetch command to a command-buffer object.
///
/// @details
///     - Prefetching may not be supported for all devices or allocation types.
///       If memory prefetching is not supported, the prefetch hint will be
///       ignored.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMemory`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_MIGRATION_FLAGS_MASK & flags`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + If `size` is higher than the allocation size of `pMemory`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferAppendUSMPrefetchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,                ///< [in] handle of the command-buffer object.
    const void *pMemory,                                          ///< [in] pointer to USM allocated memory to prefetch.
    size_t size,                                                  ///< [in] size in bytes to be fetched.
    ur_usm_migration_flags_t flags,                               ///< [in] USM prefetch flags
    uint32_t numSyncPointsInWaitList,                             ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *pSyncPoint                ///< [out][optional] sync point associated with this command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Append a USM Advise command to a command-buffer object.
///
/// @details
///     - Not all memory advice hints may be supported for all devices or
///       allocation types. If a memory advice hint is not supported, it will be
///       ignored.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMemory`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_USM_ADVICE_FLAGS_MASK & advice`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
///         + `pSyncPointWaitList == NULL && numSyncPointsInWaitList > 0`
///         + `pSyncPointWaitList != NULL && numSyncPointsInWaitList == 0`
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `size == 0`
///         + If `size` is higher than the allocation size of `pMemory`
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferAppendUSMAdviseExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,                ///< [in] handle of the command-buffer object.
    const void *pMemory,                                          ///< [in] pointer to the USM memory object.
    size_t size,                                                  ///< [in] size in bytes to be advised.
    ur_usm_advice_flags_t advice,                                 ///< [in] USM memory advice
    uint32_t numSyncPointsInWaitList,                             ///< [in] The number of sync points in the provided dependency list.
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList, ///< [in][optional] A list of sync points that this command depends on.
    ur_exp_command_buffer_sync_point_t *pSyncPoint                ///< [out][optional] sync point associated with this command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Submit a command-buffer for execution on a queue.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///         + `NULL == hQueue`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ///< [in] Handle of the command-buffer object.
    ur_queue_handle_t hQueue,                      ///< [in] The queue to submit this command-buffer for execution.
    uint32_t numEventsInWaitList,                  ///< [in] Size of the event wait list.
    const ur_event_handle_t *phEventWaitList,      ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                                   ///< events that must be complete before the command-buffer execution.
                                                   ///< If nullptr, the numEventsInWaitList must be 0, indicating no wait events.
    ur_event_handle_t *phEvent                     ///< [out][optional] return an event object that identifies this particular
                                                   ///< command-buffer execution instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Increment the command object's reference count.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommand`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferRetainCommandExp(
    ur_exp_command_buffer_command_handle_t hCommand ///< [in] Handle of the command-buffer command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Decrement the command object's reference count and delete the command
///        object if the reference count becomes zero.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommand`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferReleaseCommandExp(
    ur_exp_command_buffer_command_handle_t hCommand ///< [in] Handle of the command-buffer command.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Update a kernel launch command in a finalized command-buffer.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommand`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pUpdateKernelLaunch`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + If update functionality is not supported by the device.
///     - ::UR_RESULT_ERROR_INVALID_OPERATION
///         + If ::ur_exp_command_buffer_desc_t::isUpdatable was not set to true on creation of the command buffer `hCommand` belongs to.
///         + If the command-buffer `hCommand` belongs to has not been finalized.
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP
///     - ::UR_RESULT_ERROR_INVALID_MEM_OBJECT
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
///     - ::UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///     - ::UR_RESULT_ERROR_INVALID_WORK_DIMENSION
///     - ::UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferUpdateKernelLaunchExp(
    ur_exp_command_buffer_command_handle_t hCommand,                             ///< [in] Handle of the command-buffer kernel command to update.
    const ur_exp_command_buffer_update_kernel_launch_desc_t *pUpdateKernelLaunch ///< [in] Struct defining how the kernel command is to be updated.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get command-buffer object information.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommandBuffer`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EXP_COMMAND_BUFFER_INFO_REFERENCE_COUNT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferGetInfoExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ///< [in] handle of the command-buffer object
    ur_exp_command_buffer_info_t propName,         ///< [in] the name of the command-buffer property to query
    size_t propSize,                               ///< [in] size in bytes of the command-buffer property value
    void *pPropValue,                              ///< [out][optional][typename(propName, propSize)] value of the
                                                   ///< command-buffer property
    size_t *pPropSizeRet                           ///< [out][optional] bytes returned in command-buffer property
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Get command-buffer object information.
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hCommand`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EXP_COMMAND_BUFFER_COMMAND_INFO_REFERENCE_COUNT < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferCommandGetInfoExp(
    ur_exp_command_buffer_command_handle_t hCommand, ///< [in] handle of the command-buffer command object
    ur_exp_command_buffer_command_info_t propName,   ///< [in] the name of the command-buffer command property to query
    size_t propSize,                                 ///< [in] size in bytes of the command-buffer command property value
    void *pPropValue,                                ///< [out][optional][typename(propName, propSize)] value of the
                                                     ///< command-buffer command property
    size_t *pPropSizeRet                             ///< [out][optional] bytes returned in command-buffer command property
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime Experimental APIs for Cooperative Kernels
#if !defined(__GNUC__)
#pragma region cooperative kernels(experimental)
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef UR_COOPERATIVE_KERNELS_EXTENSION_STRING_EXP
/// @brief The extension string which defines support for cooperative-kernels
///        which is returned when querying device extensions.
#define UR_COOPERATIVE_KERNELS_EXTENSION_STRING_EXP "ur_exp_cooperative_kernels"
#endif // UR_COOPERATIVE_KERNELS_EXTENSION_STRING_EXP

///////////////////////////////////////////////////////////////////////////////
/// @brief Enqueue a command to execute a cooperative kernel
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hQueue`
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pGlobalWorkOffset`
///         + `NULL == pGlobalWorkSize`
///     - ::UR_RESULT_ERROR_INVALID_QUEUE
///     - ::UR_RESULT_ERROR_INVALID_KERNEL
///     - ::UR_RESULT_ERROR_INVALID_EVENT
///     - ::UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST
///         + `phEventWaitList == NULL && numEventsInWaitList > 0`
///         + `phEventWaitList != NULL && numEventsInWaitList == 0`
///         + If event objects in phEventWaitList are not valid events.
///     - ::UR_RESULT_ERROR_INVALID_WORK_DIMENSION
///     - ::UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE
///     - ::UR_RESULT_ERROR_INVALID_VALUE
///     - ::UR_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::UR_RESULT_ERROR_OUT_OF_RESOURCES
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueCooperativeKernelLaunchExp(
    ur_queue_handle_t hQueue,                 ///< [in] handle of the queue object
    ur_kernel_handle_t hKernel,               ///< [in] handle of the kernel object
    uint32_t workDim,                         ///< [in] number of dimensions, from 1 to 3, to specify the global and
                                              ///< work-group work-items
    const size_t *pGlobalWorkOffset,          ///< [in] pointer to an array of workDim unsigned values that specify the
                                              ///< offset used to calculate the global ID of a work-item
    const size_t *pGlobalWorkSize,            ///< [in] pointer to an array of workDim unsigned values that specify the
                                              ///< number of global work-items in workDim that will execute the kernel
                                              ///< function
    const size_t *pLocalWorkSize,             ///< [in][optional] pointer to an array of workDim unsigned values that
                                              ///< specify the number of local work-items forming a work-group that will
                                              ///< execute the kernel function.
                                              ///< If nullptr, the runtime implementation will choose the work-group
                                              ///< size.
    uint32_t numEventsInWaitList,             ///< [in] size of the event wait list
    const ur_event_handle_t *phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
                                              ///< events that must be complete before the kernel execution.
                                              ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
                                              ///< event.
    ur_event_handle_t *phEvent                ///< [out][optional] return an event object that identifies this particular
                                              ///< kernel execution instance.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Query the maximum number of work groups for a cooperative kernel
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hKernel`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pGroupCountRet`
///     - ::UR_RESULT_ERROR_INVALID_KERNEL
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSuggestMaxCooperativeGroupCountExp(
    ur_kernel_handle_t hKernel,     ///< [in] handle of the kernel object
    size_t localWorkSize,           ///< [in] number of local work-items that will form a work-group when the
                                    ///< kernel is launched
    size_t dynamicSharedMemorySize, ///< [in] size of dynamic shared memory, for each work-group, in bytes,
                                    ///< that will be used when the kernel is launched
    uint32_t *pGroupCountRet        ///< [out] pointer to maximum number of groups
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime Experimental APIs for multi-device compile
#if !defined(__GNUC__)
#pragma region multi device compile(experimental)
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef UR_MULTI_DEVICE_COMPILE_EXTENSION_STRING_EXP
/// @brief The extension string which defines support for test
///        which is returned when querying device extensions.
#define UR_MULTI_DEVICE_COMPILE_EXTENSION_STRING_EXP "ur_exp_multi_device_compile"
#endif // UR_MULTI_DEVICE_COMPILE_EXTENSION_STRING_EXP

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one program, negates need for the
///        linking step.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point, the program passed
///       will contain a binary of the ::UR_PROGRAM_BINARY_TYPE_EXECUTABLE type
///       for each device in `phDevices`.
///
/// @remarks
///   _Analogues_
///     - **clBuildProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevices`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If `hProgram` isn't a valid program object.
///     - ::UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE
///         + If an error occurred when building `hProgram`.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramBuildExp(
    ur_program_handle_t hProgram,  ///< [in] Handle of the program to build.
    uint32_t numDevices,           ///< [in] number of devices
    ur_device_handle_t *phDevices, ///< [in][range(0, numDevices)] pointer to array of device handles
    const char *pOptions           ///< [in][optional] pointer to build options null-terminated string.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one or more programs.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point `hProgram` will
///       contain a binary of the ::UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT type
///       for each device in `phDevices`.
///
/// @remarks
///   _Analogues_
///     - **clCompileProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hProgram`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevices`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If `hProgram` isn't a valid program object.
///     - ::UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE
///         + If an error occurred while compiling `hProgram`.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramCompileExp(
    ur_program_handle_t hProgram,  ///< [in][out] handle of the program to compile.
    uint32_t numDevices,           ///< [in] number of devices
    ur_device_handle_t *phDevices, ///< [in][range(0, numDevices)] pointer to array of device handles
    const char *pOptions           ///< [in][optional] pointer to build options null-terminated string.
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Produces an executable program from one or more programs.
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - Following a successful call to this entry point the program returned
///       in `phProgram` will contain a binary of the
///       ::UR_PROGRAM_BINARY_TYPE_EXECUTABLE type for each device in
///       `phDevices`.
///
/// @remarks
///   _Analogues_
///     - **clLinkProgram**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == phDevices`
///         + `NULL == phPrograms`
///         + `NULL == phProgram`
///     - ::UR_RESULT_ERROR_INVALID_PROGRAM
///         + If one of the programs in `phPrograms` isn't a valid program object.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `count == 0`
///     - ::UR_RESULT_ERROR_PROGRAM_LINK_FAILURE
///         + If an error occurred while linking `phPrograms`.
UR_APIEXPORT ur_result_t UR_APICALL
urProgramLinkExp(
    ur_context_handle_t hContext,          ///< [in] handle of the context instance.
    uint32_t numDevices,                   ///< [in] number of devices
    ur_device_handle_t *phDevices,         ///< [in][range(0, numDevices)] pointer to array of device handles
    uint32_t count,                        ///< [in] number of program handles in `phPrograms`.
    const ur_program_handle_t *phPrograms, ///< [in][range(0, count)] pointer to array of program handles.
    const char *pOptions,                  ///< [in][optional] pointer to linker options null-terminated string.
    ur_program_handle_t *phProgram         ///< [out] pointer to handle of program object created.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' USM Import/Release Extension APIs
#if !defined(__GNUC__)
#pragma region usm import release(experimental)
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Import memory into USM
///
/// @details
///     - Import memory into USM
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
///     - ::UR_RESULT_ERROR_INVALID_SIZE
UR_APIEXPORT ur_result_t UR_APICALL
urUSMImportExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem,                   ///< [in] pointer to host memory object
    size_t size                   ///< [in] size in bytes of the host memory object to be imported
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Release memory from USM
///
/// @details
///     - Release memory from USM
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == hContext`
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `NULL == pMem`
///     - ::UR_RESULT_ERROR_INVALID_CONTEXT
UR_APIEXPORT ur_result_t UR_APICALL
urUSMReleaseExp(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to host memory object
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime Experimental APIs for USM P2P
#if !defined(__GNUC__)
#pragma region usm p2p(experimental)
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported peer info
typedef enum ur_exp_peer_info_t {
    UR_EXP_PEER_INFO_UR_PEER_ACCESS_SUPPORTED = 0,  ///< [uint32_t] 1 if P2P access is supported otherwise P2P access is not
                                                    ///< supported.
    UR_EXP_PEER_INFO_UR_PEER_ATOMICS_SUPPORTED = 1, ///< [uint32_t] 1 if atomic operations are supported over the P2P link,
                                                    ///< otherwise such operations are not supported.
    /// @cond
    UR_EXP_PEER_INFO_FORCE_UINT32 = 0x7fffffff
    /// @endcond

} ur_exp_peer_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Enable access to peer device memory
///
/// @details
///     - Enables the command device to access and write device memory
///       allocations located on the peer device, provided that a P2P link
///       between the two devices is available.
///     - When Peer Access is successfully enabled, P2P memory accesses are
///       guaranteed to be allowed on the peer device until
///       ::urUsmP2PDisablePeerAccessExp is called.
///     - Note that the function operands may, but aren't guaranteed to, commute
///       for a given adapter: the peer device is not guaranteed to have access
///       to device memory allocations located on the command device.
///     - It is not guaranteed that the commutation relations of the function
///       arguments are identical for peer access and peer copies: For example,
///       for a given adapter the peer device may be able to copy data from the
///       command device, but not access and write the same data on the command
///       device.
///     - Consult the appropriate adapter driver documentation for details of
///       adapter specific behavior and native error codes that may be returned.
///
/// @remarks
///   _Analogues_
///     - **cuCtxEnablePeerAccess**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == commandDevice`
///         + `NULL == peerDevice`
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
UR_APIEXPORT ur_result_t UR_APICALL
urUsmP2PEnablePeerAccessExp(
    ur_device_handle_t commandDevice, ///< [in] handle of the command device object
    ur_device_handle_t peerDevice     ///< [in] handle of the peer device object
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Disable access to peer device memory
///
/// @details
///     - Disables the ability of the command device to access and write device
///       memory allocations located on the peer device, provided that a P2P
///       link between the two devices was enabled prior to the call.
///     - Note that the function operands may, but aren't guaranteed to, commute
///       for a given adapter. If, prior to the function call, the peer device
///       had access to device memory allocations on the command device, it is
///       not guaranteed to still have such access following the function
///       return.
///     - It is not guaranteed that the commutation relations of the function
///       arguments are identical for peer access and peer copies: For example
///       for a given adapter, if, prior to the call, the peer device had access
///       to device memory allocations on the command device, the peer device
///       may still, following the function call, be able to copy data from the
///       command device, but not access and write the same data on the command
///       device.
///     - Consult the appropriate adapter driver documentation for details of
///       adapter specific behavior and native error codes that may be returned.
///
/// @remarks
///   _Analogues_
///     - **cuCtxDisablePeerAccess**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == commandDevice`
///         + `NULL == peerDevice`
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
UR_APIEXPORT ur_result_t UR_APICALL
urUsmP2PDisablePeerAccessExp(
    ur_device_handle_t commandDevice, ///< [in] handle of the command device object
    ur_device_handle_t peerDevice     ///< [in] handle of the peer device object
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Disable access to peer device memory
///
/// @details
///     - Queries the peer access capabilities from the command device to the
///       peer device according to the query `propName`.
///
/// @remarks
///   _Analogues_
///     - **cuDeviceGetP2PAttribute**
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_UNINITIALIZED
///     - ::UR_RESULT_ERROR_DEVICE_LOST
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
///     - ::UR_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `NULL == commandDevice`
///         + `NULL == peerDevice`
///     - ::UR_RESULT_ERROR_INVALID_ENUMERATION
///         + `::UR_EXP_PEER_INFO_UR_PEER_ATOMICS_SUPPORTED < propName`
///     - ::UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION
///         + If `propName` is not supported by the adapter.
///     - ::UR_RESULT_ERROR_INVALID_SIZE
///         + `propSize == 0 && pPropValue != NULL`
///         + If `propSize` is less than the real number of bytes needed to return the info.
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///         + `propSize != 0 && pPropValue == NULL`
///         + `pPropValue == NULL && pPropSizeRet == NULL`
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_ADAPTER_SPECIFIC
UR_APIEXPORT ur_result_t UR_APICALL
urUsmP2PPeerAccessGetInfoExp(
    ur_device_handle_t commandDevice, ///< [in] handle of the command device object
    ur_device_handle_t peerDevice,    ///< [in] handle of the peer device object
    ur_exp_peer_info_t propName,      ///< [in] type of the info to retrieve
    size_t propSize,                  ///< [in] the number of bytes pointed to by pPropValue.
    void *pPropValue,                 ///< [out][optional][typename(propName, propSize)] array of bytes holding
                                      ///< the info.
                                      ///< If propSize is not equal to or greater than the real number of bytes
                                      ///< needed to return the info
                                      ///< then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
                                      ///< pPropValue is not used.
    size_t *pPropSizeRet              ///< [out][optional] pointer to the actual size in bytes of the queried propName.
);

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Unified Runtime API function parameters
#if !defined(__GNUC__)
#pragma region callbacks
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urLoaderConfigCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_loader_config_create_params_t {
    ur_loader_config_handle_t **pphLoaderConfig;
} ur_loader_config_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urLoaderConfigRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_loader_config_retain_params_t {
    ur_loader_config_handle_t *phLoaderConfig;
} ur_loader_config_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urLoaderConfigRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_loader_config_release_params_t {
    ur_loader_config_handle_t *phLoaderConfig;
} ur_loader_config_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urLoaderConfigGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_loader_config_get_info_params_t {
    ur_loader_config_handle_t *phLoaderConfig;
    ur_loader_config_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_loader_config_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urLoaderConfigEnableLayer
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_loader_config_enable_layer_params_t {
    ur_loader_config_handle_t *phLoaderConfig;
    const char **ppLayerName;
} ur_loader_config_enable_layer_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urLoaderConfigSetCodeLocationCallback
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_loader_config_set_code_location_callback_params_t {
    ur_loader_config_handle_t *phLoaderConfig;
    ur_code_location_callback_t *ppfnCodeloc;
    void **ppUserData;
} ur_loader_config_set_code_location_callback_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urPlatformGet
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_platform_get_params_t {
    ur_adapter_handle_t **pphAdapters;
    uint32_t *pNumAdapters;
    uint32_t *pNumEntries;
    ur_platform_handle_t **pphPlatforms;
    uint32_t **ppNumPlatforms;
} ur_platform_get_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urPlatformGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_platform_get_info_params_t {
    ur_platform_handle_t *phPlatform;
    ur_platform_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_platform_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urPlatformGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_platform_get_native_handle_params_t {
    ur_platform_handle_t *phPlatform;
    ur_native_handle_t **pphNativePlatform;
} ur_platform_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urPlatformCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_platform_create_with_native_handle_params_t {
    ur_native_handle_t *phNativePlatform;
    const ur_platform_native_properties_t **ppProperties;
    ur_platform_handle_t **pphPlatform;
} ur_platform_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urPlatformGetApiVersion
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_platform_get_api_version_params_t {
    ur_platform_handle_t *phPlatform;
    ur_api_version_t **ppVersion;
} ur_platform_get_api_version_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urPlatformGetBackendOption
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_platform_get_backend_option_params_t {
    ur_platform_handle_t *phPlatform;
    const char **ppFrontendOption;
    const char ***pppPlatformOption;
} ur_platform_get_backend_option_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urContextCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_create_params_t {
    uint32_t *pDeviceCount;
    const ur_device_handle_t **pphDevices;
    const ur_context_properties_t **ppProperties;
    ur_context_handle_t **pphContext;
} ur_context_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urContextRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_retain_params_t {
    ur_context_handle_t *phContext;
} ur_context_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urContextRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_release_params_t {
    ur_context_handle_t *phContext;
} ur_context_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urContextGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_get_info_params_t {
    ur_context_handle_t *phContext;
    ur_context_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_context_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urContextGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_get_native_handle_params_t {
    ur_context_handle_t *phContext;
    ur_native_handle_t **pphNativeContext;
} ur_context_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urContextCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeContext;
    uint32_t *pnumDevices;
    const ur_device_handle_t **pphDevices;
    const ur_context_native_properties_t **ppProperties;
    ur_context_handle_t **pphContext;
} ur_context_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urContextSetExtendedDeleter
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_context_set_extended_deleter_params_t {
    ur_context_handle_t *phContext;
    ur_context_extended_deleter_t *ppfnDeleter;
    void **ppUserData;
} ur_context_set_extended_deleter_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEventGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_get_info_params_t {
    ur_event_handle_t *phEvent;
    ur_event_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_event_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEventGetProfilingInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_get_profiling_info_params_t {
    ur_event_handle_t *phEvent;
    ur_profiling_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_event_get_profiling_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEventWait
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_wait_params_t {
    uint32_t *pnumEvents;
    const ur_event_handle_t **pphEventWaitList;
} ur_event_wait_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEventRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_retain_params_t {
    ur_event_handle_t *phEvent;
} ur_event_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEventRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_release_params_t {
    ur_event_handle_t *phEvent;
} ur_event_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEventGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_get_native_handle_params_t {
    ur_event_handle_t *phEvent;
    ur_native_handle_t **pphNativeEvent;
} ur_event_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEventCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeEvent;
    ur_context_handle_t *phContext;
    const ur_event_native_properties_t **ppProperties;
    ur_event_handle_t **pphEvent;
} ur_event_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEventSetCallback
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_event_set_callback_params_t {
    ur_event_handle_t *phEvent;
    ur_execution_info_t *pexecStatus;
    ur_event_callback_t *ppfnNotify;
    void **ppUserData;
} ur_event_set_callback_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramCreateWithIL
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_create_with_il_params_t {
    ur_context_handle_t *phContext;
    const void **ppIL;
    size_t *plength;
    const ur_program_properties_t **ppProperties;
    ur_program_handle_t **pphProgram;
} ur_program_create_with_il_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramCreateWithBinary
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_create_with_binary_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    size_t *psize;
    const uint8_t **ppBinary;
    const ur_program_properties_t **ppProperties;
    ur_program_handle_t **pphProgram;
} ur_program_create_with_binary_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramBuild
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_build_params_t {
    ur_context_handle_t *phContext;
    ur_program_handle_t *phProgram;
    const char **ppOptions;
} ur_program_build_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramBuildExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_build_exp_params_t {
    ur_program_handle_t *phProgram;
    uint32_t *pnumDevices;
    ur_device_handle_t **pphDevices;
    const char **ppOptions;
} ur_program_build_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramCompile
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_compile_params_t {
    ur_context_handle_t *phContext;
    ur_program_handle_t *phProgram;
    const char **ppOptions;
} ur_program_compile_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramCompileExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_compile_exp_params_t {
    ur_program_handle_t *phProgram;
    uint32_t *pnumDevices;
    ur_device_handle_t **pphDevices;
    const char **ppOptions;
} ur_program_compile_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramLink
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_link_params_t {
    ur_context_handle_t *phContext;
    uint32_t *pcount;
    const ur_program_handle_t **pphPrograms;
    const char **ppOptions;
    ur_program_handle_t **pphProgram;
} ur_program_link_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramLinkExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_link_exp_params_t {
    ur_context_handle_t *phContext;
    uint32_t *pnumDevices;
    ur_device_handle_t **pphDevices;
    uint32_t *pcount;
    const ur_program_handle_t **pphPrograms;
    const char **ppOptions;
    ur_program_handle_t **pphProgram;
} ur_program_link_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_retain_params_t {
    ur_program_handle_t *phProgram;
} ur_program_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_release_params_t {
    ur_program_handle_t *phProgram;
} ur_program_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramGetFunctionPointer
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_get_function_pointer_params_t {
    ur_device_handle_t *phDevice;
    ur_program_handle_t *phProgram;
    const char **ppFunctionName;
    void ***pppFunctionPointer;
} ur_program_get_function_pointer_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_get_info_params_t {
    ur_program_handle_t *phProgram;
    ur_program_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_program_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramGetBuildInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_get_build_info_params_t {
    ur_program_handle_t *phProgram;
    ur_device_handle_t *phDevice;
    ur_program_build_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_program_get_build_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramSetSpecializationConstants
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_set_specialization_constants_params_t {
    ur_program_handle_t *phProgram;
    uint32_t *pcount;
    const ur_specialization_constant_info_t **ppSpecConstants;
} ur_program_set_specialization_constants_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_get_native_handle_params_t {
    ur_program_handle_t *phProgram;
    ur_native_handle_t **pphNativeProgram;
} ur_program_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urProgramCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_program_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeProgram;
    ur_context_handle_t *phContext;
    const ur_program_native_properties_t **ppProperties;
    ur_program_handle_t **pphProgram;
} ur_program_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_create_params_t {
    ur_program_handle_t *phProgram;
    const char **ppKernelName;
    ur_kernel_handle_t **pphKernel;
} ur_kernel_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_get_info_params_t {
    ur_kernel_handle_t *phKernel;
    ur_kernel_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_kernel_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelGetGroupInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_get_group_info_params_t {
    ur_kernel_handle_t *phKernel;
    ur_device_handle_t *phDevice;
    ur_kernel_group_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_kernel_get_group_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelGetSubGroupInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_get_sub_group_info_params_t {
    ur_kernel_handle_t *phKernel;
    ur_device_handle_t *phDevice;
    ur_kernel_sub_group_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_kernel_get_sub_group_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_retain_params_t {
    ur_kernel_handle_t *phKernel;
} ur_kernel_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_release_params_t {
    ur_kernel_handle_t *phKernel;
} ur_kernel_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_get_native_handle_params_t {
    ur_kernel_handle_t *phKernel;
    ur_native_handle_t **pphNativeKernel;
} ur_kernel_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeKernel;
    ur_context_handle_t *phContext;
    ur_program_handle_t *phProgram;
    const ur_kernel_native_properties_t **ppProperties;
    ur_kernel_handle_t **pphKernel;
} ur_kernel_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelSetArgValue
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_arg_value_params_t {
    ur_kernel_handle_t *phKernel;
    uint32_t *pargIndex;
    size_t *pargSize;
    const ur_kernel_arg_value_properties_t **ppProperties;
    const void **ppArgValue;
} ur_kernel_set_arg_value_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelSetArgLocal
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_arg_local_params_t {
    ur_kernel_handle_t *phKernel;
    uint32_t *pargIndex;
    size_t *pargSize;
    const ur_kernel_arg_local_properties_t **ppProperties;
} ur_kernel_set_arg_local_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelSetArgPointer
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_arg_pointer_params_t {
    ur_kernel_handle_t *phKernel;
    uint32_t *pargIndex;
    const ur_kernel_arg_pointer_properties_t **ppProperties;
    const void **ppArgValue;
} ur_kernel_set_arg_pointer_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelSetExecInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_exec_info_params_t {
    ur_kernel_handle_t *phKernel;
    ur_kernel_exec_info_t *ppropName;
    size_t *ppropSize;
    const ur_kernel_exec_info_properties_t **ppProperties;
    const void **ppPropValue;
} ur_kernel_set_exec_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelSetArgSampler
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_arg_sampler_params_t {
    ur_kernel_handle_t *phKernel;
    uint32_t *pargIndex;
    const ur_kernel_arg_sampler_properties_t **ppProperties;
    ur_sampler_handle_t *phArgValue;
} ur_kernel_set_arg_sampler_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelSetArgMemObj
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_arg_mem_obj_params_t {
    ur_kernel_handle_t *phKernel;
    uint32_t *pargIndex;
    const ur_kernel_arg_mem_obj_properties_t **ppProperties;
    ur_mem_handle_t *phArgValue;
} ur_kernel_set_arg_mem_obj_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelSetSpecializationConstants
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_set_specialization_constants_params_t {
    ur_kernel_handle_t *phKernel;
    uint32_t *pcount;
    const ur_specialization_constant_info_t **ppSpecConstants;
} ur_kernel_set_specialization_constants_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urKernelSuggestMaxCooperativeGroupCountExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_kernel_suggest_max_cooperative_group_count_exp_params_t {
    ur_kernel_handle_t *phKernel;
    size_t *plocalWorkSize;
    size_t *pdynamicSharedMemorySize;
    uint32_t **ppGroupCountRet;
} ur_kernel_suggest_max_cooperative_group_count_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urQueueGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_get_info_params_t {
    ur_queue_handle_t *phQueue;
    ur_queue_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_queue_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urQueueCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_create_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    const ur_queue_properties_t **ppProperties;
    ur_queue_handle_t **pphQueue;
} ur_queue_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urQueueRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_retain_params_t {
    ur_queue_handle_t *phQueue;
} ur_queue_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urQueueRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_release_params_t {
    ur_queue_handle_t *phQueue;
} ur_queue_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urQueueGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_get_native_handle_params_t {
    ur_queue_handle_t *phQueue;
    ur_queue_native_desc_t **ppDesc;
    ur_native_handle_t **pphNativeQueue;
} ur_queue_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urQueueCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeQueue;
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    const ur_queue_native_properties_t **ppProperties;
    ur_queue_handle_t **pphQueue;
} ur_queue_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urQueueFinish
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_finish_params_t {
    ur_queue_handle_t *phQueue;
} ur_queue_finish_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urQueueFlush
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_queue_flush_params_t {
    ur_queue_handle_t *phQueue;
} ur_queue_flush_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urSamplerCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_sampler_create_params_t {
    ur_context_handle_t *phContext;
    const ur_sampler_desc_t **ppDesc;
    ur_sampler_handle_t **pphSampler;
} ur_sampler_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urSamplerRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_sampler_retain_params_t {
    ur_sampler_handle_t *phSampler;
} ur_sampler_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urSamplerRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_sampler_release_params_t {
    ur_sampler_handle_t *phSampler;
} ur_sampler_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urSamplerGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_sampler_get_info_params_t {
    ur_sampler_handle_t *phSampler;
    ur_sampler_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_sampler_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urSamplerGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_sampler_get_native_handle_params_t {
    ur_sampler_handle_t *phSampler;
    ur_native_handle_t **pphNativeSampler;
} ur_sampler_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urSamplerCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_sampler_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeSampler;
    ur_context_handle_t *phContext;
    const ur_sampler_native_properties_t **ppProperties;
    ur_sampler_handle_t **pphSampler;
} ur_sampler_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urMemImageCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_image_create_params_t {
    ur_context_handle_t *phContext;
    ur_mem_flags_t *pflags;
    const ur_image_format_t **ppImageFormat;
    const ur_image_desc_t **ppImageDesc;
    void **ppHost;
    ur_mem_handle_t **pphMem;
} ur_mem_image_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urMemBufferCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_buffer_create_params_t {
    ur_context_handle_t *phContext;
    ur_mem_flags_t *pflags;
    size_t *psize;
    const ur_buffer_properties_t **ppProperties;
    ur_mem_handle_t **pphBuffer;
} ur_mem_buffer_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urMemRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_retain_params_t {
    ur_mem_handle_t *phMem;
} ur_mem_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urMemRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_release_params_t {
    ur_mem_handle_t *phMem;
} ur_mem_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urMemBufferPartition
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_buffer_partition_params_t {
    ur_mem_handle_t *phBuffer;
    ur_mem_flags_t *pflags;
    ur_buffer_create_type_t *pbufferCreateType;
    const ur_buffer_region_t **ppRegion;
    ur_mem_handle_t **pphMem;
} ur_mem_buffer_partition_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urMemGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_get_native_handle_params_t {
    ur_mem_handle_t *phMem;
    ur_device_handle_t *phDevice;
    ur_native_handle_t **pphNativeMem;
} ur_mem_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urMemBufferCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_buffer_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeMem;
    ur_context_handle_t *phContext;
    const ur_mem_native_properties_t **ppProperties;
    ur_mem_handle_t **pphMem;
} ur_mem_buffer_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urMemImageCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_image_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeMem;
    ur_context_handle_t *phContext;
    const ur_image_format_t **ppImageFormat;
    const ur_image_desc_t **ppImageDesc;
    const ur_mem_native_properties_t **ppProperties;
    ur_mem_handle_t **pphMem;
} ur_mem_image_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urMemGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_get_info_params_t {
    ur_mem_handle_t *phMemory;
    ur_mem_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_mem_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urMemImageGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_mem_image_get_info_params_t {
    ur_mem_handle_t *phMemory;
    ur_image_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_mem_image_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urPhysicalMemCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_physical_mem_create_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    size_t *psize;
    const ur_physical_mem_properties_t **ppProperties;
    ur_physical_mem_handle_t **pphPhysicalMem;
} ur_physical_mem_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urPhysicalMemRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_physical_mem_retain_params_t {
    ur_physical_mem_handle_t *phPhysicalMem;
} ur_physical_mem_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urPhysicalMemRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_physical_mem_release_params_t {
    ur_physical_mem_handle_t *phPhysicalMem;
} ur_physical_mem_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urAdapterGet
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_adapter_get_params_t {
    uint32_t *pNumEntries;
    ur_adapter_handle_t **pphAdapters;
    uint32_t **ppNumAdapters;
} ur_adapter_get_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urAdapterRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_adapter_release_params_t {
    ur_adapter_handle_t *phAdapter;
} ur_adapter_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urAdapterRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_adapter_retain_params_t {
    ur_adapter_handle_t *phAdapter;
} ur_adapter_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urAdapterGetLastError
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_adapter_get_last_error_params_t {
    ur_adapter_handle_t *phAdapter;
    const char ***pppMessage;
    int32_t **ppError;
} ur_adapter_get_last_error_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urAdapterGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_adapter_get_info_params_t {
    ur_adapter_handle_t *phAdapter;
    ur_adapter_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_adapter_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueKernelLaunch
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_kernel_launch_params_t {
    ur_queue_handle_t *phQueue;
    ur_kernel_handle_t *phKernel;
    uint32_t *pworkDim;
    const size_t **ppGlobalWorkOffset;
    const size_t **ppGlobalWorkSize;
    const size_t **ppLocalWorkSize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_kernel_launch_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueEventsWait
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_events_wait_params_t {
    ur_queue_handle_t *phQueue;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_events_wait_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueEventsWaitWithBarrier
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_events_wait_with_barrier_params_t {
    ur_queue_handle_t *phQueue;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_events_wait_with_barrier_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueMemBufferRead
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_read_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBuffer;
    bool *pblockingRead;
    size_t *poffset;
    size_t *psize;
    void **ppDst;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_read_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueMemBufferWrite
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_write_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBuffer;
    bool *pblockingWrite;
    size_t *poffset;
    size_t *psize;
    const void **ppSrc;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_write_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueMemBufferReadRect
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_read_rect_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBuffer;
    bool *pblockingRead;
    ur_rect_offset_t *pbufferOrigin;
    ur_rect_offset_t *phostOrigin;
    ur_rect_region_t *pregion;
    size_t *pbufferRowPitch;
    size_t *pbufferSlicePitch;
    size_t *phostRowPitch;
    size_t *phostSlicePitch;
    void **ppDst;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_read_rect_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueMemBufferWriteRect
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_write_rect_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBuffer;
    bool *pblockingWrite;
    ur_rect_offset_t *pbufferOrigin;
    ur_rect_offset_t *phostOrigin;
    ur_rect_region_t *pregion;
    size_t *pbufferRowPitch;
    size_t *pbufferSlicePitch;
    size_t *phostRowPitch;
    size_t *phostSlicePitch;
    void **ppSrc;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_write_rect_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueMemBufferCopy
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_copy_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBufferSrc;
    ur_mem_handle_t *phBufferDst;
    size_t *psrcOffset;
    size_t *pdstOffset;
    size_t *psize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_copy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueMemBufferCopyRect
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_copy_rect_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBufferSrc;
    ur_mem_handle_t *phBufferDst;
    ur_rect_offset_t *psrcOrigin;
    ur_rect_offset_t *pdstOrigin;
    ur_rect_region_t *pregion;
    size_t *psrcRowPitch;
    size_t *psrcSlicePitch;
    size_t *pdstRowPitch;
    size_t *pdstSlicePitch;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_copy_rect_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueMemBufferFill
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_fill_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBuffer;
    const void **ppPattern;
    size_t *ppatternSize;
    size_t *poffset;
    size_t *psize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_buffer_fill_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueMemImageRead
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_image_read_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phImage;
    bool *pblockingRead;
    ur_rect_offset_t *porigin;
    ur_rect_region_t *pregion;
    size_t *prowPitch;
    size_t *pslicePitch;
    void **ppDst;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_image_read_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueMemImageWrite
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_image_write_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phImage;
    bool *pblockingWrite;
    ur_rect_offset_t *porigin;
    ur_rect_region_t *pregion;
    size_t *prowPitch;
    size_t *pslicePitch;
    void **ppSrc;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_image_write_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueMemImageCopy
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_image_copy_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phImageSrc;
    ur_mem_handle_t *phImageDst;
    ur_rect_offset_t *psrcOrigin;
    ur_rect_offset_t *pdstOrigin;
    ur_rect_region_t *pregion;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_image_copy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueMemBufferMap
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_buffer_map_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phBuffer;
    bool *pblockingMap;
    ur_map_flags_t *pmapFlags;
    size_t *poffset;
    size_t *psize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
    void ***pppRetMap;
} ur_enqueue_mem_buffer_map_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueMemUnmap
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_mem_unmap_params_t {
    ur_queue_handle_t *phQueue;
    ur_mem_handle_t *phMem;
    void **ppMappedPtr;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_mem_unmap_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueUSMFill
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_usm_fill_params_t {
    ur_queue_handle_t *phQueue;
    void **ppMem;
    size_t *ppatternSize;
    const void **ppPattern;
    size_t *psize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_usm_fill_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueUSMMemcpy
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_usm_memcpy_params_t {
    ur_queue_handle_t *phQueue;
    bool *pblocking;
    void **ppDst;
    const void **ppSrc;
    size_t *psize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_usm_memcpy_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueUSMPrefetch
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_usm_prefetch_params_t {
    ur_queue_handle_t *phQueue;
    const void **ppMem;
    size_t *psize;
    ur_usm_migration_flags_t *pflags;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_usm_prefetch_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueUSMAdvise
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_usm_advise_params_t {
    ur_queue_handle_t *phQueue;
    const void **ppMem;
    size_t *psize;
    ur_usm_advice_flags_t *padvice;
    ur_event_handle_t **pphEvent;
} ur_enqueue_usm_advise_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueUSMFill2D
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_usm_fill_2d_params_t {
    ur_queue_handle_t *phQueue;
    void **ppMem;
    size_t *ppitch;
    size_t *ppatternSize;
    const void **ppPattern;
    size_t *pwidth;
    size_t *pheight;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_usm_fill_2d_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueUSMMemcpy2D
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_usm_memcpy_2d_params_t {
    ur_queue_handle_t *phQueue;
    bool *pblocking;
    void **ppDst;
    size_t *pdstPitch;
    const void **ppSrc;
    size_t *psrcPitch;
    size_t *pwidth;
    size_t *pheight;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_usm_memcpy_2d_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueDeviceGlobalVariableWrite
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_device_global_variable_write_params_t {
    ur_queue_handle_t *phQueue;
    ur_program_handle_t *phProgram;
    const char **pname;
    bool *pblockingWrite;
    size_t *pcount;
    size_t *poffset;
    const void **ppSrc;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_device_global_variable_write_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueDeviceGlobalVariableRead
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_device_global_variable_read_params_t {
    ur_queue_handle_t *phQueue;
    ur_program_handle_t *phProgram;
    const char **pname;
    bool *pblockingRead;
    size_t *pcount;
    size_t *poffset;
    void **ppDst;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_device_global_variable_read_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueReadHostPipe
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_read_host_pipe_params_t {
    ur_queue_handle_t *phQueue;
    ur_program_handle_t *phProgram;
    const char **ppipe_symbol;
    bool *pblocking;
    void **ppDst;
    size_t *psize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_read_host_pipe_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueWriteHostPipe
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_write_host_pipe_params_t {
    ur_queue_handle_t *phQueue;
    ur_program_handle_t *phProgram;
    const char **ppipe_symbol;
    bool *pblocking;
    void **ppSrc;
    size_t *psize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_write_host_pipe_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urEnqueueCooperativeKernelLaunchExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_enqueue_cooperative_kernel_launch_exp_params_t {
    ur_queue_handle_t *phQueue;
    ur_kernel_handle_t *phKernel;
    uint32_t *pworkDim;
    const size_t **ppGlobalWorkOffset;
    const size_t **ppGlobalWorkSize;
    const size_t **ppLocalWorkSize;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_enqueue_cooperative_kernel_launch_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesUnsampledImageHandleDestroyExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_unsampled_image_handle_destroy_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    ur_exp_image_handle_t *phImage;
} ur_bindless_images_unsampled_image_handle_destroy_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesSampledImageHandleDestroyExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_sampled_image_handle_destroy_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    ur_exp_image_handle_t *phImage;
} ur_bindless_images_sampled_image_handle_destroy_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesImageAllocateExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_image_allocate_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    const ur_image_format_t **ppImageFormat;
    const ur_image_desc_t **ppImageDesc;
    ur_exp_image_mem_handle_t **pphImageMem;
} ur_bindless_images_image_allocate_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesImageFreeExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_image_free_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    ur_exp_image_mem_handle_t *phImageMem;
} ur_bindless_images_image_free_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesUnsampledImageCreateExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_unsampled_image_create_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    ur_exp_image_mem_handle_t *phImageMem;
    const ur_image_format_t **ppImageFormat;
    const ur_image_desc_t **ppImageDesc;
    ur_mem_handle_t **pphMem;
    ur_exp_image_handle_t **pphImage;
} ur_bindless_images_unsampled_image_create_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesSampledImageCreateExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_sampled_image_create_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    ur_exp_image_mem_handle_t *phImageMem;
    const ur_image_format_t **ppImageFormat;
    const ur_image_desc_t **ppImageDesc;
    ur_sampler_handle_t *phSampler;
    ur_mem_handle_t **pphMem;
    ur_exp_image_handle_t **pphImage;
} ur_bindless_images_sampled_image_create_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesImageCopyExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_image_copy_exp_params_t {
    ur_queue_handle_t *phQueue;
    void **ppDst;
    void **ppSrc;
    const ur_image_format_t **ppImageFormat;
    const ur_image_desc_t **ppImageDesc;
    ur_exp_image_copy_flags_t *pimageCopyFlags;
    ur_rect_offset_t *psrcOffset;
    ur_rect_offset_t *pdstOffset;
    ur_rect_region_t *pcopyExtent;
    ur_rect_region_t *phostExtent;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_bindless_images_image_copy_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesImageGetInfoExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_image_get_info_exp_params_t {
    ur_exp_image_mem_handle_t *phImageMem;
    ur_image_info_t *ppropName;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_bindless_images_image_get_info_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesMipmapGetLevelExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_mipmap_get_level_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    ur_exp_image_mem_handle_t *phImageMem;
    uint32_t *pmipmapLevel;
    ur_exp_image_mem_handle_t **pphImageMem;
} ur_bindless_images_mipmap_get_level_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesMipmapFreeExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_mipmap_free_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    ur_exp_image_mem_handle_t *phMem;
} ur_bindless_images_mipmap_free_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesImportOpaqueFDExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_import_opaque_fd_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    size_t *psize;
    ur_exp_interop_mem_desc_t **ppInteropMemDesc;
    ur_exp_interop_mem_handle_t **pphInteropMem;
} ur_bindless_images_import_opaque_fd_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesMapExternalArrayExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_map_external_array_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    const ur_image_format_t **ppImageFormat;
    const ur_image_desc_t **ppImageDesc;
    ur_exp_interop_mem_handle_t *phInteropMem;
    ur_exp_image_mem_handle_t **pphImageMem;
} ur_bindless_images_map_external_array_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesReleaseInteropExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_release_interop_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    ur_exp_interop_mem_handle_t *phInteropMem;
} ur_bindless_images_release_interop_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesImportExternalSemaphoreOpaqueFDExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_import_external_semaphore_opaque_fd_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    ur_exp_interop_semaphore_desc_t **ppInteropSemaphoreDesc;
    ur_exp_interop_semaphore_handle_t **pphInteropSemaphore;
} ur_bindless_images_import_external_semaphore_opaque_fd_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesDestroyExternalSemaphoreExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_destroy_external_semaphore_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    ur_exp_interop_semaphore_handle_t *phInteropSemaphore;
} ur_bindless_images_destroy_external_semaphore_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesWaitExternalSemaphoreExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_wait_external_semaphore_exp_params_t {
    ur_queue_handle_t *phQueue;
    ur_exp_interop_semaphore_handle_t *phSemaphore;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_bindless_images_wait_external_semaphore_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urBindlessImagesSignalExternalSemaphoreExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_bindless_images_signal_external_semaphore_exp_params_t {
    ur_queue_handle_t *phQueue;
    ur_exp_interop_semaphore_handle_t *phSemaphore;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_bindless_images_signal_external_semaphore_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUSMHostAlloc
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_host_alloc_params_t {
    ur_context_handle_t *phContext;
    const ur_usm_desc_t **ppUSMDesc;
    ur_usm_pool_handle_t *ppool;
    size_t *psize;
    void ***pppMem;
} ur_usm_host_alloc_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUSMDeviceAlloc
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_device_alloc_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    const ur_usm_desc_t **ppUSMDesc;
    ur_usm_pool_handle_t *ppool;
    size_t *psize;
    void ***pppMem;
} ur_usm_device_alloc_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUSMSharedAlloc
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_shared_alloc_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    const ur_usm_desc_t **ppUSMDesc;
    ur_usm_pool_handle_t *ppool;
    size_t *psize;
    void ***pppMem;
} ur_usm_shared_alloc_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUSMFree
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_free_params_t {
    ur_context_handle_t *phContext;
    void **ppMem;
} ur_usm_free_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUSMGetMemAllocInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_get_mem_alloc_info_params_t {
    ur_context_handle_t *phContext;
    const void **ppMem;
    ur_usm_alloc_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_usm_get_mem_alloc_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUSMPoolCreate
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_pool_create_params_t {
    ur_context_handle_t *phContext;
    ur_usm_pool_desc_t **ppPoolDesc;
    ur_usm_pool_handle_t **pppPool;
} ur_usm_pool_create_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUSMPoolRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_pool_retain_params_t {
    ur_usm_pool_handle_t *ppPool;
} ur_usm_pool_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUSMPoolRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_pool_release_params_t {
    ur_usm_pool_handle_t *ppPool;
} ur_usm_pool_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUSMPoolGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_pool_get_info_params_t {
    ur_usm_pool_handle_t *phPool;
    ur_usm_pool_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_usm_pool_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUSMPitchedAllocExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_pitched_alloc_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    const ur_usm_desc_t **ppUSMDesc;
    ur_usm_pool_handle_t *ppool;
    size_t *pwidthInBytes;
    size_t *pheight;
    size_t *pelementSizeBytes;
    void ***pppMem;
    size_t **ppResultPitch;
} ur_usm_pitched_alloc_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUSMImportExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_import_exp_params_t {
    ur_context_handle_t *phContext;
    void **ppMem;
    size_t *psize;
} ur_usm_import_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUSMReleaseExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_release_exp_params_t {
    ur_context_handle_t *phContext;
    void **ppMem;
} ur_usm_release_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferCreateExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_create_exp_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    const ur_exp_command_buffer_desc_t **ppCommandBufferDesc;
    ur_exp_command_buffer_handle_t **pphCommandBuffer;
} ur_command_buffer_create_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferRetainExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_retain_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
} ur_command_buffer_retain_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferReleaseExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_release_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
} ur_command_buffer_release_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferFinalizeExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_finalize_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
} ur_command_buffer_finalize_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferAppendKernelLaunchExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_append_kernel_launch_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    ur_kernel_handle_t *phKernel;
    uint32_t *pworkDim;
    const size_t **ppGlobalWorkOffset;
    const size_t **ppGlobalWorkSize;
    const size_t **ppLocalWorkSize;
    uint32_t *pnumSyncPointsInWaitList;
    const ur_exp_command_buffer_sync_point_t **ppSyncPointWaitList;
    ur_exp_command_buffer_sync_point_t **ppSyncPoint;
    ur_exp_command_buffer_command_handle_t **pphCommand;
} ur_command_buffer_append_kernel_launch_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferAppendUSMMemcpyExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_append_usm_memcpy_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    void **ppDst;
    const void **ppSrc;
    size_t *psize;
    uint32_t *pnumSyncPointsInWaitList;
    const ur_exp_command_buffer_sync_point_t **ppSyncPointWaitList;
    ur_exp_command_buffer_sync_point_t **ppSyncPoint;
} ur_command_buffer_append_usm_memcpy_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferAppendUSMFillExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_append_usm_fill_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    void **ppMemory;
    const void **ppPattern;
    size_t *ppatternSize;
    size_t *psize;
    uint32_t *pnumSyncPointsInWaitList;
    const ur_exp_command_buffer_sync_point_t **ppSyncPointWaitList;
    ur_exp_command_buffer_sync_point_t **ppSyncPoint;
} ur_command_buffer_append_usm_fill_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferAppendMemBufferCopyExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_append_mem_buffer_copy_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    ur_mem_handle_t *phSrcMem;
    ur_mem_handle_t *phDstMem;
    size_t *psrcOffset;
    size_t *pdstOffset;
    size_t *psize;
    uint32_t *pnumSyncPointsInWaitList;
    const ur_exp_command_buffer_sync_point_t **ppSyncPointWaitList;
    ur_exp_command_buffer_sync_point_t **ppSyncPoint;
} ur_command_buffer_append_mem_buffer_copy_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferAppendMemBufferWriteExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_append_mem_buffer_write_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    ur_mem_handle_t *phBuffer;
    size_t *poffset;
    size_t *psize;
    const void **ppSrc;
    uint32_t *pnumSyncPointsInWaitList;
    const ur_exp_command_buffer_sync_point_t **ppSyncPointWaitList;
    ur_exp_command_buffer_sync_point_t **ppSyncPoint;
} ur_command_buffer_append_mem_buffer_write_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferAppendMemBufferReadExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_append_mem_buffer_read_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    ur_mem_handle_t *phBuffer;
    size_t *poffset;
    size_t *psize;
    void **ppDst;
    uint32_t *pnumSyncPointsInWaitList;
    const ur_exp_command_buffer_sync_point_t **ppSyncPointWaitList;
    ur_exp_command_buffer_sync_point_t **ppSyncPoint;
} ur_command_buffer_append_mem_buffer_read_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferAppendMemBufferCopyRectExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_append_mem_buffer_copy_rect_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    ur_mem_handle_t *phSrcMem;
    ur_mem_handle_t *phDstMem;
    ur_rect_offset_t *psrcOrigin;
    ur_rect_offset_t *pdstOrigin;
    ur_rect_region_t *pregion;
    size_t *psrcRowPitch;
    size_t *psrcSlicePitch;
    size_t *pdstRowPitch;
    size_t *pdstSlicePitch;
    uint32_t *pnumSyncPointsInWaitList;
    const ur_exp_command_buffer_sync_point_t **ppSyncPointWaitList;
    ur_exp_command_buffer_sync_point_t **ppSyncPoint;
} ur_command_buffer_append_mem_buffer_copy_rect_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferAppendMemBufferWriteRectExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_append_mem_buffer_write_rect_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    ur_mem_handle_t *phBuffer;
    ur_rect_offset_t *pbufferOffset;
    ur_rect_offset_t *phostOffset;
    ur_rect_region_t *pregion;
    size_t *pbufferRowPitch;
    size_t *pbufferSlicePitch;
    size_t *phostRowPitch;
    size_t *phostSlicePitch;
    void **ppSrc;
    uint32_t *pnumSyncPointsInWaitList;
    const ur_exp_command_buffer_sync_point_t **ppSyncPointWaitList;
    ur_exp_command_buffer_sync_point_t **ppSyncPoint;
} ur_command_buffer_append_mem_buffer_write_rect_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferAppendMemBufferReadRectExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_append_mem_buffer_read_rect_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    ur_mem_handle_t *phBuffer;
    ur_rect_offset_t *pbufferOffset;
    ur_rect_offset_t *phostOffset;
    ur_rect_region_t *pregion;
    size_t *pbufferRowPitch;
    size_t *pbufferSlicePitch;
    size_t *phostRowPitch;
    size_t *phostSlicePitch;
    void **ppDst;
    uint32_t *pnumSyncPointsInWaitList;
    const ur_exp_command_buffer_sync_point_t **ppSyncPointWaitList;
    ur_exp_command_buffer_sync_point_t **ppSyncPoint;
} ur_command_buffer_append_mem_buffer_read_rect_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferAppendMemBufferFillExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_append_mem_buffer_fill_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    ur_mem_handle_t *phBuffer;
    const void **ppPattern;
    size_t *ppatternSize;
    size_t *poffset;
    size_t *psize;
    uint32_t *pnumSyncPointsInWaitList;
    const ur_exp_command_buffer_sync_point_t **ppSyncPointWaitList;
    ur_exp_command_buffer_sync_point_t **ppSyncPoint;
} ur_command_buffer_append_mem_buffer_fill_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferAppendUSMPrefetchExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_append_usm_prefetch_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    const void **ppMemory;
    size_t *psize;
    ur_usm_migration_flags_t *pflags;
    uint32_t *pnumSyncPointsInWaitList;
    const ur_exp_command_buffer_sync_point_t **ppSyncPointWaitList;
    ur_exp_command_buffer_sync_point_t **ppSyncPoint;
} ur_command_buffer_append_usm_prefetch_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferAppendUSMAdviseExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_append_usm_advise_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    const void **ppMemory;
    size_t *psize;
    ur_usm_advice_flags_t *padvice;
    uint32_t *pnumSyncPointsInWaitList;
    const ur_exp_command_buffer_sync_point_t **ppSyncPointWaitList;
    ur_exp_command_buffer_sync_point_t **ppSyncPoint;
} ur_command_buffer_append_usm_advise_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferEnqueueExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_enqueue_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    ur_queue_handle_t *phQueue;
    uint32_t *pnumEventsInWaitList;
    const ur_event_handle_t **pphEventWaitList;
    ur_event_handle_t **pphEvent;
} ur_command_buffer_enqueue_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferRetainCommandExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_retain_command_exp_params_t {
    ur_exp_command_buffer_command_handle_t *phCommand;
} ur_command_buffer_retain_command_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferReleaseCommandExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_release_command_exp_params_t {
    ur_exp_command_buffer_command_handle_t *phCommand;
} ur_command_buffer_release_command_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferUpdateKernelLaunchExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_update_kernel_launch_exp_params_t {
    ur_exp_command_buffer_command_handle_t *phCommand;
    const ur_exp_command_buffer_update_kernel_launch_desc_t **ppUpdateKernelLaunch;
} ur_command_buffer_update_kernel_launch_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferGetInfoExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_get_info_exp_params_t {
    ur_exp_command_buffer_handle_t *phCommandBuffer;
    ur_exp_command_buffer_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_command_buffer_get_info_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urCommandBufferCommandGetInfoExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_command_buffer_command_get_info_exp_params_t {
    ur_exp_command_buffer_command_handle_t *phCommand;
    ur_exp_command_buffer_command_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_command_buffer_command_get_info_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUsmP2PEnablePeerAccessExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_p2p_enable_peer_access_exp_params_t {
    ur_device_handle_t *pcommandDevice;
    ur_device_handle_t *ppeerDevice;
} ur_usm_p2p_enable_peer_access_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUsmP2PDisablePeerAccessExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_p2p_disable_peer_access_exp_params_t {
    ur_device_handle_t *pcommandDevice;
    ur_device_handle_t *ppeerDevice;
} ur_usm_p2p_disable_peer_access_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urUsmP2PPeerAccessGetInfoExp
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_usm_p2p_peer_access_get_info_exp_params_t {
    ur_device_handle_t *pcommandDevice;
    ur_device_handle_t *ppeerDevice;
    ur_exp_peer_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_usm_p2p_peer_access_get_info_exp_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urLoaderInit
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_loader_init_params_t {
    ur_device_init_flags_t *pdevice_flags;
    ur_loader_config_handle_t *phLoaderConfig;
} ur_loader_init_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urVirtualMemGranularityGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_virtual_mem_granularity_get_info_params_t {
    ur_context_handle_t *phContext;
    ur_device_handle_t *phDevice;
    ur_virtual_mem_granularity_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_virtual_mem_granularity_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urVirtualMemReserve
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_virtual_mem_reserve_params_t {
    ur_context_handle_t *phContext;
    const void **ppStart;
    size_t *psize;
    void ***pppStart;
} ur_virtual_mem_reserve_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urVirtualMemFree
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_virtual_mem_free_params_t {
    ur_context_handle_t *phContext;
    const void **ppStart;
    size_t *psize;
} ur_virtual_mem_free_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urVirtualMemMap
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_virtual_mem_map_params_t {
    ur_context_handle_t *phContext;
    const void **ppStart;
    size_t *psize;
    ur_physical_mem_handle_t *phPhysicalMem;
    size_t *poffset;
    ur_virtual_mem_access_flags_t *pflags;
} ur_virtual_mem_map_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urVirtualMemUnmap
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_virtual_mem_unmap_params_t {
    ur_context_handle_t *phContext;
    const void **ppStart;
    size_t *psize;
} ur_virtual_mem_unmap_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urVirtualMemSetAccess
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_virtual_mem_set_access_params_t {
    ur_context_handle_t *phContext;
    const void **ppStart;
    size_t *psize;
    ur_virtual_mem_access_flags_t *pflags;
} ur_virtual_mem_set_access_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urVirtualMemGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_virtual_mem_get_info_params_t {
    ur_context_handle_t *phContext;
    const void **ppStart;
    size_t *psize;
    ur_virtual_mem_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_virtual_mem_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urDeviceGet
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_get_params_t {
    ur_platform_handle_t *phPlatform;
    ur_device_type_t *pDeviceType;
    uint32_t *pNumEntries;
    ur_device_handle_t **pphDevices;
    uint32_t **ppNumDevices;
} ur_device_get_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urDeviceGetInfo
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_get_info_params_t {
    ur_device_handle_t *phDevice;
    ur_device_info_t *ppropName;
    size_t *ppropSize;
    void **ppPropValue;
    size_t **ppPropSizeRet;
} ur_device_get_info_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urDeviceRetain
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_retain_params_t {
    ur_device_handle_t *phDevice;
} ur_device_retain_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urDeviceRelease
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_release_params_t {
    ur_device_handle_t *phDevice;
} ur_device_release_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urDevicePartition
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_partition_params_t {
    ur_device_handle_t *phDevice;
    const ur_device_partition_properties_t **ppProperties;
    uint32_t *pNumDevices;
    ur_device_handle_t **pphSubDevices;
    uint32_t **ppNumDevicesRet;
} ur_device_partition_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urDeviceSelectBinary
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_select_binary_params_t {
    ur_device_handle_t *phDevice;
    const ur_device_binary_t **ppBinaries;
    uint32_t *pNumBinaries;
    uint32_t **ppSelectedBinary;
} ur_device_select_binary_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urDeviceGetNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_get_native_handle_params_t {
    ur_device_handle_t *phDevice;
    ur_native_handle_t **pphNativeDevice;
} ur_device_get_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urDeviceCreateWithNativeHandle
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_create_with_native_handle_params_t {
    ur_native_handle_t *phNativeDevice;
    ur_platform_handle_t *phPlatform;
    const ur_device_native_properties_t **ppProperties;
    ur_device_handle_t **pphDevice;
} ur_device_create_with_native_handle_params_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Function parameters for urDeviceGetGlobalTimestamps
/// @details Each entry is a pointer to the parameter passed to the function;
///     allowing the callback the ability to modify the parameter's value
typedef struct ur_device_get_global_timestamps_params_t {
    ur_device_handle_t *phDevice;
    uint64_t **ppDeviceTimestamp;
    uint64_t **ppHostTimestamp;
} ur_device_get_global_timestamps_params_t;

#if !defined(__GNUC__)
#pragma endregion
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // UR_API_H_INCLUDED
