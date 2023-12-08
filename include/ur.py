"""
 Copyright (C) 2022 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

 @file ur.py
 @version v0.9-r0

 """
import platform
from ctypes import *
from enum import *

# ctypes does not define c_intptr_t, so let's define it here manually
c_intptr_t = c_ssize_t

###############################################################################
__version__ = "1.0"

###############################################################################
## @brief Defines unique stable identifiers for all functions
class ur_function_v(IntEnum):
    CONTEXT_CREATE = 1                              ## Enumerator for ::urContextCreate
    CONTEXT_RETAIN = 2                              ## Enumerator for ::urContextRetain
    CONTEXT_RELEASE = 3                             ## Enumerator for ::urContextRelease
    CONTEXT_GET_INFO = 4                            ## Enumerator for ::urContextGetInfo
    CONTEXT_GET_NATIVE_HANDLE = 5                   ## Enumerator for ::urContextGetNativeHandle
    CONTEXT_CREATE_WITH_NATIVE_HANDLE = 6           ## Enumerator for ::urContextCreateWithNativeHandle
    CONTEXT_SET_EXTENDED_DELETER = 7                ## Enumerator for ::urContextSetExtendedDeleter
    DEVICE_GET = 8                                  ## Enumerator for ::urDeviceGet
    DEVICE_GET_INFO = 9                             ## Enumerator for ::urDeviceGetInfo
    DEVICE_RETAIN = 10                              ## Enumerator for ::urDeviceRetain
    DEVICE_RELEASE = 11                             ## Enumerator for ::urDeviceRelease
    DEVICE_PARTITION = 12                           ## Enumerator for ::urDevicePartition
    DEVICE_SELECT_BINARY = 13                       ## Enumerator for ::urDeviceSelectBinary
    DEVICE_GET_NATIVE_HANDLE = 14                   ## Enumerator for ::urDeviceGetNativeHandle
    DEVICE_CREATE_WITH_NATIVE_HANDLE = 15           ## Enumerator for ::urDeviceCreateWithNativeHandle
    DEVICE_GET_GLOBAL_TIMESTAMPS = 16               ## Enumerator for ::urDeviceGetGlobalTimestamps
    ENQUEUE_KERNEL_LAUNCH = 17                      ## Enumerator for ::urEnqueueKernelLaunch
    ENQUEUE_EVENTS_WAIT = 18                        ## Enumerator for ::urEnqueueEventsWait
    ENQUEUE_EVENTS_WAIT_WITH_BARRIER = 19           ## Enumerator for ::urEnqueueEventsWaitWithBarrier
    ENQUEUE_MEM_BUFFER_READ = 20                    ## Enumerator for ::urEnqueueMemBufferRead
    ENQUEUE_MEM_BUFFER_WRITE = 21                   ## Enumerator for ::urEnqueueMemBufferWrite
    ENQUEUE_MEM_BUFFER_READ_RECT = 22               ## Enumerator for ::urEnqueueMemBufferReadRect
    ENQUEUE_MEM_BUFFER_WRITE_RECT = 23              ## Enumerator for ::urEnqueueMemBufferWriteRect
    ENQUEUE_MEM_BUFFER_COPY = 24                    ## Enumerator for ::urEnqueueMemBufferCopy
    ENQUEUE_MEM_BUFFER_COPY_RECT = 25               ## Enumerator for ::urEnqueueMemBufferCopyRect
    ENQUEUE_MEM_BUFFER_FILL = 26                    ## Enumerator for ::urEnqueueMemBufferFill
    ENQUEUE_MEM_IMAGE_READ = 27                     ## Enumerator for ::urEnqueueMemImageRead
    ENQUEUE_MEM_IMAGE_WRITE = 28                    ## Enumerator for ::urEnqueueMemImageWrite
    ENQUEUE_MEM_IMAGE_COPY = 29                     ## Enumerator for ::urEnqueueMemImageCopy
    ENQUEUE_MEM_BUFFER_MAP = 30                     ## Enumerator for ::urEnqueueMemBufferMap
    ENQUEUE_MEM_UNMAP = 31                          ## Enumerator for ::urEnqueueMemUnmap
    ENQUEUE_USM_FILL = 32                           ## Enumerator for ::urEnqueueUSMFill
    ENQUEUE_USM_MEMCPY = 33                         ## Enumerator for ::urEnqueueUSMMemcpy
    ENQUEUE_USM_PREFETCH = 34                       ## Enumerator for ::urEnqueueUSMPrefetch
    ENQUEUE_USM_ADVISE = 35                         ## Enumerator for ::urEnqueueUSMAdvise
    ENQUEUE_DEVICE_GLOBAL_VARIABLE_WRITE = 38       ## Enumerator for ::urEnqueueDeviceGlobalVariableWrite
    ENQUEUE_DEVICE_GLOBAL_VARIABLE_READ = 39        ## Enumerator for ::urEnqueueDeviceGlobalVariableRead
    EVENT_GET_INFO = 40                             ## Enumerator for ::urEventGetInfo
    EVENT_GET_PROFILING_INFO = 41                   ## Enumerator for ::urEventGetProfilingInfo
    EVENT_WAIT = 42                                 ## Enumerator for ::urEventWait
    EVENT_RETAIN = 43                               ## Enumerator for ::urEventRetain
    EVENT_RELEASE = 44                              ## Enumerator for ::urEventRelease
    EVENT_GET_NATIVE_HANDLE = 45                    ## Enumerator for ::urEventGetNativeHandle
    EVENT_CREATE_WITH_NATIVE_HANDLE = 46            ## Enumerator for ::urEventCreateWithNativeHandle
    EVENT_SET_CALLBACK = 47                         ## Enumerator for ::urEventSetCallback
    KERNEL_CREATE = 48                              ## Enumerator for ::urKernelCreate
    KERNEL_SET_ARG_VALUE = 49                       ## Enumerator for ::urKernelSetArgValue
    KERNEL_SET_ARG_LOCAL = 50                       ## Enumerator for ::urKernelSetArgLocal
    KERNEL_GET_INFO = 51                            ## Enumerator for ::urKernelGetInfo
    KERNEL_GET_GROUP_INFO = 52                      ## Enumerator for ::urKernelGetGroupInfo
    KERNEL_GET_SUB_GROUP_INFO = 53                  ## Enumerator for ::urKernelGetSubGroupInfo
    KERNEL_RETAIN = 54                              ## Enumerator for ::urKernelRetain
    KERNEL_RELEASE = 55                             ## Enumerator for ::urKernelRelease
    KERNEL_SET_ARG_POINTER = 56                     ## Enumerator for ::urKernelSetArgPointer
    KERNEL_SET_EXEC_INFO = 57                       ## Enumerator for ::urKernelSetExecInfo
    KERNEL_SET_ARG_SAMPLER = 58                     ## Enumerator for ::urKernelSetArgSampler
    KERNEL_SET_ARG_MEM_OBJ = 59                     ## Enumerator for ::urKernelSetArgMemObj
    KERNEL_SET_SPECIALIZATION_CONSTANTS = 60        ## Enumerator for ::urKernelSetSpecializationConstants
    KERNEL_GET_NATIVE_HANDLE = 61                   ## Enumerator for ::urKernelGetNativeHandle
    KERNEL_CREATE_WITH_NATIVE_HANDLE = 62           ## Enumerator for ::urKernelCreateWithNativeHandle
    MEM_IMAGE_CREATE = 63                           ## Enumerator for ::urMemImageCreate
    MEM_BUFFER_CREATE = 64                          ## Enumerator for ::urMemBufferCreate
    MEM_RETAIN = 65                                 ## Enumerator for ::urMemRetain
    MEM_RELEASE = 66                                ## Enumerator for ::urMemRelease
    MEM_BUFFER_PARTITION = 67                       ## Enumerator for ::urMemBufferPartition
    MEM_GET_NATIVE_HANDLE = 68                      ## Enumerator for ::urMemGetNativeHandle
    ENQUEUE_READ_HOST_PIPE = 69                     ## Enumerator for ::urEnqueueReadHostPipe
    MEM_GET_INFO = 70                               ## Enumerator for ::urMemGetInfo
    MEM_IMAGE_GET_INFO = 71                         ## Enumerator for ::urMemImageGetInfo
    PLATFORM_GET = 72                               ## Enumerator for ::urPlatformGet
    PLATFORM_GET_INFO = 73                          ## Enumerator for ::urPlatformGetInfo
    PLATFORM_GET_API_VERSION = 74                   ## Enumerator for ::urPlatformGetApiVersion
    PLATFORM_GET_NATIVE_HANDLE = 75                 ## Enumerator for ::urPlatformGetNativeHandle
    PLATFORM_CREATE_WITH_NATIVE_HANDLE = 76         ## Enumerator for ::urPlatformCreateWithNativeHandle
    PROGRAM_CREATE_WITH_IL = 78                     ## Enumerator for ::urProgramCreateWithIL
    PROGRAM_CREATE_WITH_BINARY = 79                 ## Enumerator for ::urProgramCreateWithBinary
    PROGRAM_BUILD = 80                              ## Enumerator for ::urProgramBuild
    PROGRAM_COMPILE = 81                            ## Enumerator for ::urProgramCompile
    PROGRAM_LINK = 82                               ## Enumerator for ::urProgramLink
    PROGRAM_RETAIN = 83                             ## Enumerator for ::urProgramRetain
    PROGRAM_RELEASE = 84                            ## Enumerator for ::urProgramRelease
    PROGRAM_GET_FUNCTION_POINTER = 85               ## Enumerator for ::urProgramGetFunctionPointer
    PROGRAM_GET_INFO = 86                           ## Enumerator for ::urProgramGetInfo
    PROGRAM_GET_BUILD_INFO = 87                     ## Enumerator for ::urProgramGetBuildInfo
    PROGRAM_SET_SPECIALIZATION_CONSTANTS = 88       ## Enumerator for ::urProgramSetSpecializationConstants
    PROGRAM_GET_NATIVE_HANDLE = 89                  ## Enumerator for ::urProgramGetNativeHandle
    PROGRAM_CREATE_WITH_NATIVE_HANDLE = 90          ## Enumerator for ::urProgramCreateWithNativeHandle
    QUEUE_GET_INFO = 91                             ## Enumerator for ::urQueueGetInfo
    QUEUE_CREATE = 92                               ## Enumerator for ::urQueueCreate
    QUEUE_RETAIN = 93                               ## Enumerator for ::urQueueRetain
    QUEUE_RELEASE = 94                              ## Enumerator for ::urQueueRelease
    QUEUE_GET_NATIVE_HANDLE = 95                    ## Enumerator for ::urQueueGetNativeHandle
    QUEUE_CREATE_WITH_NATIVE_HANDLE = 96            ## Enumerator for ::urQueueCreateWithNativeHandle
    QUEUE_FINISH = 97                               ## Enumerator for ::urQueueFinish
    QUEUE_FLUSH = 98                                ## Enumerator for ::urQueueFlush
    SAMPLER_CREATE = 101                            ## Enumerator for ::urSamplerCreate
    SAMPLER_RETAIN = 102                            ## Enumerator for ::urSamplerRetain
    SAMPLER_RELEASE = 103                           ## Enumerator for ::urSamplerRelease
    SAMPLER_GET_INFO = 104                          ## Enumerator for ::urSamplerGetInfo
    SAMPLER_GET_NATIVE_HANDLE = 105                 ## Enumerator for ::urSamplerGetNativeHandle
    SAMPLER_CREATE_WITH_NATIVE_HANDLE = 106         ## Enumerator for ::urSamplerCreateWithNativeHandle
    USM_HOST_ALLOC = 107                            ## Enumerator for ::urUSMHostAlloc
    USM_DEVICE_ALLOC = 108                          ## Enumerator for ::urUSMDeviceAlloc
    USM_SHARED_ALLOC = 109                          ## Enumerator for ::urUSMSharedAlloc
    USM_FREE = 110                                  ## Enumerator for ::urUSMFree
    USM_GET_MEM_ALLOC_INFO = 111                    ## Enumerator for ::urUSMGetMemAllocInfo
    USM_POOL_CREATE = 112                           ## Enumerator for ::urUSMPoolCreate
    COMMAND_BUFFER_CREATE_EXP = 113                 ## Enumerator for ::urCommandBufferCreateExp
    PLATFORM_GET_BACKEND_OPTION = 114               ## Enumerator for ::urPlatformGetBackendOption
    MEM_BUFFER_CREATE_WITH_NATIVE_HANDLE = 115      ## Enumerator for ::urMemBufferCreateWithNativeHandle
    MEM_IMAGE_CREATE_WITH_NATIVE_HANDLE = 116       ## Enumerator for ::urMemImageCreateWithNativeHandle
    ENQUEUE_WRITE_HOST_PIPE = 117                   ## Enumerator for ::urEnqueueWriteHostPipe
    USM_POOL_RETAIN = 118                           ## Enumerator for ::urUSMPoolRetain
    USM_POOL_RELEASE = 119                          ## Enumerator for ::urUSMPoolRelease
    USM_POOL_GET_INFO = 120                         ## Enumerator for ::urUSMPoolGetInfo
    COMMAND_BUFFER_RETAIN_EXP = 121                 ## Enumerator for ::urCommandBufferRetainExp
    COMMAND_BUFFER_RELEASE_EXP = 122                ## Enumerator for ::urCommandBufferReleaseExp
    COMMAND_BUFFER_FINALIZE_EXP = 123               ## Enumerator for ::urCommandBufferFinalizeExp
    COMMAND_BUFFER_APPEND_KERNEL_LAUNCH_EXP = 125   ## Enumerator for ::urCommandBufferAppendKernelLaunchExp
    COMMAND_BUFFER_ENQUEUE_EXP = 128                ## Enumerator for ::urCommandBufferEnqueueExp
    USM_PITCHED_ALLOC_EXP = 132                     ## Enumerator for ::urUSMPitchedAllocExp
    BINDLESS_IMAGES_UNSAMPLED_IMAGE_HANDLE_DESTROY_EXP = 133## Enumerator for ::urBindlessImagesUnsampledImageHandleDestroyExp
    BINDLESS_IMAGES_SAMPLED_IMAGE_HANDLE_DESTROY_EXP = 134  ## Enumerator for ::urBindlessImagesSampledImageHandleDestroyExp
    BINDLESS_IMAGES_IMAGE_ALLOCATE_EXP = 135        ## Enumerator for ::urBindlessImagesImageAllocateExp
    BINDLESS_IMAGES_IMAGE_FREE_EXP = 136            ## Enumerator for ::urBindlessImagesImageFreeExp
    BINDLESS_IMAGES_UNSAMPLED_IMAGE_CREATE_EXP = 137## Enumerator for ::urBindlessImagesUnsampledImageCreateExp
    BINDLESS_IMAGES_SAMPLED_IMAGE_CREATE_EXP = 138  ## Enumerator for ::urBindlessImagesSampledImageCreateExp
    BINDLESS_IMAGES_IMAGE_COPY_EXP = 139            ## Enumerator for ::urBindlessImagesImageCopyExp
    BINDLESS_IMAGES_IMAGE_GET_INFO_EXP = 140        ## Enumerator for ::urBindlessImagesImageGetInfoExp
    BINDLESS_IMAGES_MIPMAP_GET_LEVEL_EXP = 141      ## Enumerator for ::urBindlessImagesMipmapGetLevelExp
    BINDLESS_IMAGES_MIPMAP_FREE_EXP = 142           ## Enumerator for ::urBindlessImagesMipmapFreeExp
    BINDLESS_IMAGES_IMPORT_OPAQUE_FD_EXP = 143      ## Enumerator for ::urBindlessImagesImportOpaqueFDExp
    BINDLESS_IMAGES_MAP_EXTERNAL_ARRAY_EXP = 144    ## Enumerator for ::urBindlessImagesMapExternalArrayExp
    BINDLESS_IMAGES_RELEASE_INTEROP_EXP = 145       ## Enumerator for ::urBindlessImagesReleaseInteropExp
    BINDLESS_IMAGES_IMPORT_EXTERNAL_SEMAPHORE_OPAQUE_FD_EXP = 146   ## Enumerator for ::urBindlessImagesImportExternalSemaphoreOpaqueFDExp
    BINDLESS_IMAGES_DESTROY_EXTERNAL_SEMAPHORE_EXP = 147## Enumerator for ::urBindlessImagesDestroyExternalSemaphoreExp
    BINDLESS_IMAGES_WAIT_EXTERNAL_SEMAPHORE_EXP = 148   ## Enumerator for ::urBindlessImagesWaitExternalSemaphoreExp
    BINDLESS_IMAGES_SIGNAL_EXTERNAL_SEMAPHORE_EXP = 149 ## Enumerator for ::urBindlessImagesSignalExternalSemaphoreExp
    ENQUEUE_USM_FILL_2D = 151                       ## Enumerator for ::urEnqueueUSMFill2D
    ENQUEUE_USM_MEMCPY_2D = 152                     ## Enumerator for ::urEnqueueUSMMemcpy2D
    VIRTUAL_MEM_GRANULARITY_GET_INFO = 153          ## Enumerator for ::urVirtualMemGranularityGetInfo
    VIRTUAL_MEM_RESERVE = 154                       ## Enumerator for ::urVirtualMemReserve
    VIRTUAL_MEM_FREE = 155                          ## Enumerator for ::urVirtualMemFree
    VIRTUAL_MEM_MAP = 156                           ## Enumerator for ::urVirtualMemMap
    VIRTUAL_MEM_UNMAP = 157                         ## Enumerator for ::urVirtualMemUnmap
    VIRTUAL_MEM_SET_ACCESS = 158                    ## Enumerator for ::urVirtualMemSetAccess
    VIRTUAL_MEM_GET_INFO = 159                      ## Enumerator for ::urVirtualMemGetInfo
    PHYSICAL_MEM_CREATE = 160                       ## Enumerator for ::urPhysicalMemCreate
    PHYSICAL_MEM_RETAIN = 161                       ## Enumerator for ::urPhysicalMemRetain
    PHYSICAL_MEM_RELEASE = 162                      ## Enumerator for ::urPhysicalMemRelease
    USM_IMPORT_EXP = 163                            ## Enumerator for ::urUSMImportExp
    USM_RELEASE_EXP = 164                           ## Enumerator for ::urUSMReleaseExp
    USM_P2P_ENABLE_PEER_ACCESS_EXP = 165            ## Enumerator for ::urUsmP2PEnablePeerAccessExp
    USM_P2P_DISABLE_PEER_ACCESS_EXP = 166           ## Enumerator for ::urUsmP2PDisablePeerAccessExp
    USM_P2P_PEER_ACCESS_GET_INFO_EXP = 167          ## Enumerator for ::urUsmP2PPeerAccessGetInfoExp
    LOADER_CONFIG_CREATE = 172                      ## Enumerator for ::urLoaderConfigCreate
    LOADER_CONFIG_RELEASE = 173                     ## Enumerator for ::urLoaderConfigRelease
    LOADER_CONFIG_RETAIN = 174                      ## Enumerator for ::urLoaderConfigRetain
    LOADER_CONFIG_GET_INFO = 175                    ## Enumerator for ::urLoaderConfigGetInfo
    LOADER_CONFIG_ENABLE_LAYER = 176                ## Enumerator for ::urLoaderConfigEnableLayer
    ADAPTER_RELEASE = 177                           ## Enumerator for ::urAdapterRelease
    ADAPTER_GET = 178                               ## Enumerator for ::urAdapterGet
    ADAPTER_RETAIN = 179                            ## Enumerator for ::urAdapterRetain
    ADAPTER_GET_LAST_ERROR = 180                    ## Enumerator for ::urAdapterGetLastError
    ADAPTER_GET_INFO = 181                          ## Enumerator for ::urAdapterGetInfo
    PROGRAM_BUILD_EXP = 197                         ## Enumerator for ::urProgramBuildExp
    PROGRAM_COMPILE_EXP = 198                       ## Enumerator for ::urProgramCompileExp
    PROGRAM_LINK_EXP = 199                          ## Enumerator for ::urProgramLinkExp
    LOADER_CONFIG_SET_CODE_LOCATION_CALLBACK = 200  ## Enumerator for ::urLoaderConfigSetCodeLocationCallback
    LOADER_INIT = 201                               ## Enumerator for ::urLoaderInit
    LOADER_TEAR_DOWN = 202                          ## Enumerator for ::urLoaderTearDown
    COMMAND_BUFFER_APPEND_USM_MEMCPY_EXP = 203      ## Enumerator for ::urCommandBufferAppendUSMMemcpyExp
    COMMAND_BUFFER_APPEND_USM_FILL_EXP = 204        ## Enumerator for ::urCommandBufferAppendUSMFillExp
    COMMAND_BUFFER_APPEND_MEM_BUFFER_COPY_EXP = 205 ## Enumerator for ::urCommandBufferAppendMemBufferCopyExp
    COMMAND_BUFFER_APPEND_MEM_BUFFER_WRITE_EXP = 206## Enumerator for ::urCommandBufferAppendMemBufferWriteExp
    COMMAND_BUFFER_APPEND_MEM_BUFFER_READ_EXP = 207 ## Enumerator for ::urCommandBufferAppendMemBufferReadExp
    COMMAND_BUFFER_APPEND_MEM_BUFFER_COPY_RECT_EXP = 208## Enumerator for ::urCommandBufferAppendMemBufferCopyRectExp
    COMMAND_BUFFER_APPEND_MEM_BUFFER_WRITE_RECT_EXP = 209   ## Enumerator for ::urCommandBufferAppendMemBufferWriteRectExp
    COMMAND_BUFFER_APPEND_MEM_BUFFER_READ_RECT_EXP = 210## Enumerator for ::urCommandBufferAppendMemBufferReadRectExp
    COMMAND_BUFFER_APPEND_MEM_BUFFER_FILL_EXP = 211 ## Enumerator for ::urCommandBufferAppendMemBufferFillExp
    COMMAND_BUFFER_APPEND_USM_PREFETCH_EXP = 212    ## Enumerator for ::urCommandBufferAppendUSMPrefetchExp
    COMMAND_BUFFER_APPEND_USM_ADVISE_EXP = 213      ## Enumerator for ::urCommandBufferAppendUSMAdviseExp
    ENQUEUE_COOPERATIVE_KERNEL_LAUNCH_EXP = 214     ## Enumerator for ::urEnqueueCooperativeKernelLaunchExp
    KERNEL_SUGGEST_MAX_COOPERATIVE_GROUP_COUNT_EXP = 215## Enumerator for ::urKernelSuggestMaxCooperativeGroupCountExp

class ur_function_t(c_int):
    def __str__(self):
        return str(ur_function_v(self.value))


###############################################################################
## @brief Defines structure types
class ur_structure_type_v(IntEnum):
    CONTEXT_PROPERTIES = 0                          ## ::ur_context_properties_t
    IMAGE_DESC = 1                                  ## ::ur_image_desc_t
    BUFFER_PROPERTIES = 2                           ## ::ur_buffer_properties_t
    BUFFER_REGION = 3                               ## ::ur_buffer_region_t
    BUFFER_CHANNEL_PROPERTIES = 4                   ## ::ur_buffer_channel_properties_t
    BUFFER_ALLOC_LOCATION_PROPERTIES = 5            ## ::ur_buffer_alloc_location_properties_t
    PROGRAM_PROPERTIES = 6                          ## ::ur_program_properties_t
    USM_DESC = 7                                    ## ::ur_usm_desc_t
    USM_HOST_DESC = 8                               ## ::ur_usm_host_desc_t
    USM_DEVICE_DESC = 9                             ## ::ur_usm_device_desc_t
    USM_POOL_DESC = 10                              ## ::ur_usm_pool_desc_t
    USM_POOL_LIMITS_DESC = 11                       ## ::ur_usm_pool_limits_desc_t
    DEVICE_BINARY = 12                              ## ::ur_device_binary_t
    SAMPLER_DESC = 13                               ## ::ur_sampler_desc_t
    QUEUE_PROPERTIES = 14                           ## ::ur_queue_properties_t
    QUEUE_INDEX_PROPERTIES = 15                     ## ::ur_queue_index_properties_t
    CONTEXT_NATIVE_PROPERTIES = 16                  ## ::ur_context_native_properties_t
    KERNEL_NATIVE_PROPERTIES = 17                   ## ::ur_kernel_native_properties_t
    QUEUE_NATIVE_PROPERTIES = 18                    ## ::ur_queue_native_properties_t
    MEM_NATIVE_PROPERTIES = 19                      ## ::ur_mem_native_properties_t
    EVENT_NATIVE_PROPERTIES = 20                    ## ::ur_event_native_properties_t
    PLATFORM_NATIVE_PROPERTIES = 21                 ## ::ur_platform_native_properties_t
    DEVICE_NATIVE_PROPERTIES = 22                   ## ::ur_device_native_properties_t
    PROGRAM_NATIVE_PROPERTIES = 23                  ## ::ur_program_native_properties_t
    SAMPLER_NATIVE_PROPERTIES = 24                  ## ::ur_sampler_native_properties_t
    QUEUE_NATIVE_DESC = 25                          ## ::ur_queue_native_desc_t
    DEVICE_PARTITION_PROPERTIES = 26                ## ::ur_device_partition_properties_t
    KERNEL_ARG_MEM_OBJ_PROPERTIES = 27              ## ::ur_kernel_arg_mem_obj_properties_t
    PHYSICAL_MEM_PROPERTIES = 28                    ## ::ur_physical_mem_properties_t
    KERNEL_ARG_POINTER_PROPERTIES = 29              ## ::ur_kernel_arg_pointer_properties_t
    KERNEL_ARG_SAMPLER_PROPERTIES = 30              ## ::ur_kernel_arg_sampler_properties_t
    KERNEL_EXEC_INFO_PROPERTIES = 31                ## ::ur_kernel_exec_info_properties_t
    KERNEL_ARG_VALUE_PROPERTIES = 32                ## ::ur_kernel_arg_value_properties_t
    KERNEL_ARG_LOCAL_PROPERTIES = 33                ## ::ur_kernel_arg_local_properties_t
    EXP_COMMAND_BUFFER_DESC = 0x1000                ## ::ur_exp_command_buffer_desc_t
    EXP_SAMPLER_MIP_PROPERTIES = 0x2000             ## ::ur_exp_sampler_mip_properties_t
    EXP_INTEROP_MEM_DESC = 0x2001                   ## ::ur_exp_interop_mem_desc_t
    EXP_INTEROP_SEMAPHORE_DESC = 0x2002             ## ::ur_exp_interop_semaphore_desc_t
    EXP_FILE_DESCRIPTOR = 0x2003                    ## ::ur_exp_file_descriptor_t
    EXP_WIN32_HANDLE = 0x2004                       ## ::ur_exp_win32_handle_t
    EXP_LAYERED_IMAGE_PROPERTIES = 0x2005           ## ::ur_exp_layered_image_properties_t
    EXP_SAMPLER_ADDR_MODES = 0x2006                 ## ::ur_exp_sampler_addr_modes_t

class ur_structure_type_t(c_int):
    def __str__(self):
        return str(ur_structure_type_v(self.value))


###############################################################################
## @brief Generates generic 'oneAPI' API versions
def UR_MAKE_VERSION( _major, _minor ):
    return (( _major << 16 )|( _minor & 0x0000ffff))

###############################################################################
## @brief Extracts 'oneAPI' API major version
def UR_MAJOR_VERSION( _ver ):
    return ( _ver >> 16 )

###############################################################################
## @brief Extracts 'oneAPI' API minor version
def UR_MINOR_VERSION( _ver ):
    return ( _ver & 0x0000ffff )

###############################################################################
## @brief Calling convention for all API functions
# UR_APICALL not required for python

###############################################################################
## @brief Microsoft-specific dllexport storage-class attribute
# UR_APIEXPORT not required for python

###############################################################################
## @brief Microsoft-specific dllexport storage-class attribute
# UR_DLLEXPORT not required for python

###############################################################################
## @brief GCC-specific dllexport storage-class attribute
# UR_DLLEXPORT not required for python

###############################################################################
## @brief compiler-independent type
class ur_bool_t(c_ubyte):
    pass

###############################################################################
## @brief Handle of a loader config object
class ur_loader_config_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of an adapter instance
class ur_adapter_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of a platform instance
class ur_platform_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of platform's device object
class ur_device_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of context object
class ur_context_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of event object
class ur_event_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of Program object
class ur_program_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of program's Kernel object
class ur_kernel_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of a queue object
class ur_queue_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of a native object
class ur_native_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of a Sampler object
class ur_sampler_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of memory object which can either be buffer or image
class ur_mem_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of physical memory object
class ur_physical_mem_handle_t(c_void_p):
    pass

###############################################################################
## @brief Generic macro for enumerator bit masks
def UR_BIT( _i ):
    return ( 1 << _i )

###############################################################################
## @brief Defines Return/Error codes
class ur_result_v(IntEnum):
    SUCCESS = 0                                     ## Success
    ERROR_INVALID_OPERATION = 1                     ## Invalid operation
    ERROR_INVALID_QUEUE_PROPERTIES = 2              ## Invalid queue properties
    ERROR_INVALID_QUEUE = 3                         ## Invalid queue
    ERROR_INVALID_VALUE = 4                         ## Invalid Value
    ERROR_INVALID_CONTEXT = 5                       ## Invalid context
    ERROR_INVALID_PLATFORM = 6                      ## Invalid platform
    ERROR_INVALID_BINARY = 7                        ## Invalid binary
    ERROR_INVALID_PROGRAM = 8                       ## Invalid program
    ERROR_INVALID_SAMPLER = 9                       ## Invalid sampler
    ERROR_INVALID_BUFFER_SIZE = 10                  ## Invalid buffer size
    ERROR_INVALID_MEM_OBJECT = 11                   ## Invalid memory object
    ERROR_INVALID_EVENT = 12                        ## Invalid event
    ERROR_INVALID_EVENT_WAIT_LIST = 13              ## Returned when the event wait list or the events in the wait list are
                                                    ## invalid.
    ERROR_MISALIGNED_SUB_BUFFER_OFFSET = 14         ## Misaligned sub buffer offset
    ERROR_INVALID_WORK_GROUP_SIZE = 15              ## Invalid work group size
    ERROR_COMPILER_NOT_AVAILABLE = 16               ## Compiler not available
    ERROR_PROFILING_INFO_NOT_AVAILABLE = 17         ## Profiling info not available
    ERROR_DEVICE_NOT_FOUND = 18                     ## Device not found
    ERROR_INVALID_DEVICE = 19                       ## Invalid device
    ERROR_DEVICE_LOST = 20                          ## Device hung, reset, was removed, or adapter update occurred
    ERROR_DEVICE_REQUIRES_RESET = 21                ## Device requires a reset
    ERROR_DEVICE_IN_LOW_POWER_STATE = 22            ## Device currently in low power state
    ERROR_DEVICE_PARTITION_FAILED = 23              ## Device partitioning failed
    ERROR_INVALID_DEVICE_PARTITION_COUNT = 24       ## Invalid counts provided with ::UR_DEVICE_PARTITION_BY_COUNTS
    ERROR_INVALID_WORK_ITEM_SIZE = 25               ## Invalid work item size
    ERROR_INVALID_WORK_DIMENSION = 26               ## Invalid work dimension
    ERROR_INVALID_KERNEL_ARGS = 27                  ## Invalid kernel args
    ERROR_INVALID_KERNEL = 28                       ## Invalid kernel
    ERROR_INVALID_KERNEL_NAME = 29                  ## [Validation] kernel name is not found in the program
    ERROR_INVALID_KERNEL_ARGUMENT_INDEX = 30        ## [Validation] kernel argument index is not valid for kernel
    ERROR_INVALID_KERNEL_ARGUMENT_SIZE = 31         ## [Validation] kernel argument size does not match kernel
    ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE = 32       ## [Validation] value of kernel attribute is not valid for the kernel or
                                                    ## device
    ERROR_INVALID_IMAGE_SIZE = 33                   ## Invalid image size
    ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR = 34      ## Invalid image format descriptor
    ERROR_IMAGE_FORMAT_NOT_SUPPORTED = 35           ## Image format not supported
    ERROR_MEM_OBJECT_ALLOCATION_FAILURE = 36        ## Memory object allocation failure
    ERROR_INVALID_PROGRAM_EXECUTABLE = 37           ## Program object parameter is invalid.
    ERROR_UNINITIALIZED = 38                        ## [Validation] adapter is not initialized or specific entry-point is not
                                                    ## implemented
    ERROR_OUT_OF_HOST_MEMORY = 39                   ## Insufficient host memory to satisfy call
    ERROR_OUT_OF_DEVICE_MEMORY = 40                 ## Insufficient device memory to satisfy call
    ERROR_OUT_OF_RESOURCES = 41                     ## Out of resources
    ERROR_PROGRAM_BUILD_FAILURE = 42                ## Error occurred when building program, see build log for details
    ERROR_PROGRAM_LINK_FAILURE = 43                 ## Error occurred when linking programs, see build log for details
    ERROR_UNSUPPORTED_VERSION = 44                  ## [Validation] generic error code for unsupported versions
    ERROR_UNSUPPORTED_FEATURE = 45                  ## [Validation] generic error code for unsupported features
    ERROR_INVALID_ARGUMENT = 46                     ## [Validation] generic error code for invalid arguments
    ERROR_INVALID_NULL_HANDLE = 47                  ## [Validation] handle argument is not valid
    ERROR_HANDLE_OBJECT_IN_USE = 48                 ## [Validation] object pointed to by handle still in-use by device
    ERROR_INVALID_NULL_POINTER = 49                 ## [Validation] pointer argument may not be nullptr
    ERROR_INVALID_SIZE = 50                         ## [Validation] invalid size or dimensions (e.g., must not be zero, or is
                                                    ## out of bounds)
    ERROR_UNSUPPORTED_SIZE = 51                     ## [Validation] size argument is not supported by the device (e.g., too
                                                    ## large)
    ERROR_UNSUPPORTED_ALIGNMENT = 52                ## [Validation] alignment argument is not supported by the device (e.g.,
                                                    ## too small)
    ERROR_INVALID_SYNCHRONIZATION_OBJECT = 53       ## [Validation] synchronization object in invalid state
    ERROR_INVALID_ENUMERATION = 54                  ## [Validation] enumerator argument is not valid
    ERROR_UNSUPPORTED_ENUMERATION = 55              ## [Validation] enumerator argument is not supported by the device
    ERROR_UNSUPPORTED_IMAGE_FORMAT = 56             ## [Validation] image format is not supported by the device
    ERROR_INVALID_NATIVE_BINARY = 57                ## [Validation] native binary is not supported by the device
    ERROR_INVALID_GLOBAL_NAME = 58                  ## [Validation] global variable is not found in the program
    ERROR_INVALID_FUNCTION_NAME = 59                ## [Validation] function name is not found in the program
    ERROR_INVALID_GROUP_SIZE_DIMENSION = 60         ## [Validation] group size dimension is not valid for the kernel or
                                                    ## device
    ERROR_INVALID_GLOBAL_WIDTH_DIMENSION = 61       ## [Validation] global width dimension is not valid for the kernel or
                                                    ## device
    ERROR_PROGRAM_UNLINKED = 62                     ## [Validation] compiled program or program with imports needs to be
                                                    ## linked before kernels can be created from it.
    ERROR_OVERLAPPING_REGIONS = 63                  ## [Validation] copy operations do not support overlapping regions of
                                                    ## memory
    ERROR_INVALID_HOST_PTR = 64                     ## Invalid host pointer
    ERROR_INVALID_USM_SIZE = 65                     ## Invalid USM size
    ERROR_OBJECT_ALLOCATION_FAILURE = 66            ## Objection allocation failure
    ERROR_ADAPTER_SPECIFIC = 67                     ## An adapter specific warning/error has been reported and can be
                                                    ## retrieved via the urPlatformGetLastError entry point.
    ERROR_LAYER_NOT_PRESENT = 68                    ## A requested layer was not found by the loader.
    ERROR_INVALID_COMMAND_BUFFER_EXP = 0x1000       ## Invalid Command-Buffer
    ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP = 0x1001## Sync point is not valid for the command-buffer
    ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP = 0x1002  ## Sync point wait list is invalid
    ERROR_UNKNOWN = 0x7ffffffe                      ## Unknown or internal error

class ur_result_t(c_int):
    def __str__(self):
        return str(ur_result_v(self.value))


###############################################################################
## @brief Base for all properties types
class ur_base_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in,out][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief Base for all descriptor types
class ur_base_desc_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief 3D offset argument passed to buffer rect operations
class ur_rect_offset_t(Structure):
    _fields_ = [
        ("x", c_ulonglong),                                             ## [in] x offset (bytes)
        ("y", c_ulonglong),                                             ## [in] y offset (scalar)
        ("z", c_ulonglong)                                              ## [in] z offset (scalar)
    ]

###############################################################################
## @brief 3D region argument passed to buffer rect operations
class ur_rect_region_t(Structure):
    _fields_ = [
        ("width", c_ulonglong),                                         ## [in] width (bytes)
        ("height", c_ulonglong),                                        ## [in] height (scalar)
        ("depth", c_ulonglong)                                          ## [in] scalar (scalar)
    ]

###############################################################################
## @brief Supported device initialization flags
class ur_device_init_flags_v(IntEnum):
    GPU = UR_BIT(0)                                 ## initialize GPU device adapters.
    CPU = UR_BIT(1)                                 ## initialize CPU device adapters.
    FPGA = UR_BIT(2)                                ## initialize FPGA device adapters.
    MCA = UR_BIT(3)                                 ## initialize MCA device adapters.
    VPU = UR_BIT(4)                                 ## initialize VPU device adapters.

class ur_device_init_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Supported loader info
class ur_loader_config_info_v(IntEnum):
    AVAILABLE_LAYERS = 0                            ## [char[]] Null-terminated, semi-colon separated list of available
                                                    ## layers.
    REFERENCE_COUNT = 1                             ## [uint32_t] Reference count of the loader config object.

class ur_loader_config_info_t(c_int):
    def __str__(self):
        return str(ur_loader_config_info_v(self.value))


###############################################################################
## @brief Code location data
class ur_code_location_t(Structure):
    _fields_ = [
        ("functionName", c_char_p),                                     ## [in][out] Function name.
        ("sourceFile", c_char_p),                                       ## [in][out] Source code file.
        ("lineNumber", c_ulong),                                        ## [in][out] Source code line number.
        ("columnNumber", c_ulong)                                       ## [in][out] Source code column number.
    ]

###############################################################################
## @brief Code location callback with user data.
def ur_code_location_callback_t(user_defined_callback):
    @CFUNCTYPE(ur_code_location_t, c_void_p)
    def ur_code_location_callback_t_wrapper(pUserData):
        return user_defined_callback(pUserData)
    return ur_code_location_callback_t_wrapper

###############################################################################
## @brief Supported adapter info
class ur_adapter_info_v(IntEnum):
    BACKEND = 0                                     ## [::ur_adapter_backend_t] Identifies the native backend supported by
                                                    ## the adapter.
    REFERENCE_COUNT = 1                             ## [uint32_t] Reference count of the adapter.
                                                    ## The reference count returned should be considered immediately stale.
                                                    ## It is unsuitable for general use in applications. This feature is
                                                    ## provided for identifying memory leaks.

class ur_adapter_info_t(c_int):
    def __str__(self):
        return str(ur_adapter_info_v(self.value))


###############################################################################
## @brief Identifies backend of the adapter
class ur_adapter_backend_v(IntEnum):
    UNKNOWN = 0                                     ## The backend is not a recognized one
    LEVEL_ZERO = 1                                  ## The backend is Level Zero
    OPENCL = 2                                      ## The backend is OpenCL
    CUDA = 3                                        ## The backend is CUDA
    HIP = 4                                         ## The backend is HIP
    NATIVE_CPU = 5                                  ## The backend is Native CPU

class ur_adapter_backend_t(c_int):
    def __str__(self):
        return str(ur_adapter_backend_v(self.value))


###############################################################################
## @brief Supported platform info
class ur_platform_info_v(IntEnum):
    NAME = 1                                        ## [char[]] The string denoting name of the platform. The size of the
                                                    ## info needs to be dynamically queried.
    VENDOR_NAME = 2                                 ## [char[]] The string denoting name of the vendor of the platform. The
                                                    ## size of the info needs to be dynamically queried.
    VERSION = 3                                     ## [char[]] The string denoting the version of the platform. The size of
                                                    ## the info needs to be dynamically queried.
    EXTENSIONS = 4                                  ## [char[]] The string denoting extensions supported by the platform. The
                                                    ## size of the info needs to be dynamically queried.
    PROFILE = 5                                     ## [char[]] The string denoting profile of the platform. The size of the
                                                    ## info needs to be dynamically queried.
    BACKEND = 6                                     ## [::ur_platform_backend_t] The backend of the platform. Identifies the
                                                    ## native backend adapter implementing this platform.

class ur_platform_info_t(c_int):
    def __str__(self):
        return str(ur_platform_info_v(self.value))


###############################################################################
## @brief Supported API versions
## 
## @details
##     - API versions contain major and minor attributes, use
##       ::UR_MAJOR_VERSION and ::UR_MINOR_VERSION
class ur_api_version_v(IntEnum):
    _0_6 = UR_MAKE_VERSION( 0, 6 )                  ## version 0.6
    _0_7 = UR_MAKE_VERSION( 0, 7 )                  ## version 0.7
    _0_8 = UR_MAKE_VERSION( 0, 8 )                  ## version 0.8
    _0_9 = UR_MAKE_VERSION( 0, 9 )                  ## version 0.9
    CURRENT = UR_MAKE_VERSION( 0, 9 )               ## latest known version

class ur_api_version_t(c_int):
    def __str__(self):
        return str(ur_api_version_v(self.value))


###############################################################################
## @brief Native platform creation properties
class ur_platform_native_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_PLATFORM_NATIVE_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("isNativeHandleOwned", c_bool)                                 ## [in] Indicates UR owns the native handle or if it came from an
                                                                        ## interoperability operation in the application that asked to not
                                                                        ## transfer the ownership to the unified-runtime.
    ]

###############################################################################
## @brief Identifies native backend adapters
class ur_platform_backend_v(IntEnum):
    UNKNOWN = 0                                     ## The backend is not a recognized one
    LEVEL_ZERO = 1                                  ## The backend is Level Zero
    OPENCL = 2                                      ## The backend is OpenCL
    CUDA = 3                                        ## The backend is CUDA
    HIP = 4                                         ## The backend is HIP
    NATIVE_CPU = 5                                  ## The backend is Native CPU

class ur_platform_backend_t(c_int):
    def __str__(self):
        return str(ur_platform_backend_v(self.value))


###############################################################################
## @brief Target identification strings for
##        ::ur_device_binary_t.pDeviceTargetSpec 
##        A device type represented by a particular target triple requires
##        specific 
##        binary images. We need to map the image type onto the device target triple
UR_DEVICE_BINARY_TARGET_UNKNOWN = "<unknown>"

###############################################################################
## @brief SPIR-V 32-bit image <-> "spir", 32-bit OpenCL device
UR_DEVICE_BINARY_TARGET_SPIRV32 = "spir"

###############################################################################
## @brief SPIR-V 64-bit image <-> "spir64", 64-bit OpenCL device
UR_DEVICE_BINARY_TARGET_SPIRV64 = "spir64"

###############################################################################
## @brief Device-specific binary images produced from SPIR-V 64-bit <-> various 
##        "spir64_*" triples for specific 64-bit OpenCL CPU devices
UR_DEVICE_BINARY_TARGET_SPIRV64_X86_64 = "spir64_x86_64"

###############################################################################
## @brief Generic GPU device (64-bit OpenCL)
UR_DEVICE_BINARY_TARGET_SPIRV64_GEN = "spir64_gen"

###############################################################################
## @brief 64-bit OpenCL FPGA device
UR_DEVICE_BINARY_TARGET_SPIRV64_FPGA = "spir64_fpga"

###############################################################################
## @brief PTX 64-bit image <-> "nvptx64", 64-bit NVIDIA PTX device
UR_DEVICE_BINARY_TARGET_NVPTX64 = "nvptx64"

###############################################################################
## @brief AMD GCN
UR_DEVICE_BINARY_TARGET_AMDGCN = "amdgcn"

###############################################################################
## @brief Native CPU
UR_DEVICE_BINARY_TARGET_NATIVE_CPU = "native_cpu"

###############################################################################
## @brief Device Binary Type
class ur_device_binary_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be ::UR_STRUCTURE_TYPE_DEVICE_BINARY
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("pDeviceTargetSpec", c_char_p)                                 ## [in] null-terminated string representation of the device's target architecture.
                                                                        ## For example: 
                                                                        ## + ::UR_DEVICE_BINARY_TARGET_UNKNOWN
                                                                        ## + ::UR_DEVICE_BINARY_TARGET_SPIRV32
                                                                        ## + ::UR_DEVICE_BINARY_TARGET_SPIRV64
                                                                        ## + ::UR_DEVICE_BINARY_TARGET_SPIRV64_X86_64
                                                                        ## + ::UR_DEVICE_BINARY_TARGET_SPIRV64_GEN
                                                                        ## + ::UR_DEVICE_BINARY_TARGET_SPIRV64_FPGA
                                                                        ## + ::UR_DEVICE_BINARY_TARGET_NVPTX64
                                                                        ## + ::UR_DEVICE_BINARY_TARGET_AMDGCN
    ]

###############################################################################
## @brief Supported device types
class ur_device_type_v(IntEnum):
    DEFAULT = 1                                     ## The default device type as preferred by the runtime
    ALL = 2                                         ## Devices of all types
    GPU = 3                                         ## Graphics Processing Unit
    CPU = 4                                         ## Central Processing Unit
    FPGA = 5                                        ## Field Programmable Gate Array
    MCA = 6                                         ## Memory Copy Accelerator
    VPU = 7                                         ## Vision Processing Unit

class ur_device_type_t(c_int):
    def __str__(self):
        return str(ur_device_type_v(self.value))


###############################################################################
## @brief Supported device info
class ur_device_info_v(IntEnum):
    TYPE = 0                                        ## [::ur_device_type_t] type of the device
    VENDOR_ID = 1                                   ## [uint32_t] vendor Id of the device
    DEVICE_ID = 2                                   ## [uint32_t] Id of the device
    MAX_COMPUTE_UNITS = 3                           ## [uint32_t] the number of compute units
    MAX_WORK_ITEM_DIMENSIONS = 4                    ## [uint32_t] max work item dimensions
    MAX_WORK_ITEM_SIZES = 5                         ## [size_t[]] return an array of max work item sizes
    MAX_WORK_GROUP_SIZE = 6                         ## [size_t] max work group size
    SINGLE_FP_CONFIG = 7                            ## [::ur_device_fp_capability_flags_t] single precision floating point
                                                    ## capability
    HALF_FP_CONFIG = 8                              ## [::ur_device_fp_capability_flags_t] half precision floating point
                                                    ## capability
    DOUBLE_FP_CONFIG = 9                            ## [::ur_device_fp_capability_flags_t] double precision floating point
                                                    ## capability
    QUEUE_PROPERTIES = 10                           ## [::ur_queue_flags_t] command queue properties supported by the device
    PREFERRED_VECTOR_WIDTH_CHAR = 11                ## [uint32_t] preferred vector width for char
    PREFERRED_VECTOR_WIDTH_SHORT = 12               ## [uint32_t] preferred vector width for short
    PREFERRED_VECTOR_WIDTH_INT = 13                 ## [uint32_t] preferred vector width for int
    PREFERRED_VECTOR_WIDTH_LONG = 14                ## [uint32_t] preferred vector width for long
    PREFERRED_VECTOR_WIDTH_FLOAT = 15               ## [uint32_t] preferred vector width for float
    PREFERRED_VECTOR_WIDTH_DOUBLE = 16              ## [uint32_t] preferred vector width for double
    PREFERRED_VECTOR_WIDTH_HALF = 17                ## [uint32_t] preferred vector width for half float
    NATIVE_VECTOR_WIDTH_CHAR = 18                   ## [uint32_t] native vector width for char
    NATIVE_VECTOR_WIDTH_SHORT = 19                  ## [uint32_t] native vector width for short
    NATIVE_VECTOR_WIDTH_INT = 20                    ## [uint32_t] native vector width for int
    NATIVE_VECTOR_WIDTH_LONG = 21                   ## [uint32_t] native vector width for long
    NATIVE_VECTOR_WIDTH_FLOAT = 22                  ## [uint32_t] native vector width for float
    NATIVE_VECTOR_WIDTH_DOUBLE = 23                 ## [uint32_t] native vector width for double
    NATIVE_VECTOR_WIDTH_HALF = 24                   ## [uint32_t] native vector width for half float
    MAX_CLOCK_FREQUENCY = 25                        ## [uint32_t] max clock frequency in MHz
    MEMORY_CLOCK_RATE = 26                          ## [uint32_t] memory clock frequency in MHz
    ADDRESS_BITS = 27                               ## [uint32_t] address bits
    MAX_MEM_ALLOC_SIZE = 28                         ## [uint64_t] max memory allocation size
    IMAGE_SUPPORTED = 29                            ## [::ur_bool_t] images are supported
    MAX_READ_IMAGE_ARGS = 30                        ## [uint32_t] max number of image objects arguments of a kernel declared
                                                    ## with the read_only qualifier
    MAX_WRITE_IMAGE_ARGS = 31                       ## [uint32_t] max number of image objects arguments of a kernel declared
                                                    ## with the write_only qualifier
    MAX_READ_WRITE_IMAGE_ARGS = 32                  ## [uint32_t] max number of image objects arguments of a kernel declared
                                                    ## with the read_write qualifier
    IMAGE2D_MAX_WIDTH = 33                          ## [size_t] max width of Image2D object
    IMAGE2D_MAX_HEIGHT = 34                         ## [size_t] max height of Image2D object
    IMAGE3D_MAX_WIDTH = 35                          ## [size_t] max width of Image3D object
    IMAGE3D_MAX_HEIGHT = 36                         ## [size_t] max height of Image3D object
    IMAGE3D_MAX_DEPTH = 37                          ## [size_t] max depth of Image3D object
    IMAGE_MAX_BUFFER_SIZE = 38                      ## [size_t] max image buffer size
    IMAGE_MAX_ARRAY_SIZE = 39                       ## [size_t] max image array size
    MAX_SAMPLERS = 40                               ## [uint32_t] max number of samplers that can be used in a kernel
    MAX_PARAMETER_SIZE = 41                         ## [size_t] max size in bytes of all arguments passed to a kernel
    MEM_BASE_ADDR_ALIGN = 42                        ## [uint32_t] memory base address alignment
    GLOBAL_MEM_CACHE_TYPE = 43                      ## [::ur_device_mem_cache_type_t] global memory cache type
    GLOBAL_MEM_CACHELINE_SIZE = 44                  ## [uint32_t] global memory cache line size in bytes
    GLOBAL_MEM_CACHE_SIZE = 45                      ## [uint64_t] size of global memory cache in bytes
    GLOBAL_MEM_SIZE = 46                            ## [uint64_t] size of global memory in bytes
    GLOBAL_MEM_FREE = 47                            ## [uint64_t] size of global memory which is free in bytes
    MAX_CONSTANT_BUFFER_SIZE = 48                   ## [uint64_t] max constant buffer size in bytes
    MAX_CONSTANT_ARGS = 49                          ## [uint32_t] max number of __const declared arguments in a kernel
    LOCAL_MEM_TYPE = 50                             ## [::ur_device_local_mem_type_t] local memory type
    LOCAL_MEM_SIZE = 51                             ## [uint64_t] local memory size in bytes
    ERROR_CORRECTION_SUPPORT = 52                   ## [::ur_bool_t] support error correction to global and local memory
    HOST_UNIFIED_MEMORY = 53                        ## [::ur_bool_t] unified host device memory
    PROFILING_TIMER_RESOLUTION = 54                 ## [size_t] profiling timer resolution in nanoseconds
    ENDIAN_LITTLE = 55                              ## [::ur_bool_t] little endian byte order
    AVAILABLE = 56                                  ## [::ur_bool_t] device is available
    COMPILER_AVAILABLE = 57                         ## [::ur_bool_t] device compiler is available
    LINKER_AVAILABLE = 58                           ## [::ur_bool_t] device linker is available
    EXECUTION_CAPABILITIES = 59                     ## [::ur_device_exec_capability_flags_t] device kernel execution
                                                    ## capability bit-field
    QUEUE_ON_DEVICE_PROPERTIES = 60                 ## [::ur_queue_flags_t] device command queue property bit-field
    QUEUE_ON_HOST_PROPERTIES = 61                   ## [::ur_queue_flags_t] host queue property bit-field
    BUILT_IN_KERNELS = 62                           ## [char[]] a semi-colon separated list of built-in kernels
    PLATFORM = 63                                   ## [::ur_platform_handle_t] the platform associated with the device
    REFERENCE_COUNT = 64                            ## [uint32_t] Reference count of the device object.
                                                    ## The reference count returned should be considered immediately stale. 
                                                    ## It is unsuitable for general use in applications. This feature is
                                                    ## provided for identifying memory leaks.
    IL_VERSION = 65                                 ## [char[]] IL version
    NAME = 66                                       ## [char[]] Device name
    VENDOR = 67                                     ## [char[]] Device vendor
    DRIVER_VERSION = 68                             ## [char[]] Driver version
    PROFILE = 69                                    ## [char[]] Device profile
    VERSION = 70                                    ## [char[]] Device version
    BACKEND_RUNTIME_VERSION = 71                    ## [char[]] Version of backend runtime
    EXTENSIONS = 72                                 ## [char[]] Return a space separated list of extension names
    PRINTF_BUFFER_SIZE = 73                         ## [size_t] Maximum size in bytes of internal printf buffer
    PREFERRED_INTEROP_USER_SYNC = 74                ## [::ur_bool_t] prefer user synchronization when sharing object with
                                                    ## other API
    PARENT_DEVICE = 75                              ## [::ur_device_handle_t] return parent device handle
    SUPPORTED_PARTITIONS = 76                       ## [::ur_device_partition_t[]] Returns an array of partition types
                                                    ## supported by the device
    PARTITION_MAX_SUB_DEVICES = 77                  ## [uint32_t] maximum number of sub-devices when the device is
                                                    ## partitioned
    PARTITION_AFFINITY_DOMAIN = 78                  ## [::ur_device_affinity_domain_flags_t] Returns a bit-field of the
                                                    ## supported affinity domains for partitioning. 
                                                    ## If the device does not support any affinity domains, then 0 will be returned.
    PARTITION_TYPE = 79                             ## [::ur_device_partition_property_t[]] return an array of
                                                    ## ::ur_device_partition_property_t for properties specified in
                                                    ## ::urDevicePartition
    MAX_NUM_SUB_GROUPS = 80                         ## [uint32_t] max number of sub groups
    SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 81     ## [::ur_bool_t] support sub group independent forward progress
    SUB_GROUP_SIZES_INTEL = 82                      ## [uint32_t[]] return an array of sub group sizes supported on Intel
                                                    ## device
    USM_HOST_SUPPORT = 83                           ## [::ur_device_usm_access_capability_flags_t] support USM host memory
                                                    ## access
    USM_DEVICE_SUPPORT = 84                         ## [::ur_device_usm_access_capability_flags_t] support USM device memory
                                                    ## access
    USM_SINGLE_SHARED_SUPPORT = 85                  ## [::ur_device_usm_access_capability_flags_t] support USM single device
                                                    ## shared memory access
    USM_CROSS_SHARED_SUPPORT = 86                   ## [::ur_device_usm_access_capability_flags_t] support USM cross device
                                                    ## shared memory access
    USM_SYSTEM_SHARED_SUPPORT = 87                  ## [::ur_device_usm_access_capability_flags_t] support USM system wide
                                                    ## shared memory access
    UUID = 88                                       ## [char[]] return device UUID
    PCI_ADDRESS = 89                                ## [char[]] return device PCI address
    GPU_EU_COUNT = 90                               ## [uint32_t] return Intel GPU EU count
    GPU_EU_SIMD_WIDTH = 91                          ## [uint32_t] return Intel GPU EU SIMD width
    GPU_EU_SLICES = 92                              ## [uint32_t] return Intel GPU number of slices
    GPU_EU_COUNT_PER_SUBSLICE = 93                  ## [uint32_t] return Intel GPU EU count per subslice
    GPU_SUBSLICES_PER_SLICE = 94                    ## [uint32_t] return Intel GPU number of subslices per slice
    GPU_HW_THREADS_PER_EU = 95                      ## [uint32_t] return Intel GPU number of threads per EU
    MAX_MEMORY_BANDWIDTH = 96                       ## [uint32_t] return max memory bandwidth in Mb/s
    IMAGE_SRGB = 97                                 ## [::ur_bool_t] device supports sRGB images
    BUILD_ON_SUBDEVICE = 98                         ## [::ur_bool_t] Return true if sub-device should do its own program
                                                    ## build
    ATOMIC_64 = 99                                  ## [::ur_bool_t] support 64 bit atomics
    ATOMIC_MEMORY_ORDER_CAPABILITIES = 100          ## [::ur_memory_order_capability_flags_t] return a bit-field of atomic
                                                    ## memory order capabilities
    ATOMIC_MEMORY_SCOPE_CAPABILITIES = 101          ## [::ur_memory_scope_capability_flags_t] return a bit-field of atomic
                                                    ## memory scope capabilities
    ATOMIC_FENCE_ORDER_CAPABILITIES = 102           ## [::ur_memory_order_capability_flags_t] return a bit-field of atomic
                                                    ## memory fence order capabilities
    ATOMIC_FENCE_SCOPE_CAPABILITIES = 103           ## [::ur_memory_scope_capability_flags_t] return a bit-field of atomic
                                                    ## memory fence scope capabilities
    BFLOAT16 = 104                                  ## [::ur_bool_t] support for bfloat16
    MAX_COMPUTE_QUEUE_INDICES = 105                 ## [uint32_t] Returns 1 if the device doesn't have a notion of a 
                                                    ## queue index. Otherwise, returns the number of queue indices that are
                                                    ## available for this device.
    KERNEL_SET_SPECIALIZATION_CONSTANTS = 106       ## [::ur_bool_t] support the ::urKernelSetSpecializationConstants entry
                                                    ## point
    MEMORY_BUS_WIDTH = 107                          ## [uint32_t] return the width in bits of the memory bus interface of the
                                                    ## device.
    MAX_WORK_GROUPS_3D = 108                        ## [size_t[3]] return max 3D work groups
    ASYNC_BARRIER = 109                             ## [::ur_bool_t] return true if Async Barrier is supported
    MEM_CHANNEL_SUPPORT = 110                       ## [::ur_bool_t] return true if specifying memory channels is supported
    HOST_PIPE_READ_WRITE_SUPPORTED = 111            ## [::ur_bool_t] Return true if the device supports enqueueing commands
                                                    ## to read and write pipes from the host.
    MAX_REGISTERS_PER_WORK_GROUP = 112              ## [uint32_t] The maximum number of registers available per block.
    IP_VERSION = 113                                ## [uint32_t] The device IP version. The meaning of the device IP version
                                                    ## is implementation-defined, but newer devices should have a higher
                                                    ## version than older devices.
    VIRTUAL_MEMORY_SUPPORT = 114                    ## [::ur_bool_t] return true if the device supports virtual memory.
    ESIMD_SUPPORT = 115                             ## [::ur_bool_t] return true if the device supports ESIMD.
    BINDLESS_IMAGES_SUPPORT_EXP = 0x2000            ## [::ur_bool_t] returns true if the device supports the creation of
                                                    ## bindless images
    BINDLESS_IMAGES_SHARED_USM_SUPPORT_EXP = 0x2001 ## [::ur_bool_t] returns true if the device supports the creation of
                                                    ## bindless images backed by shared USM
    BINDLESS_IMAGES_1D_USM_SUPPORT_EXP = 0x2002     ## [::ur_bool_t] returns true if the device supports the creation of 1D
                                                    ## bindless images backed by USM
    BINDLESS_IMAGES_2D_USM_SUPPORT_EXP = 0x2003     ## [::ur_bool_t] returns true if the device supports the creation of 2D
                                                    ## bindless images backed by USM
    IMAGE_PITCH_ALIGN_EXP = 0x2004                  ## [uint32_t] returns the required alignment of the pitch between two
                                                    ## rows of an image in bytes
    MAX_IMAGE_LINEAR_WIDTH_EXP = 0x2005             ## [size_t] returns the maximum linear width allowed for images allocated
                                                    ## using USM
    MAX_IMAGE_LINEAR_HEIGHT_EXP = 0x2006            ## [size_t] returns the maximum linear height allowed for images
                                                    ## allocated using USM
    MAX_IMAGE_LINEAR_PITCH_EXP = 0x2007             ## [size_t] returns the maximum linear pitch allowed for images allocated
                                                    ## using USM
    MIPMAP_SUPPORT_EXP = 0x2008                     ## [::ur_bool_t] returns true if the device supports allocating mipmap
                                                    ## resources
    MIPMAP_ANISOTROPY_SUPPORT_EXP = 0x2009          ## [::ur_bool_t] returns true if the device supports sampling mipmap
                                                    ## images with anisotropic filtering
    MIPMAP_MAX_ANISOTROPY_EXP = 0x200A              ## [uint32_t] returns the maximum anisotropic ratio supported by the
                                                    ## device
    MIPMAP_LEVEL_REFERENCE_SUPPORT_EXP = 0x200B     ## [::ur_bool_t] returns true if the device supports using images created
                                                    ## from individual mipmap levels
    INTEROP_MEMORY_IMPORT_SUPPORT_EXP = 0x200C      ## [::ur_bool_t] returns true if the device supports importing external
                                                    ## memory resources
    INTEROP_MEMORY_EXPORT_SUPPORT_EXP = 0x200D      ## [::ur_bool_t] returns true if the device supports exporting internal
                                                    ## memory resources
    INTEROP_SEMAPHORE_IMPORT_SUPPORT_EXP = 0x200E   ## [::ur_bool_t] returns true if the device supports importing external
                                                    ## semaphore resources
    INTEROP_SEMAPHORE_EXPORT_SUPPORT_EXP = 0x200F   ## [::ur_bool_t] returns true if the device supports exporting internal
                                                    ## event resources

class ur_device_info_t(c_int):
    def __str__(self):
        return str(ur_device_info_v(self.value))


###############################################################################
## @brief Device affinity domain
class ur_device_affinity_domain_flags_v(IntEnum):
    NUMA = UR_BIT(0)                                ## Split the device into sub devices comprised of compute units that
                                                    ## share a NUMA node.
    L4_CACHE = UR_BIT(1)                            ## Split the device into sub devices comprised of compute units that
                                                    ## share a level 4 data cache.
    L3_CACHE = UR_BIT(2)                            ## Split the device into sub devices comprised of compute units that
                                                    ## share a level 3 data cache.
    L2_CACHE = UR_BIT(3)                            ## Split the device into sub devices comprised of compute units that
                                                    ## share a level 2 data cache.
    L1_CACHE = UR_BIT(4)                            ## Split the device into sub devices comprised of compute units that
                                                    ## share a level 1 data cache.
    NEXT_PARTITIONABLE = UR_BIT(5)                  ## Split the device along the next partitionable affinity domain. 
                                                    ## The implementation shall find the first level along which the device
                                                    ## or sub device may be further subdivided in the order: 
                                                    ## ::UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA,
                                                    ## ::UR_DEVICE_AFFINITY_DOMAIN_FLAG_L4_CACHE,
                                                    ## ::UR_DEVICE_AFFINITY_DOMAIN_FLAG_L3_CACHE,
                                                    ## ::UR_DEVICE_AFFINITY_DOMAIN_FLAG_L2_CACHE,
                                                    ## ::UR_DEVICE_AFFINITY_DOMAIN_FLAG_L1_CACHE, 
                                                    ## and partition the device into sub devices comprised of compute units
                                                    ## that share memory subsystems at this level.

class ur_device_affinity_domain_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Partition Properties
class ur_device_partition_v(IntEnum):
    EQUALLY = 0x1086                                ## Partition Equally
    BY_COUNTS = 0x1087                              ## Partition by counts
    BY_AFFINITY_DOMAIN = 0x1088                     ## Partition by affinity domain
    BY_CSLICE = 0x1089                              ## Partition by c-slice

class ur_device_partition_t(c_int):
    def __str__(self):
        return str(ur_device_partition_v(self.value))


###############################################################################
## @brief Device partition value.
class ur_device_partition_value_t(Structure):
    _fields_ = [
        ("equally", c_ulong),                                           ## [in] Number of compute units per sub-device when partitioning with
                                                                        ## ::UR_DEVICE_PARTITION_EQUALLY.
        ("count", c_ulong),                                             ## [in] Number of compute units in a sub-device when partitioning with
                                                                        ## ::UR_DEVICE_PARTITION_BY_COUNTS.
        ("affinity_domain", ur_device_affinity_domain_flags_t)          ## [in] The affinity domain to partition for when partitioning with
                                                                        ## ::UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN.
    ]

###############################################################################
## @brief Device partition property
class ur_device_partition_property_t(Structure):
    _fields_ = [
        ("type", ur_device_partition_t),                                ## [in] The partitioning type to be used.
        ("value", ur_device_partition_value_t)                          ## [in][tagged_by(type)] The partitioning value.
    ]

###############################################################################
## @brief Device Partition Properties
class ur_device_partition_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("pProperties", POINTER(ur_device_partition_property_t)),       ## [in] Pointer to the beginning of the properties array.
        ("PropCount", c_size_t)                                         ## [in] The length of properties pointed to by `pProperties`.
    ]

###############################################################################
## @brief FP capabilities
class ur_device_fp_capability_flags_v(IntEnum):
    CORRECTLY_ROUNDED_DIVIDE_SQRT = UR_BIT(0)       ## Support correctly rounded divide and sqrt
    ROUND_TO_NEAREST = UR_BIT(1)                    ## Support round to nearest
    ROUND_TO_ZERO = UR_BIT(2)                       ## Support round to zero
    ROUND_TO_INF = UR_BIT(3)                        ## Support round to infinity
    INF_NAN = UR_BIT(4)                             ## Support INF to NAN
    DENORM = UR_BIT(5)                              ## Support denorm
    FMA = UR_BIT(6)                                 ## Support FMA
    SOFT_FLOAT = UR_BIT(7)                          ## Basic floating point operations implemented in software.

class ur_device_fp_capability_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device memory cache type
class ur_device_mem_cache_type_v(IntEnum):
    NONE = 0                                        ## Has none cache
    READ_ONLY_CACHE = 1                             ## Has read only cache
    READ_WRITE_CACHE = 2                            ## Has read write cache

class ur_device_mem_cache_type_t(c_int):
    def __str__(self):
        return str(ur_device_mem_cache_type_v(self.value))


###############################################################################
## @brief Device local memory type
class ur_device_local_mem_type_v(IntEnum):
    NONE = 0                                        ## No local memory support
    LOCAL = 1                                       ## Dedicated local memory
    GLOBAL = 2                                      ## Global memory

class ur_device_local_mem_type_t(c_int):
    def __str__(self):
        return str(ur_device_local_mem_type_v(self.value))


###############################################################################
## @brief Device kernel execution capability
class ur_device_exec_capability_flags_v(IntEnum):
    KERNEL = UR_BIT(0)                              ## Support kernel execution
    NATIVE_KERNEL = UR_BIT(1)                       ## Support native kernel execution

class ur_device_exec_capability_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Native device creation properties
class ur_device_native_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_DEVICE_NATIVE_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("isNativeHandleOwned", c_bool)                                 ## [in] Indicates UR owns the native handle or if it came from an
                                                                        ## interoperability operation in the application that asked to not
                                                                        ## transfer the ownership to the unified-runtime.
    ]

###############################################################################
## @brief Memory order capabilities
class ur_memory_order_capability_flags_v(IntEnum):
    RELAXED = UR_BIT(0)                             ## Relaxed memory ordering
    ACQUIRE = UR_BIT(1)                             ## Acquire memory ordering
    RELEASE = UR_BIT(2)                             ## Release memory ordering
    ACQ_REL = UR_BIT(3)                             ## Acquire/release memory ordering
    SEQ_CST = UR_BIT(4)                             ## Sequentially consistent memory ordering

class ur_memory_order_capability_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Memory scope capabilities
class ur_memory_scope_capability_flags_v(IntEnum):
    WORK_ITEM = UR_BIT(0)                           ## Work item scope
    SUB_GROUP = UR_BIT(1)                           ## Sub group scope
    WORK_GROUP = UR_BIT(2)                          ## Work group scope
    DEVICE = UR_BIT(3)                              ## Device scope
    SYSTEM = UR_BIT(4)                              ## System scope

class ur_memory_scope_capability_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief USM access capabilities
class ur_device_usm_access_capability_flags_v(IntEnum):
    ACCESS = UR_BIT(0)                              ## Memory can be accessed
    ATOMIC_ACCESS = UR_BIT(1)                       ## Memory can be accessed atomically
    CONCURRENT_ACCESS = UR_BIT(2)                   ## Memory can be accessed concurrently
    ATOMIC_CONCURRENT_ACCESS = UR_BIT(3)            ## Memory can be accessed atomically and concurrently

class ur_device_usm_access_capability_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Context property type
class ur_context_flags_v(IntEnum):
    TBD = UR_BIT(0)                                 ## reserved for future use

class ur_context_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Context creation properties
class ur_context_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_CONTEXT_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("flags", ur_context_flags_t)                                   ## [in] context creation flags.
    ]

###############################################################################
## @brief Supported context info
class ur_context_info_v(IntEnum):
    NUM_DEVICES = 0                                 ## [uint32_t] The number of the devices in the context
    DEVICES = 1                                     ## [::ur_device_handle_t[]] The array of the device handles in the
                                                    ## context
    REFERENCE_COUNT = 2                             ## [uint32_t] Reference count of the context object.
                                                    ## The reference count returned should be considered immediately stale. 
                                                    ## It is unsuitable for general use in applications. This feature is
                                                    ## provided for identifying memory leaks.
    USM_MEMCPY2D_SUPPORT = 3                        ## [::ur_bool_t] to indicate if the ::urEnqueueUSMMemcpy2D entrypoint is
                                                    ## supported.
    USM_FILL2D_SUPPORT = 4                          ## [::ur_bool_t] to indicate if the ::urEnqueueUSMFill2D entrypoint is
                                                    ## supported.
    ATOMIC_MEMORY_ORDER_CAPABILITIES = 5            ## [::ur_memory_order_capability_flags_t] return a bit-field of atomic
                                                    ## memory order capabilities.
    ATOMIC_MEMORY_SCOPE_CAPABILITIES = 6            ## [::ur_memory_scope_capability_flags_t] return a bit-field of atomic
                                                    ## memory scope capabilities.
    ATOMIC_FENCE_ORDER_CAPABILITIES = 7             ## [::ur_memory_order_capability_flags_t] return a bit-field of atomic
                                                    ## memory fence order capabilities. 
                                                    ## Zero is returned if the backend does not support context-level fences.
    ATOMIC_FENCE_SCOPE_CAPABILITIES = 8             ## [::ur_memory_scope_capability_flags_t] return a bit-field of atomic
                                                    ## memory fence scope capabilities. 
                                                    ## Zero is returned if the backend does not support context-level fences.

class ur_context_info_t(c_int):
    def __str__(self):
        return str(ur_context_info_v(self.value))


###############################################################################
## @brief Properties for for ::urContextCreateWithNativeHandle.
class ur_context_native_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_CONTEXT_NATIVE_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("isNativeHandleOwned", c_bool)                                 ## [in] Indicates UR owns the native handle or if it came from an interoperability
                                                                        ## operation in the application that asked to not transfer the ownership to
                                                                        ## the unified-runtime.
    ]

###############################################################################
## @brief Context's extended deleter callback function with user data.
def ur_context_extended_deleter_t(user_defined_callback):
    @CFUNCTYPE(None, c_void_p)
    def ur_context_extended_deleter_t_wrapper(pUserData):
        return user_defined_callback(pUserData)
    return ur_context_extended_deleter_t_wrapper

###############################################################################
## @brief Memory flags
class ur_mem_flags_v(IntEnum):
    READ_WRITE = UR_BIT(0)                          ## The memory object will be read and written by a kernel. This is the
                                                    ## default
    WRITE_ONLY = UR_BIT(1)                          ## The memory object will be written but not read by a kernel
    READ_ONLY = UR_BIT(2)                           ## The memory object is a read-only inside a kernel
    USE_HOST_POINTER = UR_BIT(3)                    ## Use memory pointed by a host pointer parameter as the storage bits for
                                                    ## the memory object
    ALLOC_HOST_POINTER = UR_BIT(4)                  ## Allocate memory object from host accessible memory
    ALLOC_COPY_HOST_POINTER = UR_BIT(5)             ## Allocate memory and copy the data from host pointer pointed memory

class ur_mem_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Memory types
class ur_mem_type_v(IntEnum):
    BUFFER = 0                                      ## Buffer object
    IMAGE2D = 1                                     ## 2D image object
    IMAGE3D = 2                                     ## 3D image object
    IMAGE2D_ARRAY = 3                               ## 2D image array object
    IMAGE1D = 4                                     ## 1D image object
    IMAGE1D_ARRAY = 5                               ## 1D image array object
    IMAGE1D_BUFFER = 6                              ## 1D image buffer object

class ur_mem_type_t(c_int):
    def __str__(self):
        return str(ur_mem_type_v(self.value))


###############################################################################
## @brief Memory Information type
class ur_mem_info_v(IntEnum):
    SIZE = 0                                        ## [size_t] actual size of of memory object in bytes
    CONTEXT = 1                                     ## [::ur_context_handle_t] context in which the memory object was created

class ur_mem_info_t(c_int):
    def __str__(self):
        return str(ur_mem_info_v(self.value))


###############################################################################
## @brief Image channel order info: number of channels and the channel layout
class ur_image_channel_order_v(IntEnum):
    A = 0                                           ## channel order A
    R = 1                                           ## channel order R
    RG = 2                                          ## channel order RG
    RA = 3                                          ## channel order RA
    RGB = 4                                         ## channel order RGB
    RGBA = 5                                        ## channel order RGBA
    BGRA = 6                                        ## channel order BGRA
    ARGB = 7                                        ## channel order ARGB
    ABGR = 8                                        ## channel order ABGR
    INTENSITY = 9                                   ## channel order intensity
    LUMINANCE = 10                                  ## channel order luminance
    RX = 11                                         ## channel order Rx
    RGX = 12                                        ## channel order RGx
    RGBX = 13                                       ## channel order RGBx
    SRGBA = 14                                      ## channel order sRGBA

class ur_image_channel_order_t(c_int):
    def __str__(self):
        return str(ur_image_channel_order_v(self.value))


###############################################################################
## @brief Image channel type info: describe the size of the channel data type
class ur_image_channel_type_v(IntEnum):
    SNORM_INT8 = 0                                  ## channel type snorm int8
    SNORM_INT16 = 1                                 ## channel type snorm int16
    UNORM_INT8 = 2                                  ## channel type unorm int8
    UNORM_INT16 = 3                                 ## channel type unorm int16
    UNORM_SHORT_565 = 4                             ## channel type unorm short 565
    UNORM_SHORT_555 = 5                             ## channel type unorm short 555
    INT_101010 = 6                                  ## channel type int 101010
    SIGNED_INT8 = 7                                 ## channel type signed int8
    SIGNED_INT16 = 8                                ## channel type signed int16
    SIGNED_INT32 = 9                                ## channel type signed int32
    UNSIGNED_INT8 = 10                              ## channel type unsigned int8
    UNSIGNED_INT16 = 11                             ## channel type unsigned int16
    UNSIGNED_INT32 = 12                             ## channel type unsigned int32
    HALF_FLOAT = 13                                 ## channel type half float
    FLOAT = 14                                      ## channel type float

class ur_image_channel_type_t(c_int):
    def __str__(self):
        return str(ur_image_channel_type_v(self.value))


###############################################################################
## @brief Image information types
class ur_image_info_v(IntEnum):
    FORMAT = 0                                      ## [::ur_image_format_t] image format
    ELEMENT_SIZE = 1                                ## [size_t] element size
    ROW_PITCH = 2                                   ## [size_t] row pitch
    SLICE_PITCH = 3                                 ## [size_t] slice pitch
    WIDTH = 4                                       ## [size_t] image width
    HEIGHT = 5                                      ## [size_t] image height
    DEPTH = 6                                       ## [size_t] image depth

class ur_image_info_t(c_int):
    def __str__(self):
        return str(ur_image_info_v(self.value))


###############################################################################
## @brief Image format including channel layout and data type
class ur_image_format_t(Structure):
    _fields_ = [
        ("channelOrder", ur_image_channel_order_t),                     ## [in] image channel order
        ("channelType", ur_image_channel_type_t)                        ## [in] image channel type
    ]

###############################################################################
## @brief Image descriptor type.
class ur_image_desc_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be ::UR_STRUCTURE_TYPE_IMAGE_DESC
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("type", ur_mem_type_t),                                        ## [in][nocheck] memory object type
        ("width", c_size_t),                                            ## [in] image width
        ("height", c_size_t),                                           ## [in] image height
        ("depth", c_size_t),                                            ## [in] image depth
        ("arraySize", c_size_t),                                        ## [in] image array size
        ("rowPitch", c_size_t),                                         ## [in] image row pitch
        ("slicePitch", c_size_t),                                       ## [in] image slice pitch
        ("numMipLevel", c_ulong),                                       ## [in] number of MIP levels
        ("numSamples", c_ulong)                                         ## [in] number of samples
    ]

###############################################################################
## @brief Buffer creation properties
class ur_buffer_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_BUFFER_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("pHost", c_void_p)                                             ## [in][optional] pointer to the buffer data
    ]

###############################################################################
## @brief Buffer memory channel creation properties
## 
## @details
##     - Specify these properties in ::urMemBufferCreate via
##       ::ur_buffer_properties_t as part of a `pNext` chain.
## 
## @remarks
##   _Analogues_
##     - cl_intel_mem_channel_property
class ur_buffer_channel_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_BUFFER_CHANNEL_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("channel", c_ulong)                                            ## [in] Identifies the channel/region to which the buffer should be mapped.
    ]

###############################################################################
## @brief Buffer allocation location creation properties
## 
## @details
##     - Specify these properties in ::urMemBufferCreate via
##       ::ur_buffer_properties_t as part of a `pNext` chain.
## 
## @remarks
##   _Analogues_
##     - cl_intel_mem_alloc_buffer_location
class ur_buffer_alloc_location_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_BUFFER_ALLOC_LOCATION_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("location", c_ulong)                                           ## [in] Identifies the ID of global memory partition to which the memory
                                                                        ## should be allocated.
    ]

###############################################################################
## @brief Buffer region type, used to describe a sub buffer
class ur_buffer_region_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be ::UR_STRUCTURE_TYPE_BUFFER_REGION
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("origin", c_size_t),                                           ## [in] buffer origin offset
        ("size", c_size_t)                                              ## [in] size of the buffer region
    ]

###############################################################################
## @brief Buffer creation type
class ur_buffer_create_type_v(IntEnum):
    REGION = 0                                      ## buffer create type is region

class ur_buffer_create_type_t(c_int):
    def __str__(self):
        return str(ur_buffer_create_type_v(self.value))


###############################################################################
## @brief Native memory object creation properties
class ur_mem_native_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_MEM_NATIVE_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("isNativeHandleOwned", c_bool)                                 ## [in] Indicates UR owns the native handle or if it came from an
                                                                        ## interoperability operation in the application that asked to not
                                                                        ## transfer the ownership to the unified-runtime.
    ]

###############################################################################
## @brief Sampler Filter Mode
class ur_sampler_filter_mode_v(IntEnum):
    NEAREST = 0                                     ## Filter mode nearest.
    LINEAR = 1                                      ## Filter mode linear.

class ur_sampler_filter_mode_t(c_int):
    def __str__(self):
        return str(ur_sampler_filter_mode_v(self.value))


###############################################################################
## @brief Sampler addressing mode
class ur_sampler_addressing_mode_v(IntEnum):
    NONE = 0                                        ## None
    CLAMP_TO_EDGE = 1                               ## Clamp to edge
    CLAMP = 2                                       ## Clamp
    REPEAT = 3                                      ## Repeat
    MIRRORED_REPEAT = 4                             ## Mirrored Repeat

class ur_sampler_addressing_mode_t(c_int):
    def __str__(self):
        return str(ur_sampler_addressing_mode_v(self.value))


###############################################################################
## @brief Get sample object information
class ur_sampler_info_v(IntEnum):
    REFERENCE_COUNT = 0                             ## [uint32_t] Reference count of the sampler object.
                                                    ## The reference count returned should be considered immediately stale. 
                                                    ## It is unsuitable for general use in applications. This feature is
                                                    ## provided for identifying memory leaks.
    CONTEXT = 1                                     ## [::ur_context_handle_t] Sampler context info
    NORMALIZED_COORDS = 2                           ## [::ur_bool_t] Sampler normalized coordinate setting
    ADDRESSING_MODE = 3                             ## [::ur_sampler_addressing_mode_t] Sampler addressing mode setting
    FILTER_MODE = 4                                 ## [::ur_sampler_filter_mode_t] Sampler filter mode setting

class ur_sampler_info_t(c_int):
    def __str__(self):
        return str(ur_sampler_info_v(self.value))


###############################################################################
## @brief Sampler description.
class ur_sampler_desc_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be ::UR_STRUCTURE_TYPE_SAMPLER_DESC
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("normalizedCoords", c_bool),                                   ## [in] Specify if image coordinates are normalized (true) or not (false)
        ("addressingMode", ur_sampler_addressing_mode_t),               ## [in] Specify the address mode of the sampler
        ("filterMode", ur_sampler_filter_mode_t)                        ## [in] Specify the filter mode of the sampler
    ]

###############################################################################
## @brief Native sampler creation properties
class ur_sampler_native_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_SAMPLER_NATIVE_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("isNativeHandleOwned", c_bool)                                 ## [in] Indicates UR owns the native handle or if it came from an
                                                                        ## interoperability operation in the application that asked to not
                                                                        ## transfer the ownership to the unified-runtime.
    ]

###############################################################################
## @brief USM host memory property flags
class ur_usm_host_mem_flags_v(IntEnum):
    INITIAL_PLACEMENT = UR_BIT(0)                   ## Optimize shared allocation for first access on the host

class ur_usm_host_mem_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief USM device memory property flags
class ur_usm_device_mem_flags_v(IntEnum):
    WRITE_COMBINED = UR_BIT(0)                      ## Memory should be allocated write-combined (WC)
    INITIAL_PLACEMENT = UR_BIT(1)                   ## Optimize shared allocation for first access on the device
    DEVICE_READ_ONLY = UR_BIT(2)                    ## Memory is only possibly modified from the host, but read-only in all
                                                    ## device code

class ur_usm_device_mem_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief USM memory property flags
class ur_usm_pool_flags_v(IntEnum):
    ZERO_INITIALIZE_BLOCK = UR_BIT(0)               ## All coarse-grain allocations (allocations from the driver) will be
                                                    ## zero-initialized.

class ur_usm_pool_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief USM allocation type
class ur_usm_type_v(IntEnum):
    UNKNOWN = 0                                     ## Unknown USM type
    HOST = 1                                        ## Host USM type
    DEVICE = 2                                      ## Device USM type
    SHARED = 3                                      ## Shared USM type

class ur_usm_type_t(c_int):
    def __str__(self):
        return str(ur_usm_type_v(self.value))


###############################################################################
## @brief USM memory allocation information type
class ur_usm_alloc_info_v(IntEnum):
    TYPE = 0                                        ## [::ur_usm_type_t] Memory allocation type info
    BASE_PTR = 1                                    ## [void *] Memory allocation base pointer info
    SIZE = 2                                        ## [size_t] Memory allocation size info
    DEVICE = 3                                      ## [::ur_device_handle_t] Memory allocation device info
    POOL = 4                                        ## [::ur_usm_pool_handle_t] Memory allocation pool info

class ur_usm_alloc_info_t(c_int):
    def __str__(self):
        return str(ur_usm_alloc_info_v(self.value))


###############################################################################
## @brief USM memory advice
class ur_usm_advice_flags_v(IntEnum):
    DEFAULT = UR_BIT(0)                             ## The USM memory advice is default
    SET_READ_MOSTLY = UR_BIT(1)                     ## Hint that memory will be read from frequently and written to rarely
    CLEAR_READ_MOSTLY = UR_BIT(2)                   ## Removes the affect of ::UR_USM_ADVICE_FLAG_SET_READ_MOSTLY
    SET_PREFERRED_LOCATION = UR_BIT(3)              ## Hint that the preferred memory location is the specified device
    CLEAR_PREFERRED_LOCATION = UR_BIT(4)            ## Removes the affect of ::UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION
    SET_NON_ATOMIC_MOSTLY = UR_BIT(5)               ## Hint that memory will mostly be accessed non-atomically
    CLEAR_NON_ATOMIC_MOSTLY = UR_BIT(6)             ## Removes the affect of ::UR_USM_ADVICE_FLAG_SET_NON_ATOMIC_MOSTLY
    BIAS_CACHED = UR_BIT(7)                         ## Hint that memory should be cached
    BIAS_UNCACHED = UR_BIT(8)                       ## Hint that memory should be not be cached
    SET_ACCESSED_BY_DEVICE = UR_BIT(9)              ## Hint that memory will be mostly accessed by the specified device
    CLEAR_ACCESSED_BY_DEVICE = UR_BIT(10)           ## Removes the affect of ::UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE
    SET_ACCESSED_BY_HOST = UR_BIT(11)               ## Hint that memory will be mostly accessed by the host
    CLEAR_ACCESSED_BY_HOST = UR_BIT(12)             ## Removes the affect of ::UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_HOST
    SET_PREFERRED_LOCATION_HOST = UR_BIT(13)        ## Hint that the preferred memory location is the host
    CLEAR_PREFERRED_LOCATION_HOST = UR_BIT(14)      ## Removes the affect of ::UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION_HOST

class ur_usm_advice_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Handle of USM pool
class ur_usm_pool_handle_t(c_void_p):
    pass

###############################################################################
## @brief USM allocation descriptor type.
class ur_usm_desc_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be ::UR_STRUCTURE_TYPE_USM_DESC
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("hints", ur_usm_advice_flags_t),                               ## [in] Memory advice hints
        ("align", c_ulong)                                              ## [in] alignment of the USM memory object
                                                                        ## Must be zero or a power of 2.
                                                                        ## Must be equal to or smaller than the size of the largest data type
                                                                        ## supported by `hDevice`.
    ]

###############################################################################
## @brief USM host allocation descriptor type.
## 
## @details
##     - Specify these properties in ::urUSMHostAlloc and ::urUSMSharedAlloc
##       via ::ur_usm_desc_t as part of a `pNext` chain.
class ur_usm_host_desc_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be ::UR_STRUCTURE_TYPE_USM_HOST_DESC
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("flags", ur_usm_host_mem_flags_t)                              ## [in] host memory allocation flags
    ]

###############################################################################
## @brief USM device allocation descriptor type.
## 
## @details
##     - Specify these properties in ::urUSMDeviceAlloc and ::urUSMSharedAlloc
##       via ::ur_usm_desc_t as part of a `pNext` chain.
class ur_usm_device_desc_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_USM_DEVICE_DESC
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("flags", ur_usm_device_mem_flags_t)                            ## [in] device memory allocation flags.
    ]

###############################################################################
## @brief USM pool descriptor type
class ur_usm_pool_desc_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be ::UR_STRUCTURE_TYPE_USM_POOL_DESC
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("flags", ur_usm_pool_flags_t)                                  ## [in] memory allocation flags
    ]

###############################################################################
## @brief USM pool limits descriptor type
## 
## @details
##     - Specify these properties in ::urUSMPoolCreate via ::ur_usm_pool_desc_t
##       as part of a `pNext` chain.
class ur_usm_pool_limits_desc_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_USM_POOL_LIMITS_DESC
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("maxPoolableSize", c_size_t),                                  ## [in] Allocations up to this limit will be subject to pooling
        ("minDriverAllocSize", c_size_t)                                ## [in] Minimum allocation size that will be requested from the driver
    ]

###############################################################################
## @brief Get USM memory pool information
class ur_usm_pool_info_v(IntEnum):
    REFERENCE_COUNT = 0                             ## [uint32_t] Reference count of the pool object.
                                                    ## The reference count returned should be considered immediately stale. 
                                                    ## It is unsuitable for general use in applications. This feature is
                                                    ## provided for identifying memory leaks.
    CONTEXT = 1                                     ## [::ur_context_handle_t] USM memory pool context info

class ur_usm_pool_info_t(c_int):
    def __str__(self):
        return str(ur_usm_pool_info_v(self.value))


###############################################################################
## @brief Virtual memory granularity info
class ur_virtual_mem_granularity_info_v(IntEnum):
    MINIMUM = 0x30100                               ## [size_t] size in bytes of the minimum virtual memory granularity.
    RECOMMENDED = 0x30101                           ## [size_t] size in bytes of the recommended virtual memory granularity.

class ur_virtual_mem_granularity_info_t(c_int):
    def __str__(self):
        return str(ur_virtual_mem_granularity_info_v(self.value))


###############################################################################
## @brief Virtual memory access mode flags.
class ur_virtual_mem_access_flags_v(IntEnum):
    NONE = UR_BIT(0)                                ## Virtual memory has no access.
    READ_WRITE = UR_BIT(1)                          ## Virtual memory has both read and write access.
    READ_ONLY = UR_BIT(2)                           ## Virtual memory has read only access.

class ur_virtual_mem_access_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Virtual memory range info queries.
class ur_virtual_mem_info_v(IntEnum):
    ACCESS_MODE = 0                                 ## [::ur_virtual_mem_access_flags_t] access flags of a mapped virtual
                                                    ## memory range.

class ur_virtual_mem_info_t(c_int):
    def __str__(self):
        return str(ur_virtual_mem_info_v(self.value))


###############################################################################
## @brief Physical memory creation properties.
class ur_physical_mem_flags_v(IntEnum):
    TBD = UR_BIT(0)                                 ## reserved for future use.

class ur_physical_mem_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Physical memory creation properties.
class ur_physical_mem_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("flags", ur_physical_mem_flags_t)                              ## [in] physical memory creation flags
    ]

###############################################################################
## @brief Program metadata property type.
class ur_program_metadata_type_v(IntEnum):
    UINT32 = 0                                      ## type is a 32-bit integer.
    UINT64 = 1                                      ## type is a 64-bit integer.
    BYTE_ARRAY = 2                                  ## type is a byte array.
    STRING = 3                                      ## type is a null-terminated string.

class ur_program_metadata_type_t(c_int):
    def __str__(self):
        return str(ur_program_metadata_type_v(self.value))


###############################################################################
## @brief Program metadata value union.
class ur_program_metadata_value_t(Structure):
    _fields_ = [
        ("data32", c_ulong),                                            ## [in] inline storage for the 32-bit data, type
                                                                        ## ::UR_PROGRAM_METADATA_TYPE_UINT32.
        ("data64", c_ulonglong),                                        ## [in] inline storage for the 64-bit data, type
                                                                        ## ::UR_PROGRAM_METADATA_TYPE_UINT64.
        ("pString", c_char_p),                                          ## [in] pointer to null-terminated string data, type
                                                                        ## ::UR_PROGRAM_METADATA_TYPE_STRING.
        ("pData", c_void_p)                                             ## [in] pointer to binary data, type
                                                                        ## ::UR_PROGRAM_METADATA_TYPE_BYTE_ARRAY.
    ]

###############################################################################
## @brief Program metadata property.
class ur_program_metadata_t(Structure):
    _fields_ = [
        ("pName", c_char_p),                                            ## [in] null-terminated metadata name.
        ("type", ur_program_metadata_type_t),                           ## [in] the type of metadata value.
        ("size", c_size_t),                                             ## [in] size in bytes of the data pointed to by value.pData, or 0 when
                                                                        ## value size is less than 64-bits and is stored directly in value.data.
        ("value", ur_program_metadata_value_t)                          ## [in][tagged_by(type)] the metadata value storage.
    ]

###############################################################################
## @brief Program creation properties.
class ur_program_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("count", c_ulong),                                             ## [in] the number of entries in pMetadatas, if count is greater than
                                                                        ## zero then pMetadatas must not be null.
        ("pMetadatas", POINTER(ur_program_metadata_t))                  ## [in][optional][range(0,count)] pointer to array of metadata entries.
    ]

###############################################################################
## @brief Get Program object information
class ur_program_info_v(IntEnum):
    REFERENCE_COUNT = 0                             ## [uint32_t] Reference count of the program object.
                                                    ## The reference count returned should be considered immediately stale. 
                                                    ## It is unsuitable for general use in applications. This feature is
                                                    ## provided for identifying memory leaks.
    CONTEXT = 1                                     ## [::ur_context_handle_t] Program context info.
    NUM_DEVICES = 2                                 ## [uint32_t] Return number of devices associated with Program.
    DEVICES = 3                                     ## [::ur_device_handle_t[]] Return list of devices associated with
                                                    ## Program.
    SOURCE = 4                                      ## [char[]] Return program source associated with Program.
    BINARY_SIZES = 5                                ## [size_t[]] Return program binary sizes for each device.
    BINARIES = 6                                    ## [unsigned char[]] Return program binaries for all devices for this
                                                    ## Program.
    NUM_KERNELS = 7                                 ## [size_t] Number of kernels in Program, return type size_t.
    KERNEL_NAMES = 8                                ## [char[]] Return a null-terminated, semi-colon separated list of kernel
                                                    ## names in Program.

class ur_program_info_t(c_int):
    def __str__(self):
        return str(ur_program_info_v(self.value))


###############################################################################
## @brief Program object build status
class ur_program_build_status_v(IntEnum):
    NONE = 0                                        ## Program build status none
    ERROR = 1                                       ## Program build error
    SUCCESS = 2                                     ## Program build success
    IN_PROGRESS = 3                                 ## Program build in progress

class ur_program_build_status_t(c_int):
    def __str__(self):
        return str(ur_program_build_status_v(self.value))


###############################################################################
## @brief Program object binary type
class ur_program_binary_type_v(IntEnum):
    NONE = 0                                        ## No program binary is associated with device
    COMPILED_OBJECT = 1                             ## Program binary is compiled object
    LIBRARY = 2                                     ## Program binary is library object
    EXECUTABLE = 3                                  ## Program binary is executable

class ur_program_binary_type_t(c_int):
    def __str__(self):
        return str(ur_program_binary_type_v(self.value))


###############################################################################
## @brief Get Program object build information
class ur_program_build_info_v(IntEnum):
    STATUS = 0                                      ## [::ur_program_build_status_t] Program build status.
    OPTIONS = 1                                     ## [char[]] Null-terminated options string specified by last build,
                                                    ## compile or link operation performed on the program.
    LOG = 2                                         ## [char[]] Null-terminated program build log.
    BINARY_TYPE = 3                                 ## [::ur_program_binary_type_t] Program binary type.

class ur_program_build_info_t(c_int):
    def __str__(self):
        return str(ur_program_build_info_v(self.value))


###############################################################################
## @brief Specialization constant information
class ur_specialization_constant_info_t(Structure):
    _fields_ = [
        ("id", c_ulong),                                                ## [in] specialization constant Id
        ("size", c_size_t),                                             ## [in] size of the specialization constant value
        ("pValue", c_void_p)                                            ## [in] pointer to the specialization constant value bytes
    ]

###############################################################################
## @brief Native program creation properties
class ur_program_native_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_PROGRAM_NATIVE_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("isNativeHandleOwned", c_bool)                                 ## [in] Indicates UR owns the native handle or if it came from an
                                                                        ## interoperability operation in the application that asked to not
                                                                        ## transfer the ownership to the unified-runtime.
    ]

###############################################################################
## @brief Properties for for ::urKernelSetArgValue.
class ur_kernel_arg_value_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_KERNEL_ARG_VALUE_PROPERTIES
        ("pNext", c_void_p)                                             ## [in,out][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief Properties for for ::urKernelSetArgLocal.
class ur_kernel_arg_local_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_KERNEL_ARG_LOCAL_PROPERTIES
        ("pNext", c_void_p)                                             ## [in,out][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief Get Kernel object information
class ur_kernel_info_v(IntEnum):
    FUNCTION_NAME = 0                               ## [char[]] Return null-terminated kernel function name.
    NUM_ARGS = 1                                    ## [size_t] Return Kernel number of arguments.
    REFERENCE_COUNT = 2                             ## [uint32_t] Reference count of the kernel object.
                                                    ## The reference count returned should be considered immediately stale. 
                                                    ## It is unsuitable for general use in applications. This feature is
                                                    ## provided for identifying memory leaks.
    CONTEXT = 3                                     ## [::ur_context_handle_t] Return Context object associated with Kernel.
    PROGRAM = 4                                     ## [::ur_program_handle_t] Return Program object associated with Kernel.
    ATTRIBUTES = 5                                  ## [char[]] Return null-terminated kernel attributes string.
    NUM_REGS = 6                                    ## [uint32_t] Return the number of registers used by the compiled kernel
                                                    ## (device specific).

class ur_kernel_info_t(c_int):
    def __str__(self):
        return str(ur_kernel_info_v(self.value))


###############################################################################
## @brief Get Kernel Work Group information
class ur_kernel_group_info_v(IntEnum):
    GLOBAL_WORK_SIZE = 0                            ## [size_t[3]] Return Work Group maximum global size
    WORK_GROUP_SIZE = 1                             ## [size_t] Return maximum Work Group size
    COMPILE_WORK_GROUP_SIZE = 2                     ## [size_t[3]] Return Work Group size required by the source code, such
                                                    ## as __attribute__((required_work_group_size(X,Y,Z))
    LOCAL_MEM_SIZE = 3                              ## [size_t] Return local memory required by the Kernel
    PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 4          ## [size_t] Return preferred multiple of Work Group size for launch
    PRIVATE_MEM_SIZE = 5                            ## [size_t] Return minimum amount of private memory in bytes used by each
                                                    ## work item in the Kernel

class ur_kernel_group_info_t(c_int):
    def __str__(self):
        return str(ur_kernel_group_info_v(self.value))


###############################################################################
## @brief Get Kernel SubGroup information
class ur_kernel_sub_group_info_v(IntEnum):
    MAX_SUB_GROUP_SIZE = 0                          ## [uint32_t] Return maximum SubGroup size
    MAX_NUM_SUB_GROUPS = 1                          ## [uint32_t] Return maximum number of SubGroup
    COMPILE_NUM_SUB_GROUPS = 2                      ## [uint32_t] Return number of SubGroup required by the source code
    SUB_GROUP_SIZE_INTEL = 3                        ## [uint32_t] Return SubGroup size required by Intel

class ur_kernel_sub_group_info_t(c_int):
    def __str__(self):
        return str(ur_kernel_sub_group_info_v(self.value))


###############################################################################
## @brief Kernel Cache Configuration.
class ur_kernel_cache_config_v(IntEnum):
    DEFAULT = 0                                     ## No preference for SLM or data cache.
    LARGE_SLM = 1                                   ## Large Shared Local Memory (SLM) size.
    LARGE_DATA = 2                                  ## Large General Data size.

class ur_kernel_cache_config_t(c_int):
    def __str__(self):
        return str(ur_kernel_cache_config_v(self.value))


###############################################################################
## @brief Set additional Kernel execution information
class ur_kernel_exec_info_v(IntEnum):
    USM_INDIRECT_ACCESS = 0                         ## [::ur_bool_t] Kernel might access data through USM pointer.
    USM_PTRS = 1                                    ## [void *[]] Provide an explicit array of USM pointers that the kernel
                                                    ## will access.
    CACHE_CONFIG = 2                                ## [::ur_kernel_cache_config_t] Provide the preferred cache configuration

class ur_kernel_exec_info_t(c_int):
    def __str__(self):
        return str(ur_kernel_exec_info_v(self.value))


###############################################################################
## @brief Properties for for ::urKernelSetArgPointer.
class ur_kernel_arg_pointer_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_KERNEL_ARG_POINTER_PROPERTIES
        ("pNext", c_void_p)                                             ## [in,out][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief Properties for for ::urKernelSetExecInfo.
class ur_kernel_exec_info_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_KERNEL_EXEC_INFO_PROPERTIES
        ("pNext", c_void_p)                                             ## [in,out][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief Properties for for ::urKernelSetArgSampler.
class ur_kernel_arg_sampler_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_KERNEL_ARG_SAMPLER_PROPERTIES
        ("pNext", c_void_p)                                             ## [in,out][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief Properties for for ::urKernelSetArgMemObj.
class ur_kernel_arg_mem_obj_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_KERNEL_ARG_MEM_OBJ_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("memoryAccess", ur_mem_flags_t)                                ## [in] Memory access flag. Allowed values are: ::UR_MEM_FLAG_READ_WRITE,
                                                                        ## ::UR_MEM_FLAG_WRITE_ONLY, ::UR_MEM_FLAG_READ_ONLY.
    ]

###############################################################################
## @brief Properties for for ::urKernelCreateWithNativeHandle.
class ur_kernel_native_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_KERNEL_NATIVE_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("isNativeHandleOwned", c_bool)                                 ## [in] Indicates UR owns the native handle or if it came from an interoperability
                                                                        ## operation in the application that asked to not transfer the ownership to
                                                                        ## the unified-runtime.
    ]

###############################################################################
## @brief Query queue info
class ur_queue_info_v(IntEnum):
    CONTEXT = 0                                     ## [::ur_queue_handle_t] context associated with this queue.
    DEVICE = 1                                      ## [::ur_device_handle_t] device associated with this queue.
    DEVICE_DEFAULT = 2                              ## [::ur_queue_handle_t] the current default queue of the underlying
                                                    ## device.
    FLAGS = 3                                       ## [::ur_queue_flags_t] the properties associated with
                                                    ## ::ur_queue_properties_t::flags.
    REFERENCE_COUNT = 4                             ## [uint32_t] Reference count of the queue object.
                                                    ## The reference count returned should be considered immediately stale. 
                                                    ## It is unsuitable for general use in applications. This feature is
                                                    ## provided for identifying memory leaks.
    SIZE = 5                                        ## [uint32_t] The size of the queue
    EMPTY = 6                                       ## [::ur_bool_t] return true if the queue was empty at the time of the
                                                    ## query

class ur_queue_info_t(c_int):
    def __str__(self):
        return str(ur_queue_info_v(self.value))


###############################################################################
## @brief Queue property flags
class ur_queue_flags_v(IntEnum):
    OUT_OF_ORDER_EXEC_MODE_ENABLE = UR_BIT(0)       ## Enable/disable out of order execution
    PROFILING_ENABLE = UR_BIT(1)                    ## Enable/disable profiling
    ON_DEVICE = UR_BIT(2)                           ## Is a device queue
    ON_DEVICE_DEFAULT = UR_BIT(3)                   ## Is the default queue for a device
    DISCARD_EVENTS = UR_BIT(4)                      ## Events will be discarded
    PRIORITY_LOW = UR_BIT(5)                        ## Low priority queue
    PRIORITY_HIGH = UR_BIT(6)                       ## High priority queue
    SUBMISSION_BATCHED = UR_BIT(7)                  ## Hint: enqueue and submit in a batch later. No change in queue
                                                    ## semantics. Implementation chooses submission mode.
    SUBMISSION_IMMEDIATE = UR_BIT(8)                ## Hint: enqueue and submit immediately. No change in queue semantics.
                                                    ## Implementation chooses submission mode.
    USE_DEFAULT_STREAM = UR_BIT(9)                  ## Use the default stream. Only meaningful for CUDA. Other platforms may
                                                    ## ignore this flag.
    SYNC_WITH_DEFAULT_STREAM = UR_BIT(10)           ## Synchronize with the default stream. Only meaningful for CUDA. Other
                                                    ## platforms may ignore this flag.

class ur_queue_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Queue creation properties
class ur_queue_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_QUEUE_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("flags", ur_queue_flags_t)                                     ## [in] Bitfield of queue creation flags
    ]

###############################################################################
## @brief Queue index creation properties
## 
## @details
##     - Specify these properties in ::urQueueCreate via
##       ::ur_queue_properties_t as part of a `pNext` chain.
class ur_queue_index_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_QUEUE_INDEX_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("computeIndex", c_ulong)                                       ## [in] Specifies the compute index as described in the
                                                                        ## sycl_ext_intel_queue_index extension.
    ]

###############################################################################
## @brief Descriptor for ::urQueueGetNativeHandle and
##        ::urQueueCreateWithNativeHandle.
## 
## @details
##     - Specify this descriptor in ::urQueueGetNativeHandle directly or
##       ::urQueueCreateWithNativeHandle via ::ur_queue_native_properties_t as
##       part of a `pNext` chain.
class ur_queue_native_desc_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_QUEUE_NATIVE_DESC
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("pNativeData", c_void_p)                                       ## [in][optional] Adapter-specific metadata needed to create the handle.
    ]

###############################################################################
## @brief Properties for for ::urQueueCreateWithNativeHandle.
class ur_queue_native_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_QUEUE_NATIVE_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("isNativeHandleOwned", c_bool)                                 ## [in] Indicates UR owns the native handle or if it came from an interoperability
                                                                        ## operation in the application that asked to not transfer the ownership to
                                                                        ## the unified-runtime.
    ]

###############################################################################
## @brief Command type
class ur_command_v(IntEnum):
    KERNEL_LAUNCH = 0                               ## Event created by ::urEnqueueKernelLaunch
    EVENTS_WAIT = 1                                 ## Event created by ::urEnqueueEventsWait
    EVENTS_WAIT_WITH_BARRIER = 2                    ## Event created by ::urEnqueueEventsWaitWithBarrier
    MEM_BUFFER_READ = 3                             ## Event created by ::urEnqueueMemBufferRead
    MEM_BUFFER_WRITE = 4                            ## Event created by ::urEnqueueMemBufferWrite
    MEM_BUFFER_READ_RECT = 5                        ## Event created by ::urEnqueueMemBufferReadRect
    MEM_BUFFER_WRITE_RECT = 6                       ## Event created by ::urEnqueueMemBufferWriteRect
    MEM_BUFFER_COPY = 7                             ## Event created by ::urEnqueueMemBufferCopy
    MEM_BUFFER_COPY_RECT = 8                        ## Event created by ::urEnqueueMemBufferCopyRect
    MEM_BUFFER_FILL = 9                             ## Event created by ::urEnqueueMemBufferFill
    MEM_IMAGE_READ = 10                             ## Event created by ::urEnqueueMemImageRead
    MEM_IMAGE_WRITE = 11                            ## Event created by ::urEnqueueMemImageWrite
    MEM_IMAGE_COPY = 12                             ## Event created by ::urEnqueueMemImageCopy
    MEM_BUFFER_MAP = 14                             ## Event created by ::urEnqueueMemBufferMap
    MEM_UNMAP = 16                                  ## Event created by ::urEnqueueMemUnmap
    USM_FILL = 17                                   ## Event created by ::urEnqueueUSMFill
    USM_MEMCPY = 18                                 ## Event created by ::urEnqueueUSMMemcpy
    USM_PREFETCH = 19                               ## Event created by ::urEnqueueUSMPrefetch
    USM_ADVISE = 20                                 ## Event created by ::urEnqueueUSMAdvise
    USM_FILL_2D = 21                                ## Event created by ::urEnqueueUSMFill2D
    USM_MEMCPY_2D = 22                              ## Event created by ::urEnqueueUSMMemcpy2D
    DEVICE_GLOBAL_VARIABLE_WRITE = 23               ## Event created by ::urEnqueueDeviceGlobalVariableWrite
    DEVICE_GLOBAL_VARIABLE_READ = 24                ## Event created by ::urEnqueueDeviceGlobalVariableRead
    READ_HOST_PIPE = 25                             ## Event created by ::urEnqueueReadHostPipe
    WRITE_HOST_PIPE = 26                            ## Event created by ::urEnqueueWriteHostPipe
    COMMAND_BUFFER_ENQUEUE_EXP = 0x1000             ## Event created by ::urCommandBufferEnqueueExp
    INTEROP_SEMAPHORE_WAIT_EXP = 0x2000             ## Event created by ::urBindlessImagesWaitExternalSemaphoreExp
    INTEROP_SEMAPHORE_SIGNAL_EXP = 0x2001           ## Event created by ::urBindlessImagesSignalExternalSemaphoreExp

class ur_command_t(c_int):
    def __str__(self):
        return str(ur_command_v(self.value))


###############################################################################
## @brief Event Status
class ur_event_status_v(IntEnum):
    COMPLETE = 0                                    ## Command is complete
    RUNNING = 1                                     ## Command is running
    SUBMITTED = 2                                   ## Command is submitted
    QUEUED = 3                                      ## Command is queued

class ur_event_status_t(c_int):
    def __str__(self):
        return str(ur_event_status_v(self.value))


###############################################################################
## @brief Event query information type
class ur_event_info_v(IntEnum):
    COMMAND_QUEUE = 0                               ## [::ur_queue_handle_t] Command queue information of an event object
    CONTEXT = 1                                     ## [::ur_context_handle_t] Context information of an event object
    COMMAND_TYPE = 2                                ## [::ur_command_t] Command type information of an event object
    COMMAND_EXECUTION_STATUS = 3                    ## [::ur_event_status_t] Command execution status of an event object
    REFERENCE_COUNT = 4                             ## [uint32_t] Reference count of the event object.
                                                    ## The reference count returned should be considered immediately stale. 
                                                    ## It is unsuitable for general use in applications. This feature is
                                                    ## provided for identifying memory leaks.

class ur_event_info_t(c_int):
    def __str__(self):
        return str(ur_event_info_v(self.value))


###############################################################################
## @brief Profiling query information type
class ur_profiling_info_v(IntEnum):
    COMMAND_QUEUED = 0                              ## [uint64_t] A 64-bit value of current device counter in nanoseconds
                                                    ## when the event is enqueued
    COMMAND_SUBMIT = 1                              ## [uint64_t] A 64-bit value of current device counter in nanoseconds
                                                    ## when the event is submitted
    COMMAND_START = 2                               ## [uint64_t] A 64-bit value of current device counter in nanoseconds
                                                    ## when the event starts execution
    COMMAND_END = 3                                 ## [uint64_t] A 64-bit value of current device counter in nanoseconds
                                                    ## when the event has finished execution
    COMMAND_COMPLETE = 4                            ## [uint64_t] A 64-bit value of current device counter in nanoseconds
                                                    ## when the event and any child events enqueued by this event on the
                                                    ## device have finished execution

class ur_profiling_info_t(c_int):
    def __str__(self):
        return str(ur_profiling_info_v(self.value))


###############################################################################
## @brief Properties for for ::urEventCreateWithNativeHandle.
class ur_event_native_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_EVENT_NATIVE_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("isNativeHandleOwned", c_bool)                                 ## [in] Indicates UR owns the native handle or if it came from an interoperability
                                                                        ## operation in the application that asked to not transfer the ownership to
                                                                        ## the unified-runtime.
    ]

###############################################################################
## @brief Event states for all events.
class ur_execution_info_v(IntEnum):
    COMPLETE = 0                                    ## Indicates that the event has completed.
    RUNNING = 1                                     ## Indicates that the device has started processing this event.
    SUBMITTED = 2                                   ## Indicates that the event has been submitted by the host to the device.
    QUEUED = 3                                      ## Indicates that the event has been queued, this is the initial state of
                                                    ## events.

class ur_execution_info_t(c_int):
    def __str__(self):
        return str(ur_execution_info_v(self.value))


###############################################################################
## @brief Event callback function that can be registered by the application.
def ur_event_callback_t(user_defined_callback):
    @CFUNCTYPE(None, ur_event_handle_t, ur_execution_info_t, c_void_p)
    def ur_event_callback_t_wrapper(hEvent, execStatus, pUserData):
        return user_defined_callback(hEvent, execStatus, pUserData)
    return ur_event_callback_t_wrapper

###############################################################################
## @brief Map flags
class ur_map_flags_v(IntEnum):
    READ = UR_BIT(0)                                ## Map for read access
    WRITE = UR_BIT(1)                               ## Map for write access
    WRITE_INVALIDATE_REGION = UR_BIT(2)             ## Map for discard_write access

class ur_map_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Map flags
class ur_usm_migration_flags_v(IntEnum):
    DEFAULT = UR_BIT(0)                             ## Default migration TODO: Add more enums! 

class ur_usm_migration_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Handle of bindless image
class ur_exp_image_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of bindless image memory
class ur_exp_image_mem_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of interop memory
class ur_exp_interop_mem_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of interop semaphore
class ur_exp_interop_semaphore_handle_t(c_void_p):
    pass

###############################################################################
## @brief Dictates the type of memory copy.
class ur_exp_image_copy_flags_v(IntEnum):
    HOST_TO_DEVICE = UR_BIT(0)                      ## Host to device
    DEVICE_TO_HOST = UR_BIT(1)                      ## Device to host
    DEVICE_TO_DEVICE = UR_BIT(2)                    ## Device to device

class ur_exp_image_copy_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief File descriptor
class ur_exp_file_descriptor_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("fd", c_int)                                                   ## [in] A file descriptor used for Linux and & MacOS operating systems.
    ]

###############################################################################
## @brief Windows specific file handle
class ur_exp_win32_handle_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("handle", c_void_p)                                            ## [in] A win32 file handle.
    ]

###############################################################################
## @brief Describes mipmap sampler properties
## 
## @details
##     - Specify these properties in ::urSamplerCreate via ::ur_sampler_desc_t
##       as part of a `pNext` chain.
class ur_exp_sampler_mip_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_EXP_SAMPLER_MIP_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("minMipmapLevelClamp", c_float),                               ## [in] minimum mipmap level from which we can sample, minimum value
                                                                        ## being 0
        ("maxMipmapLevelClamp", c_float),                               ## [in] maximum mipmap level from which we can sample, maximum value
                                                                        ## being the number of levels
        ("maxAnisotropy", c_float),                                     ## [in] anisotropic ratio used when samplling the mipmap with anisotropic
                                                                        ## filtering
        ("mipFilterMode", ur_sampler_filter_mode_t)                     ## [in] mipmap filter mode used for filtering between mipmap levels
    ]

###############################################################################
## @brief Describes unique sampler addressing mode per dimension
## 
## @details
##     - Specify these properties in ::urSamplerCreate via ::ur_sampler_desc_t
##       as part of a `pNext` chain.
class ur_exp_sampler_addr_modes_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_EXP_SAMPLER_ADDR_MODES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("addrModes", ur_sampler_addressing_mode_t * 3)                 ## [in] Specify the address mode of the sampler per dimension
    ]

###############################################################################
## @brief Describes an interop memory resource descriptor
class ur_exp_interop_mem_desc_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_EXP_INTEROP_MEM_DESC
        ("pNext", c_void_p)                                             ## [in][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief Describes an interop semaphore resource descriptor
class ur_exp_interop_semaphore_desc_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_EXP_INTEROP_SEMAPHORE_DESC
        ("pNext", c_void_p)                                             ## [in][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief Describes layered image properties
## 
## @details
##     - Specify these properties in ::urBindlessImagesUnsampledImageCreateExp
##       or ::urBindlessImagesSampledImageCreateExp via ::ur_image_desc_t as
##       part of a `pNext` chain.
class ur_exp_layered_image_properties_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_EXP_LAYERED_IMAGE_PROPERTIES
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("numLayers", c_ulong)                                          ## [in] number of layers the image should have
    ]

###############################################################################
## @brief The extension string which defines support for command-buffers which
##        is returned when querying device extensions.
UR_COMMAND_BUFFER_EXTENSION_STRING_EXP = "ur_exp_command_buffer"

###############################################################################
## @brief Command-Buffer Descriptor Type
class ur_exp_command_buffer_desc_t(Structure):
    _fields_ = [
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure, must be
                                                                        ## ::UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC
        ("pNext", c_void_p)                                             ## [in][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief A value that identifies a command inside of a command-buffer, used for
##        defining dependencies between commands in the same command-buffer.
class ur_exp_command_buffer_sync_point_t(c_ulong):
    pass

###############################################################################
## @brief Handle of Command-Buffer object
class ur_exp_command_buffer_handle_t(c_void_p):
    pass

###############################################################################
## @brief The extension string which defines support for cooperative-kernels
##        which is returned when querying device extensions.
UR_COOPERATIVE_KERNELS_EXTENSION_STRING_EXP = "ur_exp_cooperative_kernels"

###############################################################################
## @brief The extension string which defines support for test
##        which is returned when querying device extensions.
UR_MULTI_DEVICE_COMPILE_EXTENSION_STRING_EXP = "ur_exp_multi_device_compile"

###############################################################################
## @brief Supported peer info
class ur_exp_peer_info_v(IntEnum):
    UR_PEER_ACCESS_SUPPORTED = 0                    ## [uint32_t] 1 if P2P access is supported otherwise P2P access is not
                                                    ## supported.
    UR_PEER_ATOMICS_SUPPORTED = 1                   ## [uint32_t] 1 if atomic operations are supported over the P2P link,
                                                    ## otherwise such operations are not supported.

class ur_exp_peer_info_t(c_int):
    def __str__(self):
        return str(ur_exp_peer_info_v(self.value))


###############################################################################
__use_win_types = "Windows" == platform.uname()[0]

###############################################################################
## @brief Function-pointer for urPlatformGet
if __use_win_types:
    _urPlatformGet_t = WINFUNCTYPE( ur_result_t, POINTER(ur_adapter_handle_t), c_ulong, c_ulong, POINTER(ur_platform_handle_t), POINTER(c_ulong) )
else:
    _urPlatformGet_t = CFUNCTYPE( ur_result_t, POINTER(ur_adapter_handle_t), c_ulong, c_ulong, POINTER(ur_platform_handle_t), POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for urPlatformGetInfo
if __use_win_types:
    _urPlatformGetInfo_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_platform_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urPlatformGetInfo_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_platform_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urPlatformGetNativeHandle
if __use_win_types:
    _urPlatformGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, POINTER(ur_native_handle_t) )
else:
    _urPlatformGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urPlatformCreateWithNativeHandle
if __use_win_types:
    _urPlatformCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_native_handle_t, POINTER(ur_platform_native_properties_t), POINTER(ur_platform_handle_t) )
else:
    _urPlatformCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_native_handle_t, POINTER(ur_platform_native_properties_t), POINTER(ur_platform_handle_t) )

###############################################################################
## @brief Function-pointer for urPlatformGetApiVersion
if __use_win_types:
    _urPlatformGetApiVersion_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, POINTER(ur_api_version_t) )
else:
    _urPlatformGetApiVersion_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, POINTER(ur_api_version_t) )

###############################################################################
## @brief Function-pointer for urPlatformGetBackendOption
if __use_win_types:
    _urPlatformGetBackendOption_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, c_char_p, POINTER(c_char_p) )
else:
    _urPlatformGetBackendOption_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, c_char_p, POINTER(c_char_p) )


###############################################################################
## @brief Table of Platform functions pointers
class ur_platform_dditable_t(Structure):
    _fields_ = [
        ("pfnGet", c_void_p),                                           ## _urPlatformGet_t
        ("pfnGetInfo", c_void_p),                                       ## _urPlatformGetInfo_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urPlatformGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p),                        ## _urPlatformCreateWithNativeHandle_t
        ("pfnGetApiVersion", c_void_p),                                 ## _urPlatformGetApiVersion_t
        ("pfnGetBackendOption", c_void_p)                               ## _urPlatformGetBackendOption_t
    ]

###############################################################################
## @brief Function-pointer for urContextCreate
if __use_win_types:
    _urContextCreate_t = WINFUNCTYPE( ur_result_t, c_ulong, POINTER(ur_device_handle_t), POINTER(ur_context_properties_t), POINTER(ur_context_handle_t) )
else:
    _urContextCreate_t = CFUNCTYPE( ur_result_t, c_ulong, POINTER(ur_device_handle_t), POINTER(ur_context_properties_t), POINTER(ur_context_handle_t) )

###############################################################################
## @brief Function-pointer for urContextRetain
if __use_win_types:
    _urContextRetain_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t )
else:
    _urContextRetain_t = CFUNCTYPE( ur_result_t, ur_context_handle_t )

###############################################################################
## @brief Function-pointer for urContextRelease
if __use_win_types:
    _urContextRelease_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t )
else:
    _urContextRelease_t = CFUNCTYPE( ur_result_t, ur_context_handle_t )

###############################################################################
## @brief Function-pointer for urContextGetInfo
if __use_win_types:
    _urContextGetInfo_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_context_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urContextGetInfo_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_context_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urContextGetNativeHandle
if __use_win_types:
    _urContextGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_native_handle_t) )
else:
    _urContextGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urContextCreateWithNativeHandle
if __use_win_types:
    _urContextCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_native_handle_t, c_ulong, POINTER(ur_device_handle_t), POINTER(ur_context_native_properties_t), POINTER(ur_context_handle_t) )
else:
    _urContextCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_native_handle_t, c_ulong, POINTER(ur_device_handle_t), POINTER(ur_context_native_properties_t), POINTER(ur_context_handle_t) )

###############################################################################
## @brief Function-pointer for urContextSetExtendedDeleter
if __use_win_types:
    _urContextSetExtendedDeleter_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_void_p )
else:
    _urContextSetExtendedDeleter_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_void_p )


###############################################################################
## @brief Table of Context functions pointers
class ur_context_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _urContextCreate_t
        ("pfnRetain", c_void_p),                                        ## _urContextRetain_t
        ("pfnRelease", c_void_p),                                       ## _urContextRelease_t
        ("pfnGetInfo", c_void_p),                                       ## _urContextGetInfo_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urContextGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p),                        ## _urContextCreateWithNativeHandle_t
        ("pfnSetExtendedDeleter", c_void_p)                             ## _urContextSetExtendedDeleter_t
    ]

###############################################################################
## @brief Function-pointer for urEventGetInfo
if __use_win_types:
    _urEventGetInfo_t = WINFUNCTYPE( ur_result_t, ur_event_handle_t, ur_event_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urEventGetInfo_t = CFUNCTYPE( ur_result_t, ur_event_handle_t, ur_event_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urEventGetProfilingInfo
if __use_win_types:
    _urEventGetProfilingInfo_t = WINFUNCTYPE( ur_result_t, ur_event_handle_t, ur_profiling_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urEventGetProfilingInfo_t = CFUNCTYPE( ur_result_t, ur_event_handle_t, ur_profiling_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urEventWait
if __use_win_types:
    _urEventWait_t = WINFUNCTYPE( ur_result_t, c_ulong, POINTER(ur_event_handle_t) )
else:
    _urEventWait_t = CFUNCTYPE( ur_result_t, c_ulong, POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEventRetain
if __use_win_types:
    _urEventRetain_t = WINFUNCTYPE( ur_result_t, ur_event_handle_t )
else:
    _urEventRetain_t = CFUNCTYPE( ur_result_t, ur_event_handle_t )

###############################################################################
## @brief Function-pointer for urEventRelease
if __use_win_types:
    _urEventRelease_t = WINFUNCTYPE( ur_result_t, ur_event_handle_t )
else:
    _urEventRelease_t = CFUNCTYPE( ur_result_t, ur_event_handle_t )

###############################################################################
## @brief Function-pointer for urEventGetNativeHandle
if __use_win_types:
    _urEventGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_event_handle_t, POINTER(ur_native_handle_t) )
else:
    _urEventGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_event_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urEventCreateWithNativeHandle
if __use_win_types:
    _urEventCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, POINTER(ur_event_native_properties_t), POINTER(ur_event_handle_t) )
else:
    _urEventCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, POINTER(ur_event_native_properties_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEventSetCallback
if __use_win_types:
    _urEventSetCallback_t = WINFUNCTYPE( ur_result_t, ur_event_handle_t, ur_execution_info_t, c_void_p, c_void_p )
else:
    _urEventSetCallback_t = CFUNCTYPE( ur_result_t, ur_event_handle_t, ur_execution_info_t, c_void_p, c_void_p )


###############################################################################
## @brief Table of Event functions pointers
class ur_event_dditable_t(Structure):
    _fields_ = [
        ("pfnGetInfo", c_void_p),                                       ## _urEventGetInfo_t
        ("pfnGetProfilingInfo", c_void_p),                              ## _urEventGetProfilingInfo_t
        ("pfnWait", c_void_p),                                          ## _urEventWait_t
        ("pfnRetain", c_void_p),                                        ## _urEventRetain_t
        ("pfnRelease", c_void_p),                                       ## _urEventRelease_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urEventGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p),                        ## _urEventCreateWithNativeHandle_t
        ("pfnSetCallback", c_void_p)                                    ## _urEventSetCallback_t
    ]

###############################################################################
## @brief Function-pointer for urProgramCreateWithIL
if __use_win_types:
    _urProgramCreateWithIL_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t, POINTER(ur_program_properties_t), POINTER(ur_program_handle_t) )
else:
    _urProgramCreateWithIL_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t, POINTER(ur_program_properties_t), POINTER(ur_program_handle_t) )

###############################################################################
## @brief Function-pointer for urProgramCreateWithBinary
if __use_win_types:
    _urProgramCreateWithBinary_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, c_size_t, POINTER(c_ubyte), POINTER(ur_program_properties_t), POINTER(ur_program_handle_t) )
else:
    _urProgramCreateWithBinary_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, c_size_t, POINTER(c_ubyte), POINTER(ur_program_properties_t), POINTER(ur_program_handle_t) )

###############################################################################
## @brief Function-pointer for urProgramBuild
if __use_win_types:
    _urProgramBuild_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_program_handle_t, c_char_p )
else:
    _urProgramBuild_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_program_handle_t, c_char_p )

###############################################################################
## @brief Function-pointer for urProgramCompile
if __use_win_types:
    _urProgramCompile_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_program_handle_t, c_char_p )
else:
    _urProgramCompile_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_program_handle_t, c_char_p )

###############################################################################
## @brief Function-pointer for urProgramLink
if __use_win_types:
    _urProgramLink_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_ulong, POINTER(ur_program_handle_t), c_char_p, POINTER(ur_program_handle_t) )
else:
    _urProgramLink_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_ulong, POINTER(ur_program_handle_t), c_char_p, POINTER(ur_program_handle_t) )

###############################################################################
## @brief Function-pointer for urProgramRetain
if __use_win_types:
    _urProgramRetain_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t )
else:
    _urProgramRetain_t = CFUNCTYPE( ur_result_t, ur_program_handle_t )

###############################################################################
## @brief Function-pointer for urProgramRelease
if __use_win_types:
    _urProgramRelease_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t )
else:
    _urProgramRelease_t = CFUNCTYPE( ur_result_t, ur_program_handle_t )

###############################################################################
## @brief Function-pointer for urProgramGetFunctionPointer
if __use_win_types:
    _urProgramGetFunctionPointer_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t, ur_program_handle_t, c_char_p, POINTER(c_void_p) )
else:
    _urProgramGetFunctionPointer_t = CFUNCTYPE( ur_result_t, ur_device_handle_t, ur_program_handle_t, c_char_p, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for urProgramGetInfo
if __use_win_types:
    _urProgramGetInfo_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t, ur_program_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urProgramGetInfo_t = CFUNCTYPE( ur_result_t, ur_program_handle_t, ur_program_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urProgramGetBuildInfo
if __use_win_types:
    _urProgramGetBuildInfo_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t, ur_device_handle_t, ur_program_build_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urProgramGetBuildInfo_t = CFUNCTYPE( ur_result_t, ur_program_handle_t, ur_device_handle_t, ur_program_build_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urProgramSetSpecializationConstants
if __use_win_types:
    _urProgramSetSpecializationConstants_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t, c_ulong, POINTER(ur_specialization_constant_info_t) )
else:
    _urProgramSetSpecializationConstants_t = CFUNCTYPE( ur_result_t, ur_program_handle_t, c_ulong, POINTER(ur_specialization_constant_info_t) )

###############################################################################
## @brief Function-pointer for urProgramGetNativeHandle
if __use_win_types:
    _urProgramGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t, POINTER(ur_native_handle_t) )
else:
    _urProgramGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_program_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urProgramCreateWithNativeHandle
if __use_win_types:
    _urProgramCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, POINTER(ur_program_native_properties_t), POINTER(ur_program_handle_t) )
else:
    _urProgramCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, POINTER(ur_program_native_properties_t), POINTER(ur_program_handle_t) )


###############################################################################
## @brief Table of Program functions pointers
class ur_program_dditable_t(Structure):
    _fields_ = [
        ("pfnCreateWithIL", c_void_p),                                  ## _urProgramCreateWithIL_t
        ("pfnCreateWithBinary", c_void_p),                              ## _urProgramCreateWithBinary_t
        ("pfnBuild", c_void_p),                                         ## _urProgramBuild_t
        ("pfnCompile", c_void_p),                                       ## _urProgramCompile_t
        ("pfnLink", c_void_p),                                          ## _urProgramLink_t
        ("pfnRetain", c_void_p),                                        ## _urProgramRetain_t
        ("pfnRelease", c_void_p),                                       ## _urProgramRelease_t
        ("pfnGetFunctionPointer", c_void_p),                            ## _urProgramGetFunctionPointer_t
        ("pfnGetInfo", c_void_p),                                       ## _urProgramGetInfo_t
        ("pfnGetBuildInfo", c_void_p),                                  ## _urProgramGetBuildInfo_t
        ("pfnSetSpecializationConstants", c_void_p),                    ## _urProgramSetSpecializationConstants_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urProgramGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p)                         ## _urProgramCreateWithNativeHandle_t
    ]

###############################################################################
## @brief Function-pointer for urProgramBuildExp
if __use_win_types:
    _urProgramBuildExp_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t, c_ulong, POINTER(ur_device_handle_t), c_char_p )
else:
    _urProgramBuildExp_t = CFUNCTYPE( ur_result_t, ur_program_handle_t, c_ulong, POINTER(ur_device_handle_t), c_char_p )

###############################################################################
## @brief Function-pointer for urProgramCompileExp
if __use_win_types:
    _urProgramCompileExp_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t, c_ulong, POINTER(ur_device_handle_t), c_char_p )
else:
    _urProgramCompileExp_t = CFUNCTYPE( ur_result_t, ur_program_handle_t, c_ulong, POINTER(ur_device_handle_t), c_char_p )

###############################################################################
## @brief Function-pointer for urProgramLinkExp
if __use_win_types:
    _urProgramLinkExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_ulong, POINTER(ur_device_handle_t), c_ulong, POINTER(ur_program_handle_t), c_char_p, POINTER(ur_program_handle_t) )
else:
    _urProgramLinkExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_ulong, POINTER(ur_device_handle_t), c_ulong, POINTER(ur_program_handle_t), c_char_p, POINTER(ur_program_handle_t) )


###############################################################################
## @brief Table of ProgramExp functions pointers
class ur_program_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnBuildExp", c_void_p),                                      ## _urProgramBuildExp_t
        ("pfnCompileExp", c_void_p),                                    ## _urProgramCompileExp_t
        ("pfnLinkExp", c_void_p)                                        ## _urProgramLinkExp_t
    ]

###############################################################################
## @brief Function-pointer for urKernelCreate
if __use_win_types:
    _urKernelCreate_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t, c_char_p, POINTER(ur_kernel_handle_t) )
else:
    _urKernelCreate_t = CFUNCTYPE( ur_result_t, ur_program_handle_t, c_char_p, POINTER(ur_kernel_handle_t) )

###############################################################################
## @brief Function-pointer for urKernelGetInfo
if __use_win_types:
    _urKernelGetInfo_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_kernel_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urKernelGetInfo_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_kernel_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urKernelGetGroupInfo
if __use_win_types:
    _urKernelGetGroupInfo_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_device_handle_t, ur_kernel_group_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urKernelGetGroupInfo_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_device_handle_t, ur_kernel_group_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urKernelGetSubGroupInfo
if __use_win_types:
    _urKernelGetSubGroupInfo_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_device_handle_t, ur_kernel_sub_group_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urKernelGetSubGroupInfo_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_device_handle_t, ur_kernel_sub_group_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urKernelRetain
if __use_win_types:
    _urKernelRetain_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t )
else:
    _urKernelRetain_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t )

###############################################################################
## @brief Function-pointer for urKernelRelease
if __use_win_types:
    _urKernelRelease_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t )
else:
    _urKernelRelease_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t )

###############################################################################
## @brief Function-pointer for urKernelGetNativeHandle
if __use_win_types:
    _urKernelGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, POINTER(ur_native_handle_t) )
else:
    _urKernelGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urKernelCreateWithNativeHandle
if __use_win_types:
    _urKernelCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, ur_program_handle_t, POINTER(ur_kernel_native_properties_t), POINTER(ur_kernel_handle_t) )
else:
    _urKernelCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, ur_program_handle_t, POINTER(ur_kernel_native_properties_t), POINTER(ur_kernel_handle_t) )

###############################################################################
## @brief Function-pointer for urKernelSetArgValue
if __use_win_types:
    _urKernelSetArgValue_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, c_size_t, POINTER(ur_kernel_arg_value_properties_t), c_void_p )
else:
    _urKernelSetArgValue_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, c_size_t, POINTER(ur_kernel_arg_value_properties_t), c_void_p )

###############################################################################
## @brief Function-pointer for urKernelSetArgLocal
if __use_win_types:
    _urKernelSetArgLocal_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, c_size_t, POINTER(ur_kernel_arg_local_properties_t) )
else:
    _urKernelSetArgLocal_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, c_size_t, POINTER(ur_kernel_arg_local_properties_t) )

###############################################################################
## @brief Function-pointer for urKernelSetArgPointer
if __use_win_types:
    _urKernelSetArgPointer_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, POINTER(ur_kernel_arg_pointer_properties_t), c_void_p )
else:
    _urKernelSetArgPointer_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, POINTER(ur_kernel_arg_pointer_properties_t), c_void_p )

###############################################################################
## @brief Function-pointer for urKernelSetExecInfo
if __use_win_types:
    _urKernelSetExecInfo_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_kernel_exec_info_t, c_size_t, POINTER(ur_kernel_exec_info_properties_t), c_void_p )
else:
    _urKernelSetExecInfo_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_kernel_exec_info_t, c_size_t, POINTER(ur_kernel_exec_info_properties_t), c_void_p )

###############################################################################
## @brief Function-pointer for urKernelSetArgSampler
if __use_win_types:
    _urKernelSetArgSampler_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, POINTER(ur_kernel_arg_sampler_properties_t), ur_sampler_handle_t )
else:
    _urKernelSetArgSampler_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, POINTER(ur_kernel_arg_sampler_properties_t), ur_sampler_handle_t )

###############################################################################
## @brief Function-pointer for urKernelSetArgMemObj
if __use_win_types:
    _urKernelSetArgMemObj_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, POINTER(ur_kernel_arg_mem_obj_properties_t), ur_mem_handle_t )
else:
    _urKernelSetArgMemObj_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, POINTER(ur_kernel_arg_mem_obj_properties_t), ur_mem_handle_t )

###############################################################################
## @brief Function-pointer for urKernelSetSpecializationConstants
if __use_win_types:
    _urKernelSetSpecializationConstants_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, POINTER(ur_specialization_constant_info_t) )
else:
    _urKernelSetSpecializationConstants_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, POINTER(ur_specialization_constant_info_t) )


###############################################################################
## @brief Table of Kernel functions pointers
class ur_kernel_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _urKernelCreate_t
        ("pfnGetInfo", c_void_p),                                       ## _urKernelGetInfo_t
        ("pfnGetGroupInfo", c_void_p),                                  ## _urKernelGetGroupInfo_t
        ("pfnGetSubGroupInfo", c_void_p),                               ## _urKernelGetSubGroupInfo_t
        ("pfnRetain", c_void_p),                                        ## _urKernelRetain_t
        ("pfnRelease", c_void_p),                                       ## _urKernelRelease_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urKernelGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p),                        ## _urKernelCreateWithNativeHandle_t
        ("pfnSetArgValue", c_void_p),                                   ## _urKernelSetArgValue_t
        ("pfnSetArgLocal", c_void_p),                                   ## _urKernelSetArgLocal_t
        ("pfnSetArgPointer", c_void_p),                                 ## _urKernelSetArgPointer_t
        ("pfnSetExecInfo", c_void_p),                                   ## _urKernelSetExecInfo_t
        ("pfnSetArgSampler", c_void_p),                                 ## _urKernelSetArgSampler_t
        ("pfnSetArgMemObj", c_void_p),                                  ## _urKernelSetArgMemObj_t
        ("pfnSetSpecializationConstants", c_void_p)                     ## _urKernelSetSpecializationConstants_t
    ]

###############################################################################
## @brief Function-pointer for urKernelSuggestMaxCooperativeGroupCountExp
if __use_win_types:
    _urKernelSuggestMaxCooperativeGroupCountExp_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, POINTER(c_ulong) )
else:
    _urKernelSuggestMaxCooperativeGroupCountExp_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, POINTER(c_ulong) )


###############################################################################
## @brief Table of KernelExp functions pointers
class ur_kernel_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnSuggestMaxCooperativeGroupCountExp", c_void_p)             ## _urKernelSuggestMaxCooperativeGroupCountExp_t
    ]

###############################################################################
## @brief Function-pointer for urSamplerCreate
if __use_win_types:
    _urSamplerCreate_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_sampler_desc_t), POINTER(ur_sampler_handle_t) )
else:
    _urSamplerCreate_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_sampler_desc_t), POINTER(ur_sampler_handle_t) )

###############################################################################
## @brief Function-pointer for urSamplerRetain
if __use_win_types:
    _urSamplerRetain_t = WINFUNCTYPE( ur_result_t, ur_sampler_handle_t )
else:
    _urSamplerRetain_t = CFUNCTYPE( ur_result_t, ur_sampler_handle_t )

###############################################################################
## @brief Function-pointer for urSamplerRelease
if __use_win_types:
    _urSamplerRelease_t = WINFUNCTYPE( ur_result_t, ur_sampler_handle_t )
else:
    _urSamplerRelease_t = CFUNCTYPE( ur_result_t, ur_sampler_handle_t )

###############################################################################
## @brief Function-pointer for urSamplerGetInfo
if __use_win_types:
    _urSamplerGetInfo_t = WINFUNCTYPE( ur_result_t, ur_sampler_handle_t, ur_sampler_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urSamplerGetInfo_t = CFUNCTYPE( ur_result_t, ur_sampler_handle_t, ur_sampler_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urSamplerGetNativeHandle
if __use_win_types:
    _urSamplerGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_sampler_handle_t, POINTER(ur_native_handle_t) )
else:
    _urSamplerGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_sampler_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urSamplerCreateWithNativeHandle
if __use_win_types:
    _urSamplerCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, POINTER(ur_sampler_native_properties_t), POINTER(ur_sampler_handle_t) )
else:
    _urSamplerCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, POINTER(ur_sampler_native_properties_t), POINTER(ur_sampler_handle_t) )


###############################################################################
## @brief Table of Sampler functions pointers
class ur_sampler_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _urSamplerCreate_t
        ("pfnRetain", c_void_p),                                        ## _urSamplerRetain_t
        ("pfnRelease", c_void_p),                                       ## _urSamplerRelease_t
        ("pfnGetInfo", c_void_p),                                       ## _urSamplerGetInfo_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urSamplerGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p)                         ## _urSamplerCreateWithNativeHandle_t
    ]

###############################################################################
## @brief Function-pointer for urMemImageCreate
if __use_win_types:
    _urMemImageCreate_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_mem_flags_t, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), c_void_p, POINTER(ur_mem_handle_t) )
else:
    _urMemImageCreate_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_mem_flags_t, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), c_void_p, POINTER(ur_mem_handle_t) )

###############################################################################
## @brief Function-pointer for urMemBufferCreate
if __use_win_types:
    _urMemBufferCreate_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_mem_flags_t, c_size_t, POINTER(ur_buffer_properties_t), POINTER(ur_mem_handle_t) )
else:
    _urMemBufferCreate_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_mem_flags_t, c_size_t, POINTER(ur_buffer_properties_t), POINTER(ur_mem_handle_t) )

###############################################################################
## @brief Function-pointer for urMemRetain
if __use_win_types:
    _urMemRetain_t = WINFUNCTYPE( ur_result_t, ur_mem_handle_t )
else:
    _urMemRetain_t = CFUNCTYPE( ur_result_t, ur_mem_handle_t )

###############################################################################
## @brief Function-pointer for urMemRelease
if __use_win_types:
    _urMemRelease_t = WINFUNCTYPE( ur_result_t, ur_mem_handle_t )
else:
    _urMemRelease_t = CFUNCTYPE( ur_result_t, ur_mem_handle_t )

###############################################################################
## @brief Function-pointer for urMemBufferPartition
if __use_win_types:
    _urMemBufferPartition_t = WINFUNCTYPE( ur_result_t, ur_mem_handle_t, ur_mem_flags_t, ur_buffer_create_type_t, POINTER(ur_buffer_region_t), POINTER(ur_mem_handle_t) )
else:
    _urMemBufferPartition_t = CFUNCTYPE( ur_result_t, ur_mem_handle_t, ur_mem_flags_t, ur_buffer_create_type_t, POINTER(ur_buffer_region_t), POINTER(ur_mem_handle_t) )

###############################################################################
## @brief Function-pointer for urMemGetNativeHandle
if __use_win_types:
    _urMemGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_mem_handle_t, POINTER(ur_native_handle_t) )
else:
    _urMemGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_mem_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urMemBufferCreateWithNativeHandle
if __use_win_types:
    _urMemBufferCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, POINTER(ur_mem_native_properties_t), POINTER(ur_mem_handle_t) )
else:
    _urMemBufferCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, POINTER(ur_mem_native_properties_t), POINTER(ur_mem_handle_t) )

###############################################################################
## @brief Function-pointer for urMemImageCreateWithNativeHandle
if __use_win_types:
    _urMemImageCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), POINTER(ur_mem_native_properties_t), POINTER(ur_mem_handle_t) )
else:
    _urMemImageCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), POINTER(ur_mem_native_properties_t), POINTER(ur_mem_handle_t) )

###############################################################################
## @brief Function-pointer for urMemGetInfo
if __use_win_types:
    _urMemGetInfo_t = WINFUNCTYPE( ur_result_t, ur_mem_handle_t, ur_mem_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urMemGetInfo_t = CFUNCTYPE( ur_result_t, ur_mem_handle_t, ur_mem_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urMemImageGetInfo
if __use_win_types:
    _urMemImageGetInfo_t = WINFUNCTYPE( ur_result_t, ur_mem_handle_t, ur_image_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urMemImageGetInfo_t = CFUNCTYPE( ur_result_t, ur_mem_handle_t, ur_image_info_t, c_size_t, c_void_p, POINTER(c_size_t) )


###############################################################################
## @brief Table of Mem functions pointers
class ur_mem_dditable_t(Structure):
    _fields_ = [
        ("pfnImageCreate", c_void_p),                                   ## _urMemImageCreate_t
        ("pfnBufferCreate", c_void_p),                                  ## _urMemBufferCreate_t
        ("pfnRetain", c_void_p),                                        ## _urMemRetain_t
        ("pfnRelease", c_void_p),                                       ## _urMemRelease_t
        ("pfnBufferPartition", c_void_p),                               ## _urMemBufferPartition_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urMemGetNativeHandle_t
        ("pfnBufferCreateWithNativeHandle", c_void_p),                  ## _urMemBufferCreateWithNativeHandle_t
        ("pfnImageCreateWithNativeHandle", c_void_p),                   ## _urMemImageCreateWithNativeHandle_t
        ("pfnGetInfo", c_void_p),                                       ## _urMemGetInfo_t
        ("pfnImageGetInfo", c_void_p)                                   ## _urMemImageGetInfo_t
    ]

###############################################################################
## @brief Function-pointer for urPhysicalMemCreate
if __use_win_types:
    _urPhysicalMemCreate_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, c_size_t, POINTER(ur_physical_mem_properties_t), POINTER(ur_physical_mem_handle_t) )
else:
    _urPhysicalMemCreate_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, c_size_t, POINTER(ur_physical_mem_properties_t), POINTER(ur_physical_mem_handle_t) )

###############################################################################
## @brief Function-pointer for urPhysicalMemRetain
if __use_win_types:
    _urPhysicalMemRetain_t = WINFUNCTYPE( ur_result_t, ur_physical_mem_handle_t )
else:
    _urPhysicalMemRetain_t = CFUNCTYPE( ur_result_t, ur_physical_mem_handle_t )

###############################################################################
## @brief Function-pointer for urPhysicalMemRelease
if __use_win_types:
    _urPhysicalMemRelease_t = WINFUNCTYPE( ur_result_t, ur_physical_mem_handle_t )
else:
    _urPhysicalMemRelease_t = CFUNCTYPE( ur_result_t, ur_physical_mem_handle_t )


###############################################################################
## @brief Table of PhysicalMem functions pointers
class ur_physical_mem_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _urPhysicalMemCreate_t
        ("pfnRetain", c_void_p),                                        ## _urPhysicalMemRetain_t
        ("pfnRelease", c_void_p)                                        ## _urPhysicalMemRelease_t
    ]

###############################################################################
## @brief Function-pointer for urAdapterGet
if __use_win_types:
    _urAdapterGet_t = WINFUNCTYPE( ur_result_t, c_ulong, POINTER(ur_adapter_handle_t), POINTER(c_ulong) )
else:
    _urAdapterGet_t = CFUNCTYPE( ur_result_t, c_ulong, POINTER(ur_adapter_handle_t), POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for urAdapterRelease
if __use_win_types:
    _urAdapterRelease_t = WINFUNCTYPE( ur_result_t, ur_adapter_handle_t )
else:
    _urAdapterRelease_t = CFUNCTYPE( ur_result_t, ur_adapter_handle_t )

###############################################################################
## @brief Function-pointer for urAdapterRetain
if __use_win_types:
    _urAdapterRetain_t = WINFUNCTYPE( ur_result_t, ur_adapter_handle_t )
else:
    _urAdapterRetain_t = CFUNCTYPE( ur_result_t, ur_adapter_handle_t )

###############################################################################
## @brief Function-pointer for urAdapterGetLastError
if __use_win_types:
    _urAdapterGetLastError_t = WINFUNCTYPE( ur_result_t, ur_adapter_handle_t, POINTER(c_char_p), POINTER(c_long) )
else:
    _urAdapterGetLastError_t = CFUNCTYPE( ur_result_t, ur_adapter_handle_t, POINTER(c_char_p), POINTER(c_long) )

###############################################################################
## @brief Function-pointer for urAdapterGetInfo
if __use_win_types:
    _urAdapterGetInfo_t = WINFUNCTYPE( ur_result_t, ur_adapter_handle_t, ur_adapter_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urAdapterGetInfo_t = CFUNCTYPE( ur_result_t, ur_adapter_handle_t, ur_adapter_info_t, c_size_t, c_void_p, POINTER(c_size_t) )


###############################################################################
## @brief Table of Global functions pointers
class ur_global_dditable_t(Structure):
    _fields_ = [
        ("pfnAdapterGet", c_void_p),                                    ## _urAdapterGet_t
        ("pfnAdapterRelease", c_void_p),                                ## _urAdapterRelease_t
        ("pfnAdapterRetain", c_void_p),                                 ## _urAdapterRetain_t
        ("pfnAdapterGetLastError", c_void_p),                           ## _urAdapterGetLastError_t
        ("pfnAdapterGetInfo", c_void_p)                                 ## _urAdapterGetInfo_t
    ]

###############################################################################
## @brief Function-pointer for urEnqueueKernelLaunch
if __use_win_types:
    _urEnqueueKernelLaunch_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_kernel_handle_t, c_ulong, POINTER(c_size_t), POINTER(c_size_t), POINTER(c_size_t), c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueKernelLaunch_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_kernel_handle_t, c_ulong, POINTER(c_size_t), POINTER(c_size_t), POINTER(c_size_t), c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueEventsWait
if __use_win_types:
    _urEnqueueEventsWait_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueEventsWait_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueEventsWaitWithBarrier
if __use_win_types:
    _urEnqueueEventsWaitWithBarrier_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueEventsWaitWithBarrier_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemBufferRead
if __use_win_types:
    _urEnqueueMemBufferRead_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemBufferRead_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemBufferWrite
if __use_win_types:
    _urEnqueueMemBufferWrite_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemBufferWrite_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemBufferReadRect
if __use_win_types:
    _urEnqueueMemBufferReadRect_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemBufferReadRect_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemBufferWriteRect
if __use_win_types:
    _urEnqueueMemBufferWriteRect_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemBufferWriteRect_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemBufferCopy
if __use_win_types:
    _urEnqueueMemBufferCopy_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, ur_mem_handle_t, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemBufferCopy_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, ur_mem_handle_t, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemBufferCopyRect
if __use_win_types:
    _urEnqueueMemBufferCopyRect_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, ur_mem_handle_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemBufferCopyRect_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, ur_mem_handle_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemBufferFill
if __use_win_types:
    _urEnqueueMemBufferFill_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_void_p, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemBufferFill_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_void_p, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemImageRead
if __use_win_types:
    _urEnqueueMemImageRead_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemImageRead_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemImageWrite
if __use_win_types:
    _urEnqueueMemImageWrite_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemImageWrite_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemImageCopy
if __use_win_types:
    _urEnqueueMemImageCopy_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, ur_mem_handle_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemImageCopy_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, ur_mem_handle_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemBufferMap
if __use_win_types:
    _urEnqueueMemBufferMap_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, ur_map_flags_t, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t), POINTER(c_void_p) )
else:
    _urEnqueueMemBufferMap_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_bool, ur_map_flags_t, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t), POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for urEnqueueMemUnmap
if __use_win_types:
    _urEnqueueMemUnmap_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemUnmap_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueUSMFill
if __use_win_types:
    _urEnqueueUSMFill_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_size_t, c_void_p, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueUSMFill_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_size_t, c_void_p, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueUSMMemcpy
if __use_win_types:
    _urEnqueueUSMMemcpy_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, c_bool, c_void_p, c_void_p, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueUSMMemcpy_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, c_bool, c_void_p, c_void_p, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueUSMPrefetch
if __use_win_types:
    _urEnqueueUSMPrefetch_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_size_t, ur_usm_migration_flags_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueUSMPrefetch_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_size_t, ur_usm_migration_flags_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueUSMAdvise
if __use_win_types:
    _urEnqueueUSMAdvise_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_size_t, ur_usm_advice_flags_t, POINTER(ur_event_handle_t) )
else:
    _urEnqueueUSMAdvise_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_size_t, ur_usm_advice_flags_t, POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueUSMFill2D
if __use_win_types:
    _urEnqueueUSMFill2D_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_size_t, c_size_t, c_void_p, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueUSMFill2D_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_size_t, c_size_t, c_void_p, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueUSMMemcpy2D
if __use_win_types:
    _urEnqueueUSMMemcpy2D_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, c_bool, c_void_p, c_size_t, c_void_p, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueUSMMemcpy2D_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, c_bool, c_void_p, c_size_t, c_void_p, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueDeviceGlobalVariableWrite
if __use_win_types:
    _urEnqueueDeviceGlobalVariableWrite_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_program_handle_t, c_char_p, c_bool, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueDeviceGlobalVariableWrite_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_program_handle_t, c_char_p, c_bool, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueDeviceGlobalVariableRead
if __use_win_types:
    _urEnqueueDeviceGlobalVariableRead_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_program_handle_t, c_char_p, c_bool, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueDeviceGlobalVariableRead_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_program_handle_t, c_char_p, c_bool, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueReadHostPipe
if __use_win_types:
    _urEnqueueReadHostPipe_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_program_handle_t, c_char_p, c_bool, c_void_p, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueReadHostPipe_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_program_handle_t, c_char_p, c_bool, c_void_p, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueWriteHostPipe
if __use_win_types:
    _urEnqueueWriteHostPipe_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_program_handle_t, c_char_p, c_bool, c_void_p, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueWriteHostPipe_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_program_handle_t, c_char_p, c_bool, c_void_p, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )


###############################################################################
## @brief Table of Enqueue functions pointers
class ur_enqueue_dditable_t(Structure):
    _fields_ = [
        ("pfnKernelLaunch", c_void_p),                                  ## _urEnqueueKernelLaunch_t
        ("pfnEventsWait", c_void_p),                                    ## _urEnqueueEventsWait_t
        ("pfnEventsWaitWithBarrier", c_void_p),                         ## _urEnqueueEventsWaitWithBarrier_t
        ("pfnMemBufferRead", c_void_p),                                 ## _urEnqueueMemBufferRead_t
        ("pfnMemBufferWrite", c_void_p),                                ## _urEnqueueMemBufferWrite_t
        ("pfnMemBufferReadRect", c_void_p),                             ## _urEnqueueMemBufferReadRect_t
        ("pfnMemBufferWriteRect", c_void_p),                            ## _urEnqueueMemBufferWriteRect_t
        ("pfnMemBufferCopy", c_void_p),                                 ## _urEnqueueMemBufferCopy_t
        ("pfnMemBufferCopyRect", c_void_p),                             ## _urEnqueueMemBufferCopyRect_t
        ("pfnMemBufferFill", c_void_p),                                 ## _urEnqueueMemBufferFill_t
        ("pfnMemImageRead", c_void_p),                                  ## _urEnqueueMemImageRead_t
        ("pfnMemImageWrite", c_void_p),                                 ## _urEnqueueMemImageWrite_t
        ("pfnMemImageCopy", c_void_p),                                  ## _urEnqueueMemImageCopy_t
        ("pfnMemBufferMap", c_void_p),                                  ## _urEnqueueMemBufferMap_t
        ("pfnMemUnmap", c_void_p),                                      ## _urEnqueueMemUnmap_t
        ("pfnUSMFill", c_void_p),                                       ## _urEnqueueUSMFill_t
        ("pfnUSMMemcpy", c_void_p),                                     ## _urEnqueueUSMMemcpy_t
        ("pfnUSMPrefetch", c_void_p),                                   ## _urEnqueueUSMPrefetch_t
        ("pfnUSMAdvise", c_void_p),                                     ## _urEnqueueUSMAdvise_t
        ("pfnUSMFill2D", c_void_p),                                     ## _urEnqueueUSMFill2D_t
        ("pfnUSMMemcpy2D", c_void_p),                                   ## _urEnqueueUSMMemcpy2D_t
        ("pfnDeviceGlobalVariableWrite", c_void_p),                     ## _urEnqueueDeviceGlobalVariableWrite_t
        ("pfnDeviceGlobalVariableRead", c_void_p),                      ## _urEnqueueDeviceGlobalVariableRead_t
        ("pfnReadHostPipe", c_void_p),                                  ## _urEnqueueReadHostPipe_t
        ("pfnWriteHostPipe", c_void_p)                                  ## _urEnqueueWriteHostPipe_t
    ]

###############################################################################
## @brief Function-pointer for urEnqueueCooperativeKernelLaunchExp
if __use_win_types:
    _urEnqueueCooperativeKernelLaunchExp_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_kernel_handle_t, c_ulong, POINTER(c_size_t), POINTER(c_size_t), POINTER(c_size_t), c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueCooperativeKernelLaunchExp_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_kernel_handle_t, c_ulong, POINTER(c_size_t), POINTER(c_size_t), POINTER(c_size_t), c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )


###############################################################################
## @brief Table of EnqueueExp functions pointers
class ur_enqueue_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnCooperativeKernelLaunchExp", c_void_p)                     ## _urEnqueueCooperativeKernelLaunchExp_t
    ]

###############################################################################
## @brief Function-pointer for urQueueGetInfo
if __use_win_types:
    _urQueueGetInfo_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_queue_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urQueueGetInfo_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_queue_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urQueueCreate
if __use_win_types:
    _urQueueCreate_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_queue_properties_t), POINTER(ur_queue_handle_t) )
else:
    _urQueueCreate_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_queue_properties_t), POINTER(ur_queue_handle_t) )

###############################################################################
## @brief Function-pointer for urQueueRetain
if __use_win_types:
    _urQueueRetain_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t )
else:
    _urQueueRetain_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t )

###############################################################################
## @brief Function-pointer for urQueueRelease
if __use_win_types:
    _urQueueRelease_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t )
else:
    _urQueueRelease_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t )

###############################################################################
## @brief Function-pointer for urQueueGetNativeHandle
if __use_win_types:
    _urQueueGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, POINTER(ur_queue_native_desc_t), POINTER(ur_native_handle_t) )
else:
    _urQueueGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, POINTER(ur_queue_native_desc_t), POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urQueueCreateWithNativeHandle
if __use_win_types:
    _urQueueCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_queue_native_properties_t), POINTER(ur_queue_handle_t) )
else:
    _urQueueCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_native_handle_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_queue_native_properties_t), POINTER(ur_queue_handle_t) )

###############################################################################
## @brief Function-pointer for urQueueFinish
if __use_win_types:
    _urQueueFinish_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t )
else:
    _urQueueFinish_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t )

###############################################################################
## @brief Function-pointer for urQueueFlush
if __use_win_types:
    _urQueueFlush_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t )
else:
    _urQueueFlush_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t )


###############################################################################
## @brief Table of Queue functions pointers
class ur_queue_dditable_t(Structure):
    _fields_ = [
        ("pfnGetInfo", c_void_p),                                       ## _urQueueGetInfo_t
        ("pfnCreate", c_void_p),                                        ## _urQueueCreate_t
        ("pfnRetain", c_void_p),                                        ## _urQueueRetain_t
        ("pfnRelease", c_void_p),                                       ## _urQueueRelease_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urQueueGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p),                        ## _urQueueCreateWithNativeHandle_t
        ("pfnFinish", c_void_p),                                        ## _urQueueFinish_t
        ("pfnFlush", c_void_p)                                          ## _urQueueFlush_t
    ]

###############################################################################
## @brief Function-pointer for urBindlessImagesUnsampledImageHandleDestroyExp
if __use_win_types:
    _urBindlessImagesUnsampledImageHandleDestroyExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_handle_t )
else:
    _urBindlessImagesUnsampledImageHandleDestroyExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_handle_t )

###############################################################################
## @brief Function-pointer for urBindlessImagesSampledImageHandleDestroyExp
if __use_win_types:
    _urBindlessImagesSampledImageHandleDestroyExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_handle_t )
else:
    _urBindlessImagesSampledImageHandleDestroyExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_handle_t )

###############################################################################
## @brief Function-pointer for urBindlessImagesImageAllocateExp
if __use_win_types:
    _urBindlessImagesImageAllocateExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), POINTER(ur_exp_image_mem_handle_t) )
else:
    _urBindlessImagesImageAllocateExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), POINTER(ur_exp_image_mem_handle_t) )

###############################################################################
## @brief Function-pointer for urBindlessImagesImageFreeExp
if __use_win_types:
    _urBindlessImagesImageFreeExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_mem_handle_t )
else:
    _urBindlessImagesImageFreeExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_mem_handle_t )

###############################################################################
## @brief Function-pointer for urBindlessImagesUnsampledImageCreateExp
if __use_win_types:
    _urBindlessImagesUnsampledImageCreateExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_mem_handle_t, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), POINTER(ur_mem_handle_t), POINTER(ur_exp_image_handle_t) )
else:
    _urBindlessImagesUnsampledImageCreateExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_mem_handle_t, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), POINTER(ur_mem_handle_t), POINTER(ur_exp_image_handle_t) )

###############################################################################
## @brief Function-pointer for urBindlessImagesSampledImageCreateExp
if __use_win_types:
    _urBindlessImagesSampledImageCreateExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_mem_handle_t, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), ur_sampler_handle_t, POINTER(ur_mem_handle_t), POINTER(ur_exp_image_handle_t) )
else:
    _urBindlessImagesSampledImageCreateExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_mem_handle_t, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), ur_sampler_handle_t, POINTER(ur_mem_handle_t), POINTER(ur_exp_image_handle_t) )

###############################################################################
## @brief Function-pointer for urBindlessImagesImageCopyExp
if __use_win_types:
    _urBindlessImagesImageCopyExp_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_void_p, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), ur_exp_image_copy_flags_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, ur_rect_region_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urBindlessImagesImageCopyExp_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_void_p, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), ur_exp_image_copy_flags_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, ur_rect_region_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urBindlessImagesImageGetInfoExp
if __use_win_types:
    _urBindlessImagesImageGetInfoExp_t = WINFUNCTYPE( ur_result_t, ur_exp_image_mem_handle_t, ur_image_info_t, c_void_p, POINTER(c_size_t) )
else:
    _urBindlessImagesImageGetInfoExp_t = CFUNCTYPE( ur_result_t, ur_exp_image_mem_handle_t, ur_image_info_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urBindlessImagesMipmapGetLevelExp
if __use_win_types:
    _urBindlessImagesMipmapGetLevelExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_mem_handle_t, c_ulong, POINTER(ur_exp_image_mem_handle_t) )
else:
    _urBindlessImagesMipmapGetLevelExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_mem_handle_t, c_ulong, POINTER(ur_exp_image_mem_handle_t) )

###############################################################################
## @brief Function-pointer for urBindlessImagesMipmapFreeExp
if __use_win_types:
    _urBindlessImagesMipmapFreeExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_mem_handle_t )
else:
    _urBindlessImagesMipmapFreeExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_image_mem_handle_t )

###############################################################################
## @brief Function-pointer for urBindlessImagesImportOpaqueFDExp
if __use_win_types:
    _urBindlessImagesImportOpaqueFDExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, c_size_t, POINTER(ur_exp_interop_mem_desc_t), POINTER(ur_exp_interop_mem_handle_t) )
else:
    _urBindlessImagesImportOpaqueFDExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, c_size_t, POINTER(ur_exp_interop_mem_desc_t), POINTER(ur_exp_interop_mem_handle_t) )

###############################################################################
## @brief Function-pointer for urBindlessImagesMapExternalArrayExp
if __use_win_types:
    _urBindlessImagesMapExternalArrayExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), ur_exp_interop_mem_handle_t, POINTER(ur_exp_image_mem_handle_t) )
else:
    _urBindlessImagesMapExternalArrayExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_image_format_t), POINTER(ur_image_desc_t), ur_exp_interop_mem_handle_t, POINTER(ur_exp_image_mem_handle_t) )

###############################################################################
## @brief Function-pointer for urBindlessImagesReleaseInteropExp
if __use_win_types:
    _urBindlessImagesReleaseInteropExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_interop_mem_handle_t )
else:
    _urBindlessImagesReleaseInteropExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_interop_mem_handle_t )

###############################################################################
## @brief Function-pointer for urBindlessImagesImportExternalSemaphoreOpaqueFDExp
if __use_win_types:
    _urBindlessImagesImportExternalSemaphoreOpaqueFDExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_exp_interop_semaphore_desc_t), POINTER(ur_exp_interop_semaphore_handle_t) )
else:
    _urBindlessImagesImportExternalSemaphoreOpaqueFDExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_exp_interop_semaphore_desc_t), POINTER(ur_exp_interop_semaphore_handle_t) )

###############################################################################
## @brief Function-pointer for urBindlessImagesDestroyExternalSemaphoreExp
if __use_win_types:
    _urBindlessImagesDestroyExternalSemaphoreExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_interop_semaphore_handle_t )
else:
    _urBindlessImagesDestroyExternalSemaphoreExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_exp_interop_semaphore_handle_t )

###############################################################################
## @brief Function-pointer for urBindlessImagesWaitExternalSemaphoreExp
if __use_win_types:
    _urBindlessImagesWaitExternalSemaphoreExp_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_exp_interop_semaphore_handle_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urBindlessImagesWaitExternalSemaphoreExp_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_exp_interop_semaphore_handle_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urBindlessImagesSignalExternalSemaphoreExp
if __use_win_types:
    _urBindlessImagesSignalExternalSemaphoreExp_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_exp_interop_semaphore_handle_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urBindlessImagesSignalExternalSemaphoreExp_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_exp_interop_semaphore_handle_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )


###############################################################################
## @brief Table of BindlessImagesExp functions pointers
class ur_bindless_images_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnUnsampledImageHandleDestroyExp", c_void_p),                ## _urBindlessImagesUnsampledImageHandleDestroyExp_t
        ("pfnSampledImageHandleDestroyExp", c_void_p),                  ## _urBindlessImagesSampledImageHandleDestroyExp_t
        ("pfnImageAllocateExp", c_void_p),                              ## _urBindlessImagesImageAllocateExp_t
        ("pfnImageFreeExp", c_void_p),                                  ## _urBindlessImagesImageFreeExp_t
        ("pfnUnsampledImageCreateExp", c_void_p),                       ## _urBindlessImagesUnsampledImageCreateExp_t
        ("pfnSampledImageCreateExp", c_void_p),                         ## _urBindlessImagesSampledImageCreateExp_t
        ("pfnImageCopyExp", c_void_p),                                  ## _urBindlessImagesImageCopyExp_t
        ("pfnImageGetInfoExp", c_void_p),                               ## _urBindlessImagesImageGetInfoExp_t
        ("pfnMipmapGetLevelExp", c_void_p),                             ## _urBindlessImagesMipmapGetLevelExp_t
        ("pfnMipmapFreeExp", c_void_p),                                 ## _urBindlessImagesMipmapFreeExp_t
        ("pfnImportOpaqueFDExp", c_void_p),                             ## _urBindlessImagesImportOpaqueFDExp_t
        ("pfnMapExternalArrayExp", c_void_p),                           ## _urBindlessImagesMapExternalArrayExp_t
        ("pfnReleaseInteropExp", c_void_p),                             ## _urBindlessImagesReleaseInteropExp_t
        ("pfnImportExternalSemaphoreOpaqueFDExp", c_void_p),            ## _urBindlessImagesImportExternalSemaphoreOpaqueFDExp_t
        ("pfnDestroyExternalSemaphoreExp", c_void_p),                   ## _urBindlessImagesDestroyExternalSemaphoreExp_t
        ("pfnWaitExternalSemaphoreExp", c_void_p),                      ## _urBindlessImagesWaitExternalSemaphoreExp_t
        ("pfnSignalExternalSemaphoreExp", c_void_p)                     ## _urBindlessImagesSignalExternalSemaphoreExp_t
    ]

###############################################################################
## @brief Function-pointer for urUSMHostAlloc
if __use_win_types:
    _urUSMHostAlloc_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_usm_desc_t), ur_usm_pool_handle_t, c_size_t, POINTER(c_void_p) )
else:
    _urUSMHostAlloc_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_usm_desc_t), ur_usm_pool_handle_t, c_size_t, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for urUSMDeviceAlloc
if __use_win_types:
    _urUSMDeviceAlloc_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_usm_desc_t), ur_usm_pool_handle_t, c_size_t, POINTER(c_void_p) )
else:
    _urUSMDeviceAlloc_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_usm_desc_t), ur_usm_pool_handle_t, c_size_t, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for urUSMSharedAlloc
if __use_win_types:
    _urUSMSharedAlloc_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_usm_desc_t), ur_usm_pool_handle_t, c_size_t, POINTER(c_void_p) )
else:
    _urUSMSharedAlloc_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_usm_desc_t), ur_usm_pool_handle_t, c_size_t, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for urUSMFree
if __use_win_types:
    _urUSMFree_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p )
else:
    _urUSMFree_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p )

###############################################################################
## @brief Function-pointer for urUSMGetMemAllocInfo
if __use_win_types:
    _urUSMGetMemAllocInfo_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, ur_usm_alloc_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urUSMGetMemAllocInfo_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, ur_usm_alloc_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urUSMPoolCreate
if __use_win_types:
    _urUSMPoolCreate_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_usm_pool_desc_t), POINTER(ur_usm_pool_handle_t) )
else:
    _urUSMPoolCreate_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_usm_pool_desc_t), POINTER(ur_usm_pool_handle_t) )

###############################################################################
## @brief Function-pointer for urUSMPoolRetain
if __use_win_types:
    _urUSMPoolRetain_t = WINFUNCTYPE( ur_result_t, ur_usm_pool_handle_t )
else:
    _urUSMPoolRetain_t = CFUNCTYPE( ur_result_t, ur_usm_pool_handle_t )

###############################################################################
## @brief Function-pointer for urUSMPoolRelease
if __use_win_types:
    _urUSMPoolRelease_t = WINFUNCTYPE( ur_result_t, ur_usm_pool_handle_t )
else:
    _urUSMPoolRelease_t = CFUNCTYPE( ur_result_t, ur_usm_pool_handle_t )

###############################################################################
## @brief Function-pointer for urUSMPoolGetInfo
if __use_win_types:
    _urUSMPoolGetInfo_t = WINFUNCTYPE( ur_result_t, ur_usm_pool_handle_t, ur_usm_pool_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urUSMPoolGetInfo_t = CFUNCTYPE( ur_result_t, ur_usm_pool_handle_t, ur_usm_pool_info_t, c_size_t, c_void_p, POINTER(c_size_t) )


###############################################################################
## @brief Table of USM functions pointers
class ur_usm_dditable_t(Structure):
    _fields_ = [
        ("pfnHostAlloc", c_void_p),                                     ## _urUSMHostAlloc_t
        ("pfnDeviceAlloc", c_void_p),                                   ## _urUSMDeviceAlloc_t
        ("pfnSharedAlloc", c_void_p),                                   ## _urUSMSharedAlloc_t
        ("pfnFree", c_void_p),                                          ## _urUSMFree_t
        ("pfnGetMemAllocInfo", c_void_p),                               ## _urUSMGetMemAllocInfo_t
        ("pfnPoolCreate", c_void_p),                                    ## _urUSMPoolCreate_t
        ("pfnPoolRetain", c_void_p),                                    ## _urUSMPoolRetain_t
        ("pfnPoolRelease", c_void_p),                                   ## _urUSMPoolRelease_t
        ("pfnPoolGetInfo", c_void_p)                                    ## _urUSMPoolGetInfo_t
    ]

###############################################################################
## @brief Function-pointer for urUSMPitchedAllocExp
if __use_win_types:
    _urUSMPitchedAllocExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_usm_desc_t), ur_usm_pool_handle_t, c_size_t, c_size_t, c_size_t, POINTER(c_void_p), POINTER(c_size_t) )
else:
    _urUSMPitchedAllocExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_usm_desc_t), ur_usm_pool_handle_t, c_size_t, c_size_t, c_size_t, POINTER(c_void_p), POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urUSMImportExp
if __use_win_types:
    _urUSMImportExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t )
else:
    _urUSMImportExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t )

###############################################################################
## @brief Function-pointer for urUSMReleaseExp
if __use_win_types:
    _urUSMReleaseExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p )
else:
    _urUSMReleaseExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p )


###############################################################################
## @brief Table of USMExp functions pointers
class ur_usm_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnPitchedAllocExp", c_void_p),                               ## _urUSMPitchedAllocExp_t
        ("pfnImportExp", c_void_p),                                     ## _urUSMImportExp_t
        ("pfnReleaseExp", c_void_p)                                     ## _urUSMReleaseExp_t
    ]

###############################################################################
## @brief Function-pointer for urCommandBufferCreateExp
if __use_win_types:
    _urCommandBufferCreateExp_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_exp_command_buffer_desc_t), POINTER(ur_exp_command_buffer_handle_t) )
else:
    _urCommandBufferCreateExp_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_exp_command_buffer_desc_t), POINTER(ur_exp_command_buffer_handle_t) )

###############################################################################
## @brief Function-pointer for urCommandBufferRetainExp
if __use_win_types:
    _urCommandBufferRetainExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t )
else:
    _urCommandBufferRetainExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t )

###############################################################################
## @brief Function-pointer for urCommandBufferReleaseExp
if __use_win_types:
    _urCommandBufferReleaseExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t )
else:
    _urCommandBufferReleaseExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t )

###############################################################################
## @brief Function-pointer for urCommandBufferFinalizeExp
if __use_win_types:
    _urCommandBufferFinalizeExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t )
else:
    _urCommandBufferFinalizeExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t )

###############################################################################
## @brief Function-pointer for urCommandBufferAppendKernelLaunchExp
if __use_win_types:
    _urCommandBufferAppendKernelLaunchExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_kernel_handle_t, c_ulong, POINTER(c_size_t), POINTER(c_size_t), POINTER(c_size_t), c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )
else:
    _urCommandBufferAppendKernelLaunchExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_kernel_handle_t, c_ulong, POINTER(c_size_t), POINTER(c_size_t), POINTER(c_size_t), c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )

###############################################################################
## @brief Function-pointer for urCommandBufferAppendUSMMemcpyExp
if __use_win_types:
    _urCommandBufferAppendUSMMemcpyExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, c_void_p, c_void_p, c_size_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )
else:
    _urCommandBufferAppendUSMMemcpyExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, c_void_p, c_void_p, c_size_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )

###############################################################################
## @brief Function-pointer for urCommandBufferAppendUSMFillExp
if __use_win_types:
    _urCommandBufferAppendUSMFillExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, c_void_p, c_void_p, c_size_t, c_size_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )
else:
    _urCommandBufferAppendUSMFillExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, c_void_p, c_void_p, c_size_t, c_size_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )

###############################################################################
## @brief Function-pointer for urCommandBufferAppendMemBufferCopyExp
if __use_win_types:
    _urCommandBufferAppendMemBufferCopyExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, ur_mem_handle_t, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )
else:
    _urCommandBufferAppendMemBufferCopyExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, ur_mem_handle_t, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )

###############################################################################
## @brief Function-pointer for urCommandBufferAppendMemBufferWriteExp
if __use_win_types:
    _urCommandBufferAppendMemBufferWriteExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )
else:
    _urCommandBufferAppendMemBufferWriteExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )

###############################################################################
## @brief Function-pointer for urCommandBufferAppendMemBufferReadExp
if __use_win_types:
    _urCommandBufferAppendMemBufferReadExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )
else:
    _urCommandBufferAppendMemBufferReadExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )

###############################################################################
## @brief Function-pointer for urCommandBufferAppendMemBufferCopyRectExp
if __use_win_types:
    _urCommandBufferAppendMemBufferCopyRectExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, ur_mem_handle_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )
else:
    _urCommandBufferAppendMemBufferCopyRectExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, ur_mem_handle_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )

###############################################################################
## @brief Function-pointer for urCommandBufferAppendMemBufferWriteRectExp
if __use_win_types:
    _urCommandBufferAppendMemBufferWriteRectExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )
else:
    _urCommandBufferAppendMemBufferWriteRectExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )

###############################################################################
## @brief Function-pointer for urCommandBufferAppendMemBufferReadRectExp
if __use_win_types:
    _urCommandBufferAppendMemBufferReadRectExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )
else:
    _urCommandBufferAppendMemBufferReadRectExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )

###############################################################################
## @brief Function-pointer for urCommandBufferAppendMemBufferFillExp
if __use_win_types:
    _urCommandBufferAppendMemBufferFillExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, c_void_p, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )
else:
    _urCommandBufferAppendMemBufferFillExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_mem_handle_t, c_void_p, c_size_t, c_size_t, c_size_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )

###############################################################################
## @brief Function-pointer for urCommandBufferAppendUSMPrefetchExp
if __use_win_types:
    _urCommandBufferAppendUSMPrefetchExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, c_void_p, c_size_t, ur_usm_migration_flags_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )
else:
    _urCommandBufferAppendUSMPrefetchExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, c_void_p, c_size_t, ur_usm_migration_flags_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )

###############################################################################
## @brief Function-pointer for urCommandBufferAppendUSMAdviseExp
if __use_win_types:
    _urCommandBufferAppendUSMAdviseExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, c_void_p, c_size_t, ur_usm_advice_flags_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )
else:
    _urCommandBufferAppendUSMAdviseExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, c_void_p, c_size_t, ur_usm_advice_flags_t, c_ulong, POINTER(ur_exp_command_buffer_sync_point_t), POINTER(ur_exp_command_buffer_sync_point_t) )

###############################################################################
## @brief Function-pointer for urCommandBufferEnqueueExp
if __use_win_types:
    _urCommandBufferEnqueueExp_t = WINFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_queue_handle_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urCommandBufferEnqueueExp_t = CFUNCTYPE( ur_result_t, ur_exp_command_buffer_handle_t, ur_queue_handle_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )


###############################################################################
## @brief Table of CommandBufferExp functions pointers
class ur_command_buffer_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnCreateExp", c_void_p),                                     ## _urCommandBufferCreateExp_t
        ("pfnRetainExp", c_void_p),                                     ## _urCommandBufferRetainExp_t
        ("pfnReleaseExp", c_void_p),                                    ## _urCommandBufferReleaseExp_t
        ("pfnFinalizeExp", c_void_p),                                   ## _urCommandBufferFinalizeExp_t
        ("pfnAppendKernelLaunchExp", c_void_p),                         ## _urCommandBufferAppendKernelLaunchExp_t
        ("pfnAppendUSMMemcpyExp", c_void_p),                            ## _urCommandBufferAppendUSMMemcpyExp_t
        ("pfnAppendUSMFillExp", c_void_p),                              ## _urCommandBufferAppendUSMFillExp_t
        ("pfnAppendMemBufferCopyExp", c_void_p),                        ## _urCommandBufferAppendMemBufferCopyExp_t
        ("pfnAppendMemBufferWriteExp", c_void_p),                       ## _urCommandBufferAppendMemBufferWriteExp_t
        ("pfnAppendMemBufferReadExp", c_void_p),                        ## _urCommandBufferAppendMemBufferReadExp_t
        ("pfnAppendMemBufferCopyRectExp", c_void_p),                    ## _urCommandBufferAppendMemBufferCopyRectExp_t
        ("pfnAppendMemBufferWriteRectExp", c_void_p),                   ## _urCommandBufferAppendMemBufferWriteRectExp_t
        ("pfnAppendMemBufferReadRectExp", c_void_p),                    ## _urCommandBufferAppendMemBufferReadRectExp_t
        ("pfnAppendMemBufferFillExp", c_void_p),                        ## _urCommandBufferAppendMemBufferFillExp_t
        ("pfnAppendUSMPrefetchExp", c_void_p),                          ## _urCommandBufferAppendUSMPrefetchExp_t
        ("pfnAppendUSMAdviseExp", c_void_p),                            ## _urCommandBufferAppendUSMAdviseExp_t
        ("pfnEnqueueExp", c_void_p)                                     ## _urCommandBufferEnqueueExp_t
    ]

###############################################################################
## @brief Function-pointer for urUsmP2PEnablePeerAccessExp
if __use_win_types:
    _urUsmP2PEnablePeerAccessExp_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t, ur_device_handle_t )
else:
    _urUsmP2PEnablePeerAccessExp_t = CFUNCTYPE( ur_result_t, ur_device_handle_t, ur_device_handle_t )

###############################################################################
## @brief Function-pointer for urUsmP2PDisablePeerAccessExp
if __use_win_types:
    _urUsmP2PDisablePeerAccessExp_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t, ur_device_handle_t )
else:
    _urUsmP2PDisablePeerAccessExp_t = CFUNCTYPE( ur_result_t, ur_device_handle_t, ur_device_handle_t )

###############################################################################
## @brief Function-pointer for urUsmP2PPeerAccessGetInfoExp
if __use_win_types:
    _urUsmP2PPeerAccessGetInfoExp_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t, ur_device_handle_t, ur_exp_peer_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urUsmP2PPeerAccessGetInfoExp_t = CFUNCTYPE( ur_result_t, ur_device_handle_t, ur_device_handle_t, ur_exp_peer_info_t, c_size_t, c_void_p, POINTER(c_size_t) )


###############################################################################
## @brief Table of UsmP2PExp functions pointers
class ur_usm_p2p_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnEnablePeerAccessExp", c_void_p),                           ## _urUsmP2PEnablePeerAccessExp_t
        ("pfnDisablePeerAccessExp", c_void_p),                          ## _urUsmP2PDisablePeerAccessExp_t
        ("pfnPeerAccessGetInfoExp", c_void_p)                           ## _urUsmP2PPeerAccessGetInfoExp_t
    ]

###############################################################################
## @brief Function-pointer for urVirtualMemGranularityGetInfo
if __use_win_types:
    _urVirtualMemGranularityGetInfo_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_virtual_mem_granularity_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urVirtualMemGranularityGetInfo_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_virtual_mem_granularity_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urVirtualMemReserve
if __use_win_types:
    _urVirtualMemReserve_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t, POINTER(c_void_p) )
else:
    _urVirtualMemReserve_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for urVirtualMemFree
if __use_win_types:
    _urVirtualMemFree_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t )
else:
    _urVirtualMemFree_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t )

###############################################################################
## @brief Function-pointer for urVirtualMemMap
if __use_win_types:
    _urVirtualMemMap_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t, ur_physical_mem_handle_t, c_size_t, ur_virtual_mem_access_flags_t )
else:
    _urVirtualMemMap_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t, ur_physical_mem_handle_t, c_size_t, ur_virtual_mem_access_flags_t )

###############################################################################
## @brief Function-pointer for urVirtualMemUnmap
if __use_win_types:
    _urVirtualMemUnmap_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t )
else:
    _urVirtualMemUnmap_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t )

###############################################################################
## @brief Function-pointer for urVirtualMemSetAccess
if __use_win_types:
    _urVirtualMemSetAccess_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t, ur_virtual_mem_access_flags_t )
else:
    _urVirtualMemSetAccess_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t, ur_virtual_mem_access_flags_t )

###############################################################################
## @brief Function-pointer for urVirtualMemGetInfo
if __use_win_types:
    _urVirtualMemGetInfo_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t, ur_virtual_mem_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urVirtualMemGetInfo_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_size_t, ur_virtual_mem_info_t, c_size_t, c_void_p, POINTER(c_size_t) )


###############################################################################
## @brief Table of VirtualMem functions pointers
class ur_virtual_mem_dditable_t(Structure):
    _fields_ = [
        ("pfnGranularityGetInfo", c_void_p),                            ## _urVirtualMemGranularityGetInfo_t
        ("pfnReserve", c_void_p),                                       ## _urVirtualMemReserve_t
        ("pfnFree", c_void_p),                                          ## _urVirtualMemFree_t
        ("pfnMap", c_void_p),                                           ## _urVirtualMemMap_t
        ("pfnUnmap", c_void_p),                                         ## _urVirtualMemUnmap_t
        ("pfnSetAccess", c_void_p),                                     ## _urVirtualMemSetAccess_t
        ("pfnGetInfo", c_void_p)                                        ## _urVirtualMemGetInfo_t
    ]

###############################################################################
## @brief Function-pointer for urDeviceGet
if __use_win_types:
    _urDeviceGet_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_device_type_t, c_ulong, POINTER(ur_device_handle_t), POINTER(c_ulong) )
else:
    _urDeviceGet_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_device_type_t, c_ulong, POINTER(ur_device_handle_t), POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for urDeviceGetInfo
if __use_win_types:
    _urDeviceGetInfo_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t, ur_device_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urDeviceGetInfo_t = CFUNCTYPE( ur_result_t, ur_device_handle_t, ur_device_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urDeviceRetain
if __use_win_types:
    _urDeviceRetain_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t )
else:
    _urDeviceRetain_t = CFUNCTYPE( ur_result_t, ur_device_handle_t )

###############################################################################
## @brief Function-pointer for urDeviceRelease
if __use_win_types:
    _urDeviceRelease_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t )
else:
    _urDeviceRelease_t = CFUNCTYPE( ur_result_t, ur_device_handle_t )

###############################################################################
## @brief Function-pointer for urDevicePartition
if __use_win_types:
    _urDevicePartition_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t, POINTER(ur_device_partition_properties_t), c_ulong, POINTER(ur_device_handle_t), POINTER(c_ulong) )
else:
    _urDevicePartition_t = CFUNCTYPE( ur_result_t, ur_device_handle_t, POINTER(ur_device_partition_properties_t), c_ulong, POINTER(ur_device_handle_t), POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for urDeviceSelectBinary
if __use_win_types:
    _urDeviceSelectBinary_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t, POINTER(ur_device_binary_t), c_ulong, POINTER(c_ulong) )
else:
    _urDeviceSelectBinary_t = CFUNCTYPE( ur_result_t, ur_device_handle_t, POINTER(ur_device_binary_t), c_ulong, POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for urDeviceGetNativeHandle
if __use_win_types:
    _urDeviceGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t, POINTER(ur_native_handle_t) )
else:
    _urDeviceGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_device_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urDeviceCreateWithNativeHandle
if __use_win_types:
    _urDeviceCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_native_handle_t, ur_platform_handle_t, POINTER(ur_device_native_properties_t), POINTER(ur_device_handle_t) )
else:
    _urDeviceCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_native_handle_t, ur_platform_handle_t, POINTER(ur_device_native_properties_t), POINTER(ur_device_handle_t) )

###############################################################################
## @brief Function-pointer for urDeviceGetGlobalTimestamps
if __use_win_types:
    _urDeviceGetGlobalTimestamps_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t, POINTER(c_ulonglong), POINTER(c_ulonglong) )
else:
    _urDeviceGetGlobalTimestamps_t = CFUNCTYPE( ur_result_t, ur_device_handle_t, POINTER(c_ulonglong), POINTER(c_ulonglong) )


###############################################################################
## @brief Table of Device functions pointers
class ur_device_dditable_t(Structure):
    _fields_ = [
        ("pfnGet", c_void_p),                                           ## _urDeviceGet_t
        ("pfnGetInfo", c_void_p),                                       ## _urDeviceGetInfo_t
        ("pfnRetain", c_void_p),                                        ## _urDeviceRetain_t
        ("pfnRelease", c_void_p),                                       ## _urDeviceRelease_t
        ("pfnPartition", c_void_p),                                     ## _urDevicePartition_t
        ("pfnSelectBinary", c_void_p),                                  ## _urDeviceSelectBinary_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urDeviceGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p),                        ## _urDeviceCreateWithNativeHandle_t
        ("pfnGetGlobalTimestamps", c_void_p)                            ## _urDeviceGetGlobalTimestamps_t
    ]

###############################################################################
class ur_dditable_t(Structure):
    _fields_ = [
        ("Platform", ur_platform_dditable_t),
        ("Context", ur_context_dditable_t),
        ("Event", ur_event_dditable_t),
        ("Program", ur_program_dditable_t),
        ("ProgramExp", ur_program_exp_dditable_t),
        ("Kernel", ur_kernel_dditable_t),
        ("KernelExp", ur_kernel_exp_dditable_t),
        ("Sampler", ur_sampler_dditable_t),
        ("Mem", ur_mem_dditable_t),
        ("PhysicalMem", ur_physical_mem_dditable_t),
        ("Global", ur_global_dditable_t),
        ("Enqueue", ur_enqueue_dditable_t),
        ("EnqueueExp", ur_enqueue_exp_dditable_t),
        ("Queue", ur_queue_dditable_t),
        ("BindlessImagesExp", ur_bindless_images_exp_dditable_t),
        ("USM", ur_usm_dditable_t),
        ("USMExp", ur_usm_exp_dditable_t),
        ("CommandBufferExp", ur_command_buffer_exp_dditable_t),
        ("UsmP2PExp", ur_usm_p2p_exp_dditable_t),
        ("VirtualMem", ur_virtual_mem_dditable_t),
        ("Device", ur_device_dditable_t)
    ]

###############################################################################
## @brief ur device-driver interfaces
class UR_DDI:
    def __init__(self, version : ur_api_version_t):
        # load the ur_loader library
        if "Windows" == platform.uname()[0]:
            self.__dll = WinDLL("ur_loader.dll", winmode=0)
        else:
            self.__dll = CDLL("libur_loader.so")

        # fill the ddi tables
        self.__dditable = ur_dditable_t()

        # initialize the UR
        self.__dll.urLoaderInit(0, 0)

        # call driver to get function pointers
        Platform = ur_platform_dditable_t()
        r = ur_result_v(self.__dll.urGetPlatformProcAddrTable(version, byref(Platform)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Platform = Platform

        # attach function interface to function address
        self.urPlatformGet = _urPlatformGet_t(self.__dditable.Platform.pfnGet)
        self.urPlatformGetInfo = _urPlatformGetInfo_t(self.__dditable.Platform.pfnGetInfo)
        self.urPlatformGetNativeHandle = _urPlatformGetNativeHandle_t(self.__dditable.Platform.pfnGetNativeHandle)
        self.urPlatformCreateWithNativeHandle = _urPlatformCreateWithNativeHandle_t(self.__dditable.Platform.pfnCreateWithNativeHandle)
        self.urPlatformGetApiVersion = _urPlatformGetApiVersion_t(self.__dditable.Platform.pfnGetApiVersion)
        self.urPlatformGetBackendOption = _urPlatformGetBackendOption_t(self.__dditable.Platform.pfnGetBackendOption)

        # call driver to get function pointers
        Context = ur_context_dditable_t()
        r = ur_result_v(self.__dll.urGetContextProcAddrTable(version, byref(Context)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Context = Context

        # attach function interface to function address
        self.urContextCreate = _urContextCreate_t(self.__dditable.Context.pfnCreate)
        self.urContextRetain = _urContextRetain_t(self.__dditable.Context.pfnRetain)
        self.urContextRelease = _urContextRelease_t(self.__dditable.Context.pfnRelease)
        self.urContextGetInfo = _urContextGetInfo_t(self.__dditable.Context.pfnGetInfo)
        self.urContextGetNativeHandle = _urContextGetNativeHandle_t(self.__dditable.Context.pfnGetNativeHandle)
        self.urContextCreateWithNativeHandle = _urContextCreateWithNativeHandle_t(self.__dditable.Context.pfnCreateWithNativeHandle)
        self.urContextSetExtendedDeleter = _urContextSetExtendedDeleter_t(self.__dditable.Context.pfnSetExtendedDeleter)

        # call driver to get function pointers
        Event = ur_event_dditable_t()
        r = ur_result_v(self.__dll.urGetEventProcAddrTable(version, byref(Event)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Event = Event

        # attach function interface to function address
        self.urEventGetInfo = _urEventGetInfo_t(self.__dditable.Event.pfnGetInfo)
        self.urEventGetProfilingInfo = _urEventGetProfilingInfo_t(self.__dditable.Event.pfnGetProfilingInfo)
        self.urEventWait = _urEventWait_t(self.__dditable.Event.pfnWait)
        self.urEventRetain = _urEventRetain_t(self.__dditable.Event.pfnRetain)
        self.urEventRelease = _urEventRelease_t(self.__dditable.Event.pfnRelease)
        self.urEventGetNativeHandle = _urEventGetNativeHandle_t(self.__dditable.Event.pfnGetNativeHandle)
        self.urEventCreateWithNativeHandle = _urEventCreateWithNativeHandle_t(self.__dditable.Event.pfnCreateWithNativeHandle)
        self.urEventSetCallback = _urEventSetCallback_t(self.__dditable.Event.pfnSetCallback)

        # call driver to get function pointers
        Program = ur_program_dditable_t()
        r = ur_result_v(self.__dll.urGetProgramProcAddrTable(version, byref(Program)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Program = Program

        # attach function interface to function address
        self.urProgramCreateWithIL = _urProgramCreateWithIL_t(self.__dditable.Program.pfnCreateWithIL)
        self.urProgramCreateWithBinary = _urProgramCreateWithBinary_t(self.__dditable.Program.pfnCreateWithBinary)
        self.urProgramBuild = _urProgramBuild_t(self.__dditable.Program.pfnBuild)
        self.urProgramCompile = _urProgramCompile_t(self.__dditable.Program.pfnCompile)
        self.urProgramLink = _urProgramLink_t(self.__dditable.Program.pfnLink)
        self.urProgramRetain = _urProgramRetain_t(self.__dditable.Program.pfnRetain)
        self.urProgramRelease = _urProgramRelease_t(self.__dditable.Program.pfnRelease)
        self.urProgramGetFunctionPointer = _urProgramGetFunctionPointer_t(self.__dditable.Program.pfnGetFunctionPointer)
        self.urProgramGetInfo = _urProgramGetInfo_t(self.__dditable.Program.pfnGetInfo)
        self.urProgramGetBuildInfo = _urProgramGetBuildInfo_t(self.__dditable.Program.pfnGetBuildInfo)
        self.urProgramSetSpecializationConstants = _urProgramSetSpecializationConstants_t(self.__dditable.Program.pfnSetSpecializationConstants)
        self.urProgramGetNativeHandle = _urProgramGetNativeHandle_t(self.__dditable.Program.pfnGetNativeHandle)
        self.urProgramCreateWithNativeHandle = _urProgramCreateWithNativeHandle_t(self.__dditable.Program.pfnCreateWithNativeHandle)

        # call driver to get function pointers
        ProgramExp = ur_program_exp_dditable_t()
        r = ur_result_v(self.__dll.urGetProgramExpProcAddrTable(version, byref(ProgramExp)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.ProgramExp = ProgramExp

        # attach function interface to function address
        self.urProgramBuildExp = _urProgramBuildExp_t(self.__dditable.ProgramExp.pfnBuildExp)
        self.urProgramCompileExp = _urProgramCompileExp_t(self.__dditable.ProgramExp.pfnCompileExp)
        self.urProgramLinkExp = _urProgramLinkExp_t(self.__dditable.ProgramExp.pfnLinkExp)

        # call driver to get function pointers
        Kernel = ur_kernel_dditable_t()
        r = ur_result_v(self.__dll.urGetKernelProcAddrTable(version, byref(Kernel)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Kernel = Kernel

        # attach function interface to function address
        self.urKernelCreate = _urKernelCreate_t(self.__dditable.Kernel.pfnCreate)
        self.urKernelGetInfo = _urKernelGetInfo_t(self.__dditable.Kernel.pfnGetInfo)
        self.urKernelGetGroupInfo = _urKernelGetGroupInfo_t(self.__dditable.Kernel.pfnGetGroupInfo)
        self.urKernelGetSubGroupInfo = _urKernelGetSubGroupInfo_t(self.__dditable.Kernel.pfnGetSubGroupInfo)
        self.urKernelRetain = _urKernelRetain_t(self.__dditable.Kernel.pfnRetain)
        self.urKernelRelease = _urKernelRelease_t(self.__dditable.Kernel.pfnRelease)
        self.urKernelGetNativeHandle = _urKernelGetNativeHandle_t(self.__dditable.Kernel.pfnGetNativeHandle)
        self.urKernelCreateWithNativeHandle = _urKernelCreateWithNativeHandle_t(self.__dditable.Kernel.pfnCreateWithNativeHandle)
        self.urKernelSetArgValue = _urKernelSetArgValue_t(self.__dditable.Kernel.pfnSetArgValue)
        self.urKernelSetArgLocal = _urKernelSetArgLocal_t(self.__dditable.Kernel.pfnSetArgLocal)
        self.urKernelSetArgPointer = _urKernelSetArgPointer_t(self.__dditable.Kernel.pfnSetArgPointer)
        self.urKernelSetExecInfo = _urKernelSetExecInfo_t(self.__dditable.Kernel.pfnSetExecInfo)
        self.urKernelSetArgSampler = _urKernelSetArgSampler_t(self.__dditable.Kernel.pfnSetArgSampler)
        self.urKernelSetArgMemObj = _urKernelSetArgMemObj_t(self.__dditable.Kernel.pfnSetArgMemObj)
        self.urKernelSetSpecializationConstants = _urKernelSetSpecializationConstants_t(self.__dditable.Kernel.pfnSetSpecializationConstants)

        # call driver to get function pointers
        KernelExp = ur_kernel_exp_dditable_t()
        r = ur_result_v(self.__dll.urGetKernelExpProcAddrTable(version, byref(KernelExp)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.KernelExp = KernelExp

        # attach function interface to function address
        self.urKernelSuggestMaxCooperativeGroupCountExp = _urKernelSuggestMaxCooperativeGroupCountExp_t(self.__dditable.KernelExp.pfnSuggestMaxCooperativeGroupCountExp)

        # call driver to get function pointers
        Sampler = ur_sampler_dditable_t()
        r = ur_result_v(self.__dll.urGetSamplerProcAddrTable(version, byref(Sampler)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Sampler = Sampler

        # attach function interface to function address
        self.urSamplerCreate = _urSamplerCreate_t(self.__dditable.Sampler.pfnCreate)
        self.urSamplerRetain = _urSamplerRetain_t(self.__dditable.Sampler.pfnRetain)
        self.urSamplerRelease = _urSamplerRelease_t(self.__dditable.Sampler.pfnRelease)
        self.urSamplerGetInfo = _urSamplerGetInfo_t(self.__dditable.Sampler.pfnGetInfo)
        self.urSamplerGetNativeHandle = _urSamplerGetNativeHandle_t(self.__dditable.Sampler.pfnGetNativeHandle)
        self.urSamplerCreateWithNativeHandle = _urSamplerCreateWithNativeHandle_t(self.__dditable.Sampler.pfnCreateWithNativeHandle)

        # call driver to get function pointers
        Mem = ur_mem_dditable_t()
        r = ur_result_v(self.__dll.urGetMemProcAddrTable(version, byref(Mem)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Mem = Mem

        # attach function interface to function address
        self.urMemImageCreate = _urMemImageCreate_t(self.__dditable.Mem.pfnImageCreate)
        self.urMemBufferCreate = _urMemBufferCreate_t(self.__dditable.Mem.pfnBufferCreate)
        self.urMemRetain = _urMemRetain_t(self.__dditable.Mem.pfnRetain)
        self.urMemRelease = _urMemRelease_t(self.__dditable.Mem.pfnRelease)
        self.urMemBufferPartition = _urMemBufferPartition_t(self.__dditable.Mem.pfnBufferPartition)
        self.urMemGetNativeHandle = _urMemGetNativeHandle_t(self.__dditable.Mem.pfnGetNativeHandle)
        self.urMemBufferCreateWithNativeHandle = _urMemBufferCreateWithNativeHandle_t(self.__dditable.Mem.pfnBufferCreateWithNativeHandle)
        self.urMemImageCreateWithNativeHandle = _urMemImageCreateWithNativeHandle_t(self.__dditable.Mem.pfnImageCreateWithNativeHandle)
        self.urMemGetInfo = _urMemGetInfo_t(self.__dditable.Mem.pfnGetInfo)
        self.urMemImageGetInfo = _urMemImageGetInfo_t(self.__dditable.Mem.pfnImageGetInfo)

        # call driver to get function pointers
        PhysicalMem = ur_physical_mem_dditable_t()
        r = ur_result_v(self.__dll.urGetPhysicalMemProcAddrTable(version, byref(PhysicalMem)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.PhysicalMem = PhysicalMem

        # attach function interface to function address
        self.urPhysicalMemCreate = _urPhysicalMemCreate_t(self.__dditable.PhysicalMem.pfnCreate)
        self.urPhysicalMemRetain = _urPhysicalMemRetain_t(self.__dditable.PhysicalMem.pfnRetain)
        self.urPhysicalMemRelease = _urPhysicalMemRelease_t(self.__dditable.PhysicalMem.pfnRelease)

        # call driver to get function pointers
        Global = ur_global_dditable_t()
        r = ur_result_v(self.__dll.urGetGlobalProcAddrTable(version, byref(Global)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Global = Global

        # attach function interface to function address
        self.urAdapterGet = _urAdapterGet_t(self.__dditable.Global.pfnAdapterGet)
        self.urAdapterRelease = _urAdapterRelease_t(self.__dditable.Global.pfnAdapterRelease)
        self.urAdapterRetain = _urAdapterRetain_t(self.__dditable.Global.pfnAdapterRetain)
        self.urAdapterGetLastError = _urAdapterGetLastError_t(self.__dditable.Global.pfnAdapterGetLastError)
        self.urAdapterGetInfo = _urAdapterGetInfo_t(self.__dditable.Global.pfnAdapterGetInfo)

        # call driver to get function pointers
        Enqueue = ur_enqueue_dditable_t()
        r = ur_result_v(self.__dll.urGetEnqueueProcAddrTable(version, byref(Enqueue)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Enqueue = Enqueue

        # attach function interface to function address
        self.urEnqueueKernelLaunch = _urEnqueueKernelLaunch_t(self.__dditable.Enqueue.pfnKernelLaunch)
        self.urEnqueueEventsWait = _urEnqueueEventsWait_t(self.__dditable.Enqueue.pfnEventsWait)
        self.urEnqueueEventsWaitWithBarrier = _urEnqueueEventsWaitWithBarrier_t(self.__dditable.Enqueue.pfnEventsWaitWithBarrier)
        self.urEnqueueMemBufferRead = _urEnqueueMemBufferRead_t(self.__dditable.Enqueue.pfnMemBufferRead)
        self.urEnqueueMemBufferWrite = _urEnqueueMemBufferWrite_t(self.__dditable.Enqueue.pfnMemBufferWrite)
        self.urEnqueueMemBufferReadRect = _urEnqueueMemBufferReadRect_t(self.__dditable.Enqueue.pfnMemBufferReadRect)
        self.urEnqueueMemBufferWriteRect = _urEnqueueMemBufferWriteRect_t(self.__dditable.Enqueue.pfnMemBufferWriteRect)
        self.urEnqueueMemBufferCopy = _urEnqueueMemBufferCopy_t(self.__dditable.Enqueue.pfnMemBufferCopy)
        self.urEnqueueMemBufferCopyRect = _urEnqueueMemBufferCopyRect_t(self.__dditable.Enqueue.pfnMemBufferCopyRect)
        self.urEnqueueMemBufferFill = _urEnqueueMemBufferFill_t(self.__dditable.Enqueue.pfnMemBufferFill)
        self.urEnqueueMemImageRead = _urEnqueueMemImageRead_t(self.__dditable.Enqueue.pfnMemImageRead)
        self.urEnqueueMemImageWrite = _urEnqueueMemImageWrite_t(self.__dditable.Enqueue.pfnMemImageWrite)
        self.urEnqueueMemImageCopy = _urEnqueueMemImageCopy_t(self.__dditable.Enqueue.pfnMemImageCopy)
        self.urEnqueueMemBufferMap = _urEnqueueMemBufferMap_t(self.__dditable.Enqueue.pfnMemBufferMap)
        self.urEnqueueMemUnmap = _urEnqueueMemUnmap_t(self.__dditable.Enqueue.pfnMemUnmap)
        self.urEnqueueUSMFill = _urEnqueueUSMFill_t(self.__dditable.Enqueue.pfnUSMFill)
        self.urEnqueueUSMMemcpy = _urEnqueueUSMMemcpy_t(self.__dditable.Enqueue.pfnUSMMemcpy)
        self.urEnqueueUSMPrefetch = _urEnqueueUSMPrefetch_t(self.__dditable.Enqueue.pfnUSMPrefetch)
        self.urEnqueueUSMAdvise = _urEnqueueUSMAdvise_t(self.__dditable.Enqueue.pfnUSMAdvise)
        self.urEnqueueUSMFill2D = _urEnqueueUSMFill2D_t(self.__dditable.Enqueue.pfnUSMFill2D)
        self.urEnqueueUSMMemcpy2D = _urEnqueueUSMMemcpy2D_t(self.__dditable.Enqueue.pfnUSMMemcpy2D)
        self.urEnqueueDeviceGlobalVariableWrite = _urEnqueueDeviceGlobalVariableWrite_t(self.__dditable.Enqueue.pfnDeviceGlobalVariableWrite)
        self.urEnqueueDeviceGlobalVariableRead = _urEnqueueDeviceGlobalVariableRead_t(self.__dditable.Enqueue.pfnDeviceGlobalVariableRead)
        self.urEnqueueReadHostPipe = _urEnqueueReadHostPipe_t(self.__dditable.Enqueue.pfnReadHostPipe)
        self.urEnqueueWriteHostPipe = _urEnqueueWriteHostPipe_t(self.__dditable.Enqueue.pfnWriteHostPipe)

        # call driver to get function pointers
        EnqueueExp = ur_enqueue_exp_dditable_t()
        r = ur_result_v(self.__dll.urGetEnqueueExpProcAddrTable(version, byref(EnqueueExp)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.EnqueueExp = EnqueueExp

        # attach function interface to function address
        self.urEnqueueCooperativeKernelLaunchExp = _urEnqueueCooperativeKernelLaunchExp_t(self.__dditable.EnqueueExp.pfnCooperativeKernelLaunchExp)

        # call driver to get function pointers
        Queue = ur_queue_dditable_t()
        r = ur_result_v(self.__dll.urGetQueueProcAddrTable(version, byref(Queue)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Queue = Queue

        # attach function interface to function address
        self.urQueueGetInfo = _urQueueGetInfo_t(self.__dditable.Queue.pfnGetInfo)
        self.urQueueCreate = _urQueueCreate_t(self.__dditable.Queue.pfnCreate)
        self.urQueueRetain = _urQueueRetain_t(self.__dditable.Queue.pfnRetain)
        self.urQueueRelease = _urQueueRelease_t(self.__dditable.Queue.pfnRelease)
        self.urQueueGetNativeHandle = _urQueueGetNativeHandle_t(self.__dditable.Queue.pfnGetNativeHandle)
        self.urQueueCreateWithNativeHandle = _urQueueCreateWithNativeHandle_t(self.__dditable.Queue.pfnCreateWithNativeHandle)
        self.urQueueFinish = _urQueueFinish_t(self.__dditable.Queue.pfnFinish)
        self.urQueueFlush = _urQueueFlush_t(self.__dditable.Queue.pfnFlush)

        # call driver to get function pointers
        BindlessImagesExp = ur_bindless_images_exp_dditable_t()
        r = ur_result_v(self.__dll.urGetBindlessImagesExpProcAddrTable(version, byref(BindlessImagesExp)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.BindlessImagesExp = BindlessImagesExp

        # attach function interface to function address
        self.urBindlessImagesUnsampledImageHandleDestroyExp = _urBindlessImagesUnsampledImageHandleDestroyExp_t(self.__dditable.BindlessImagesExp.pfnUnsampledImageHandleDestroyExp)
        self.urBindlessImagesSampledImageHandleDestroyExp = _urBindlessImagesSampledImageHandleDestroyExp_t(self.__dditable.BindlessImagesExp.pfnSampledImageHandleDestroyExp)
        self.urBindlessImagesImageAllocateExp = _urBindlessImagesImageAllocateExp_t(self.__dditable.BindlessImagesExp.pfnImageAllocateExp)
        self.urBindlessImagesImageFreeExp = _urBindlessImagesImageFreeExp_t(self.__dditable.BindlessImagesExp.pfnImageFreeExp)
        self.urBindlessImagesUnsampledImageCreateExp = _urBindlessImagesUnsampledImageCreateExp_t(self.__dditable.BindlessImagesExp.pfnUnsampledImageCreateExp)
        self.urBindlessImagesSampledImageCreateExp = _urBindlessImagesSampledImageCreateExp_t(self.__dditable.BindlessImagesExp.pfnSampledImageCreateExp)
        self.urBindlessImagesImageCopyExp = _urBindlessImagesImageCopyExp_t(self.__dditable.BindlessImagesExp.pfnImageCopyExp)
        self.urBindlessImagesImageGetInfoExp = _urBindlessImagesImageGetInfoExp_t(self.__dditable.BindlessImagesExp.pfnImageGetInfoExp)
        self.urBindlessImagesMipmapGetLevelExp = _urBindlessImagesMipmapGetLevelExp_t(self.__dditable.BindlessImagesExp.pfnMipmapGetLevelExp)
        self.urBindlessImagesMipmapFreeExp = _urBindlessImagesMipmapFreeExp_t(self.__dditable.BindlessImagesExp.pfnMipmapFreeExp)
        self.urBindlessImagesImportOpaqueFDExp = _urBindlessImagesImportOpaqueFDExp_t(self.__dditable.BindlessImagesExp.pfnImportOpaqueFDExp)
        self.urBindlessImagesMapExternalArrayExp = _urBindlessImagesMapExternalArrayExp_t(self.__dditable.BindlessImagesExp.pfnMapExternalArrayExp)
        self.urBindlessImagesReleaseInteropExp = _urBindlessImagesReleaseInteropExp_t(self.__dditable.BindlessImagesExp.pfnReleaseInteropExp)
        self.urBindlessImagesImportExternalSemaphoreOpaqueFDExp = _urBindlessImagesImportExternalSemaphoreOpaqueFDExp_t(self.__dditable.BindlessImagesExp.pfnImportExternalSemaphoreOpaqueFDExp)
        self.urBindlessImagesDestroyExternalSemaphoreExp = _urBindlessImagesDestroyExternalSemaphoreExp_t(self.__dditable.BindlessImagesExp.pfnDestroyExternalSemaphoreExp)
        self.urBindlessImagesWaitExternalSemaphoreExp = _urBindlessImagesWaitExternalSemaphoreExp_t(self.__dditable.BindlessImagesExp.pfnWaitExternalSemaphoreExp)
        self.urBindlessImagesSignalExternalSemaphoreExp = _urBindlessImagesSignalExternalSemaphoreExp_t(self.__dditable.BindlessImagesExp.pfnSignalExternalSemaphoreExp)

        # call driver to get function pointers
        USM = ur_usm_dditable_t()
        r = ur_result_v(self.__dll.urGetUSMProcAddrTable(version, byref(USM)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.USM = USM

        # attach function interface to function address
        self.urUSMHostAlloc = _urUSMHostAlloc_t(self.__dditable.USM.pfnHostAlloc)
        self.urUSMDeviceAlloc = _urUSMDeviceAlloc_t(self.__dditable.USM.pfnDeviceAlloc)
        self.urUSMSharedAlloc = _urUSMSharedAlloc_t(self.__dditable.USM.pfnSharedAlloc)
        self.urUSMFree = _urUSMFree_t(self.__dditable.USM.pfnFree)
        self.urUSMGetMemAllocInfo = _urUSMGetMemAllocInfo_t(self.__dditable.USM.pfnGetMemAllocInfo)
        self.urUSMPoolCreate = _urUSMPoolCreate_t(self.__dditable.USM.pfnPoolCreate)
        self.urUSMPoolRetain = _urUSMPoolRetain_t(self.__dditable.USM.pfnPoolRetain)
        self.urUSMPoolRelease = _urUSMPoolRelease_t(self.__dditable.USM.pfnPoolRelease)
        self.urUSMPoolGetInfo = _urUSMPoolGetInfo_t(self.__dditable.USM.pfnPoolGetInfo)

        # call driver to get function pointers
        USMExp = ur_usm_exp_dditable_t()
        r = ur_result_v(self.__dll.urGetUSMExpProcAddrTable(version, byref(USMExp)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.USMExp = USMExp

        # attach function interface to function address
        self.urUSMPitchedAllocExp = _urUSMPitchedAllocExp_t(self.__dditable.USMExp.pfnPitchedAllocExp)
        self.urUSMImportExp = _urUSMImportExp_t(self.__dditable.USMExp.pfnImportExp)
        self.urUSMReleaseExp = _urUSMReleaseExp_t(self.__dditable.USMExp.pfnReleaseExp)

        # call driver to get function pointers
        CommandBufferExp = ur_command_buffer_exp_dditable_t()
        r = ur_result_v(self.__dll.urGetCommandBufferExpProcAddrTable(version, byref(CommandBufferExp)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.CommandBufferExp = CommandBufferExp

        # attach function interface to function address
        self.urCommandBufferCreateExp = _urCommandBufferCreateExp_t(self.__dditable.CommandBufferExp.pfnCreateExp)
        self.urCommandBufferRetainExp = _urCommandBufferRetainExp_t(self.__dditable.CommandBufferExp.pfnRetainExp)
        self.urCommandBufferReleaseExp = _urCommandBufferReleaseExp_t(self.__dditable.CommandBufferExp.pfnReleaseExp)
        self.urCommandBufferFinalizeExp = _urCommandBufferFinalizeExp_t(self.__dditable.CommandBufferExp.pfnFinalizeExp)
        self.urCommandBufferAppendKernelLaunchExp = _urCommandBufferAppendKernelLaunchExp_t(self.__dditable.CommandBufferExp.pfnAppendKernelLaunchExp)
        self.urCommandBufferAppendUSMMemcpyExp = _urCommandBufferAppendUSMMemcpyExp_t(self.__dditable.CommandBufferExp.pfnAppendUSMMemcpyExp)
        self.urCommandBufferAppendUSMFillExp = _urCommandBufferAppendUSMFillExp_t(self.__dditable.CommandBufferExp.pfnAppendUSMFillExp)
        self.urCommandBufferAppendMemBufferCopyExp = _urCommandBufferAppendMemBufferCopyExp_t(self.__dditable.CommandBufferExp.pfnAppendMemBufferCopyExp)
        self.urCommandBufferAppendMemBufferWriteExp = _urCommandBufferAppendMemBufferWriteExp_t(self.__dditable.CommandBufferExp.pfnAppendMemBufferWriteExp)
        self.urCommandBufferAppendMemBufferReadExp = _urCommandBufferAppendMemBufferReadExp_t(self.__dditable.CommandBufferExp.pfnAppendMemBufferReadExp)
        self.urCommandBufferAppendMemBufferCopyRectExp = _urCommandBufferAppendMemBufferCopyRectExp_t(self.__dditable.CommandBufferExp.pfnAppendMemBufferCopyRectExp)
        self.urCommandBufferAppendMemBufferWriteRectExp = _urCommandBufferAppendMemBufferWriteRectExp_t(self.__dditable.CommandBufferExp.pfnAppendMemBufferWriteRectExp)
        self.urCommandBufferAppendMemBufferReadRectExp = _urCommandBufferAppendMemBufferReadRectExp_t(self.__dditable.CommandBufferExp.pfnAppendMemBufferReadRectExp)
        self.urCommandBufferAppendMemBufferFillExp = _urCommandBufferAppendMemBufferFillExp_t(self.__dditable.CommandBufferExp.pfnAppendMemBufferFillExp)
        self.urCommandBufferAppendUSMPrefetchExp = _urCommandBufferAppendUSMPrefetchExp_t(self.__dditable.CommandBufferExp.pfnAppendUSMPrefetchExp)
        self.urCommandBufferAppendUSMAdviseExp = _urCommandBufferAppendUSMAdviseExp_t(self.__dditable.CommandBufferExp.pfnAppendUSMAdviseExp)
        self.urCommandBufferEnqueueExp = _urCommandBufferEnqueueExp_t(self.__dditable.CommandBufferExp.pfnEnqueueExp)

        # call driver to get function pointers
        UsmP2PExp = ur_usm_p2p_exp_dditable_t()
        r = ur_result_v(self.__dll.urGetUsmP2PExpProcAddrTable(version, byref(UsmP2PExp)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.UsmP2PExp = UsmP2PExp

        # attach function interface to function address
        self.urUsmP2PEnablePeerAccessExp = _urUsmP2PEnablePeerAccessExp_t(self.__dditable.UsmP2PExp.pfnEnablePeerAccessExp)
        self.urUsmP2PDisablePeerAccessExp = _urUsmP2PDisablePeerAccessExp_t(self.__dditable.UsmP2PExp.pfnDisablePeerAccessExp)
        self.urUsmP2PPeerAccessGetInfoExp = _urUsmP2PPeerAccessGetInfoExp_t(self.__dditable.UsmP2PExp.pfnPeerAccessGetInfoExp)

        # call driver to get function pointers
        VirtualMem = ur_virtual_mem_dditable_t()
        r = ur_result_v(self.__dll.urGetVirtualMemProcAddrTable(version, byref(VirtualMem)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.VirtualMem = VirtualMem

        # attach function interface to function address
        self.urVirtualMemGranularityGetInfo = _urVirtualMemGranularityGetInfo_t(self.__dditable.VirtualMem.pfnGranularityGetInfo)
        self.urVirtualMemReserve = _urVirtualMemReserve_t(self.__dditable.VirtualMem.pfnReserve)
        self.urVirtualMemFree = _urVirtualMemFree_t(self.__dditable.VirtualMem.pfnFree)
        self.urVirtualMemMap = _urVirtualMemMap_t(self.__dditable.VirtualMem.pfnMap)
        self.urVirtualMemUnmap = _urVirtualMemUnmap_t(self.__dditable.VirtualMem.pfnUnmap)
        self.urVirtualMemSetAccess = _urVirtualMemSetAccess_t(self.__dditable.VirtualMem.pfnSetAccess)
        self.urVirtualMemGetInfo = _urVirtualMemGetInfo_t(self.__dditable.VirtualMem.pfnGetInfo)

        # call driver to get function pointers
        Device = ur_device_dditable_t()
        r = ur_result_v(self.__dll.urGetDeviceProcAddrTable(version, byref(Device)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Device = Device

        # attach function interface to function address
        self.urDeviceGet = _urDeviceGet_t(self.__dditable.Device.pfnGet)
        self.urDeviceGetInfo = _urDeviceGetInfo_t(self.__dditable.Device.pfnGetInfo)
        self.urDeviceRetain = _urDeviceRetain_t(self.__dditable.Device.pfnRetain)
        self.urDeviceRelease = _urDeviceRelease_t(self.__dditable.Device.pfnRelease)
        self.urDevicePartition = _urDevicePartition_t(self.__dditable.Device.pfnPartition)
        self.urDeviceSelectBinary = _urDeviceSelectBinary_t(self.__dditable.Device.pfnSelectBinary)
        self.urDeviceGetNativeHandle = _urDeviceGetNativeHandle_t(self.__dditable.Device.pfnGetNativeHandle)
        self.urDeviceCreateWithNativeHandle = _urDeviceCreateWithNativeHandle_t(self.__dditable.Device.pfnCreateWithNativeHandle)
        self.urDeviceGetGlobalTimestamps = _urDeviceGetGlobalTimestamps_t(self.__dditable.Device.pfnGetGlobalTimestamps)

        # success!
