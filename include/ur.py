"""
 Copyright (C) 2022 Intel Corporation

 SPDX-License-Identifier: MIT

 @file ur.py
 @version v0.5-r0.5

 """
import platform
from ctypes import *
from enum import *

###############################################################################
__version__ = "1.0"

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
## @brief Handle of Module object
class ur_module_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of module's Kernel object
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
## @brief Generic macro for enumerator bit masks
def UR_BIT( _i ):
    return ( 1 << _i )

###############################################################################
## @brief Defines Return/Error codes
class ur_result_v(IntEnum):
    SUCCESS = 0                                     ## Success
    INVALID_KERNEL_NAME = 1                         ## Invalid kernel name
    INVALID_OPERATION = 2                           ## Invalid operation
    INVALID_KERNEL = 3                              ## Invalid kernel
    INVALID_QUEUE_PROPERTIES = 4                    ## Invalid queue properties
    INVALID_VALUE = 5                               ## Invalid Value
    INVALID_CONTEXT = 6                             ## Invalid context
    INVALID_PLATFORM = 7                            ## Invalid platform
    INVALID_DEVICE = 8                              ## Invalid device
    INVALID_BINARY = 9                              ## Invalid binary
    INVALID_QUEUE = 10                              ## Invalid queue
    OUT_OF_HOST_MEMORY = 11                         ## Out of host memory
    INVALID_PROGRAM = 12                            ## Invalid program
    INVALID_PROGRAM_EXECUTABLE = 13                 ## Invalid program executable
    INVALID_SAMPLER = 14                            ## Invalid sampler
    INVALID_BUFFER_SIZE = 15                        ## Invalid buffer size
    INVALID_MEM_OBJECT = 16                         ## Invalid memory object
    OUT_OF_RESOURCES = 17                           ## Out of resources
    INVALID_EVENT = 18                              ## Invalid event
    INVALID_EVENT_WAIT_LIST = 19                    ## Invalid event wait list
    MISALIGNED_SUB_BUFFER_OFFSET = 20               ## Misaligned sub buffer offset
    BUILD_PROGRAM_FAILURE = 21                      ## Build program failure
    INVALID_WORK_GROUP_SIZE = 22                    ## Invalid work group size
    COMPILER_NOT_AVAILABLE = 23                     ## Compiler not available
    PROFILING_INFO_NOT_AVAILABLE = 24               ## Profiling info not available
    DEVICE_NOT_FOUND = 25                           ## Device not found
    INVALID_WORK_ITEM_SIZE = 26                     ## Invalid work item size
    INVALID_WORK_DIMENSION = 27                     ## Invalid work dimension
    INVALID_KERNEL_ARGS = 28                        ## Invalid kernel args
    INVALID_IMAGE_SIZE = 29                         ## Invalid image size
    INVALID_IMAGE_FORMAT_DESCRIPTOR = 30            ## Invalid image format descriptor
    IMAGE_FORMAT_NOT_SUPPORTED = 31                 ## Image format not supported
    MEM_OBJECT_ALLOCATION_FAILURE = 32              ## Memory object allocation failure
    ERROR_INVALID_PROGRAM_EXECUTABLE = 33           ## Program object parameter is invalid.
    ERROR_UNINITIALIZED = 0x78000001                ## [Validation] driver is not initialized
    ERROR_DEVICE_LOST = 0x78000002                  ## Device hung, reset, was removed, or driver update occurred
    ERROR_OUT_OF_HOST_MEMORY = 0x78000003           ## Insufficient host memory to satisfy call
    ERROR_OUT_OF_DEVICE_MEMORY = 0x78000004         ## Insufficient device memory to satisfy call
    ERROR_MODULE_BUILD_FAILURE = 0x78000005         ## Error occurred when building module, see build log for details
    ERROR_MODULE_LINK_FAILURE = 0x78000006          ## Error occurred when linking modules, see build log for details
    ERROR_DEVICE_REQUIRES_RESET = 0x78000007        ## Device requires a reset
    ERROR_DEVICE_IN_LOW_POWER_STATE = 0x78000008    ## Device currently in low power state
    ERROR_UNSUPPORTED_VERSION = 0x78000009          ## [Validation] generic error code for unsupported versions
    ERROR_UNSUPPORTED_FEATURE = 0x7800000a          ## [Validation] generic error code for unsupported features
    ERROR_INVALID_ARGUMENT = 0x7800000b             ## [Validation] generic error code for invalid arguments
    ERROR_INVALID_NULL_HANDLE = 0x7800000c          ## [Validation] handle argument is not valid
    ERROR_HANDLE_OBJECT_IN_USE = 0x7800000d         ## [Validation] object pointed to by handle still in-use by device
    ERROR_INVALID_NULL_POINTER = 0x7800000e         ## [Validation] pointer argument may not be nullptr
    ERROR_INVALID_SIZE = 0x7800000f                 ## [Validation] size argument is invalid (e.g., must not be zero)
    ERROR_UNSUPPORTED_SIZE = 0x78000010             ## [Validation] size argument is not supported by the device (e.g., too
                                                    ## large)
    ERROR_UNSUPPORTED_ALIGNMENT = 0x78000011        ## [Validation] alignment argument is not supported by the device (e.g.,
                                                    ## too small)
    ERROR_INVALID_SYNCHRONIZATION_OBJECT = 0x78000012   ## [Validation] synchronization object in invalid state
    ERROR_INVALID_ENUMERATION = 0x78000013          ## [Validation] enumerator argument is not valid
    ERROR_UNSUPPORTED_ENUMERATION = 0x78000014      ## [Validation] enumerator argument is not supported by the device
    ERROR_UNSUPPORTED_IMAGE_FORMAT = 0x78000015     ## [Validation] image format is not supported by the device
    ERROR_INVALID_NATIVE_BINARY = 0x78000016        ## [Validation] native binary is not supported by the device
    ERROR_INVALID_GLOBAL_NAME = 0x78000017          ## [Validation] global variable is not found in the module
    ERROR_INVALID_KERNEL_NAME = 0x78000018          ## [Validation] kernel name is not found in the module
    ERROR_INVALID_FUNCTION_NAME = 0x78000019        ## [Validation] function name is not found in the module
    ERROR_INVALID_GROUP_SIZE_DIMENSION = 0x7800001a ## [Validation] group size dimension is not valid for the kernel or
                                                    ## device
    ERROR_INVALID_GLOBAL_WIDTH_DIMENSION = 0x7800001b   ## [Validation] global width dimension is not valid for the kernel or
                                                    ## device
    ERROR_INVALID_KERNEL_ARGUMENT_INDEX = 0x7800001c## [Validation] kernel argument index is not valid for kernel
    ERROR_INVALID_KERNEL_ARGUMENT_SIZE = 0x7800001d ## [Validation] kernel argument size does not match kernel
    ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE = 0x7800001e   ## [Validation] value of kernel attribute is not valid for the kernel or
                                                    ## device
    ERROR_INVALID_MODULE_UNLINKED = 0x7800001f      ## [Validation] module with imports needs to be linked before kernels can
                                                    ## be created from it.
    ERROR_INVALID_COMMAND_LIST_TYPE = 0x78000020    ## [Validation] command list type does not match command queue type
    ERROR_OVERLAPPING_REGIONS = 0x78000021          ## [Validation] copy operations do not support overlapping regions of
                                                    ## memory
    INVALID_HOST_PTR = 0x78000022                   ## Invalid host pointer
    INVALID_USM_SIZE = 0x78000023                   ## Invalid USM size
    OBJECT_ALLOCATION_FAILURE = 0x78000024          ## Objection allocation failure
    ERROR_UNKNOWN = 0x7ffffffe                      ## Unknown or internal error

class ur_result_t(c_int):
    def __str__(self):
        return str(ur_result_v(self.value))


###############################################################################
## @brief Defines structure types
class ur_structure_type_v(IntEnum):
    IMAGE_DESC = 0                                  ## ::ur_image_desc_t

class ur_structure_type_t(c_int):
    def __str__(self):
        return str(ur_structure_type_v(self.value))


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
## @brief Supported context info
class ur_context_info_v(IntEnum):
    NUM_DEVICES = 1                                 ## [uint32_t] The number of the devices in the context
    DEVICES = 2                                     ## [::ur_context_handle_t...] The array of the device handles in the
                                                    ## context

class ur_context_info_t(c_int):
    def __str__(self):
        return str(ur_context_info_v(self.value))


###############################################################################
## @brief Map flags
class ur_map_flags_v(IntEnum):
    READ = UR_BIT(0)                                ## Map for read access
    WRITE = UR_BIT(1)                               ## Map for write access

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
## @brief USM memory advice
class ur_mem_advice_v(IntEnum):
    MEM_ADVICE_DEFAULT = 0                          ## The USM memory advice is default

class ur_mem_advice_t(c_int):
    def __str__(self):
        return str(ur_mem_advice_v(self.value))


###############################################################################
## @brief Event query information type
class ur_event_info_v(IntEnum):
    EVENT_INFO_COMMAND_QUEUE = 0                    ## Command queue information of an event object
    EVENT_INFO_CONTEXT = 1                          ## Context information of an event object
    EVENT_INFO_COMMAND_TYPE = 2                     ## Command type information of an event object
    EVENT_INFO_COMMAND_EXECUTION_STATUS = 3         ## Command execution status of an event object
    EVENT_INFO_REFERENCE_COUNT = 4                  ## Reference count of an event object

class ur_event_info_t(c_int):
    def __str__(self):
        return str(ur_event_info_v(self.value))


###############################################################################
## @brief Profiling query information type
class ur_profiling_info_v(IntEnum):
    PROFILING_INFO_COMMAND_QUEUED = 0               ## A 64-bit value of current device counter in nanoseconds when the event
                                                    ## is enqueued
    PROFILING_INFO_COMMAND_SUBMIT = 1               ## A 64-bit value of current device counter in nanoseconds when the event
                                                    ## is submitted
    PROFILING_INFO_COMMAND_START = 2                ## A 64-bit value of current device counter in nanoseconds when the event
                                                    ## starts execution
    PROFILING_INFO_COMMAND_END = 3                  ## A 64-bit value of current device counter in nanoseconds when the event
                                                    ## has finished execution

class ur_profiling_info_t(c_int):
    def __str__(self):
        return str(ur_profiling_info_v(self.value))


###############################################################################
## @brief Memory flags
class ur_mem_flags_v(IntEnum):
    MEM_READ_WRITE = UR_BIT(0)                      ## The memory object will be read and written by a kernel. This is the
                                                    ## default
    MEM_WRITE_ONLY = UR_BIT(1)                      ## The memory object will be written but not read by a kernel
    MEM_READ_ONLY = UR_BIT(2)                       ## The memory object is a read-only inside a kernel
    MEM_USE_HOST_POINTER = UR_BIT(3)                ## Use memory pointed by a host pointer parameter as the storage bits for
                                                    ## the memory object
    MEM_ALLOC_HOST_POINTER = UR_BIT(4)              ## Allocate memory object from host accessible memory
    MEM_ALLOC_COPY_HOST_POINTER = UR_BIT(5)         ## Allocate memory and copy the data from host pointer pointed memory

class ur_mem_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Memory types
class ur_mem_type_v(IntEnum):
    MEM_TYPE_BUFFER = 0                             ## Buffer object
    MEM_TYPE_IMAGE2D = 1                            ## 2D image object
    MEM_TYPE_IMAGE3D = 2                            ## 3D image object
    MEM_TYPE_IMAGE2D_ARRAY = 3                      ## 2D image array object
    MEM_TYPE_IMAGE1D = 4                            ## 1D image object
    MEM_TYPE_IMAGE1D_ARRAY = 5                      ## 1D image array object
    MEM_TYPE_IMAGE1D_BUFFER = 6                     ## 1D image buffer object

class ur_mem_type_t(c_int):
    def __str__(self):
        return str(ur_mem_type_v(self.value))


###############################################################################
## @brief Image channel order info: number of channels and the channel layout
class ur_image_channel_order_v(IntEnum):
    CHANNEL_ORDER_A = 0                             ## channel order A
    CHANNEL_ORDER_R = 1                             ## channel order R
    CHANNEL_ORDER_RG = 2                            ## channel order RG
    CHANNEL_ORDER_RA = 3                            ## channel order RA
    CHANNEL_ORDER_RGB = 4                           ## channel order RGB
    CHANNEL_ORDER_RGBA = 5                          ## channel order RGBA
    CHANNEL_ORDER_BGRA = 6                          ## channel order BGRA
    CHANNEL_ORDER_ARGB = 7                          ## channel order ARGB
    CHANNEL_ORDER_INTENSITY = 8                     ## channel order intensity
    CHANNEL_ORDER_LUMINANCE = 9                     ## channel order luminance
    CHANNEL_ORDER_RX = 10                           ## channel order Rx
    CHANNEL_ORDER_RGX = 11                          ## channel order RGx
    CHANNEL_ORDER_RGBX = 12                         ## channel order RGBx
    CHANNEL_ORDER_SRGBA = 13                        ## channel order sRGBA

class ur_image_channel_order_t(c_int):
    def __str__(self):
        return str(ur_image_channel_order_v(self.value))


###############################################################################
## @brief Image channel type info: describe the size of the channel data type
class ur_image_channel_type_v(IntEnum):
    CHANNEL_TYPE_SNORM_INT8 = 0                     ## channel type snorm int8
    CHANNEL_TYPE_SNORM_INT16 = 1                    ## channel type snorm int16
    CHANNEL_TYPE_UNORM_INT8 = 2                     ## channel type unorm int8
    CHANNEL_TYPE_UNORM_INT16 = 3                    ## channel type unorm int16
    CHANNEL_TYPE_UNORM_SHORT_565 = 4                ## channel type unorm short 565
    CHANNEL_TYPE_UNORM_SHORT_555 = 5                ## channel type unorm short 555
    CHANNEL_TYPE_INT_101010 = 6                     ## channel type int 101010
    CHANNEL_TYPE_SIGNED_INT8 = 7                    ## channel type signed int8
    CHANNEL_TYPE_SIGNED_INT16 = 8                   ## channel type signed int16
    CHANNEL_TYPE_SIGNED_INT32 = 9                   ## channel type signed int32
    CHANNEL_TYPE_UNSIGNED_INT8 = 10                 ## channel type unsigned int8
    CHANNEL_TYPE_UNSIGNED_INT16 = 11                ## channel type unsigned int16
    CHANNEL_TYPE_UNSIGNED_INT32 = 12                ## channel type unsigned int32
    CHANNEL_TYPE_HALF_FLOAT = 13                    ## channel type half float
    CHANNEL_TYPE_FLOAT = 14                         ## channel type float

class ur_image_channel_type_t(c_int):
    def __str__(self):
        return str(ur_image_channel_type_v(self.value))


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
        ("stype", ur_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("type", ur_mem_type_t),                                        ## [in] memory object type
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
## @brief Buffer region type, used to describe a sub buffer
class ur_buffer_region_t(Structure):
    _fields_ = [
        ("origin", c_size_t),                                           ## [in] buffer origin offset
        ("size", c_size_t)                                              ## [in] size of the buffer region
    ]

###############################################################################
## @brief Buffer creation type
class ur_buffer_create_type_v(IntEnum):
    BUFFER_CREATE_TYPE_REGION = 0                   ## buffer create type is region

class ur_buffer_create_type_t(c_int):
    def __str__(self):
        return str(ur_buffer_create_type_v(self.value))


###############################################################################
## @brief Query queue info
class ur_queue_info_v(IntEnum):
    CONTEXT = 0                                     ## Queue context info
    DEVICE = 1                                      ## Queue device info
    DEVICE_DEFAULT = 2                              ## Queue device default info
    PROPERTIES = 3                                  ## Queue properties info
    REFERENCE_COUNT = 4                             ## Queue reference count
    SIZE = 5                                        ## Queue size info

class ur_queue_info_t(c_int):
    def __str__(self):
        return str(ur_queue_info_v(self.value))


###############################################################################
## @brief Queue properties
class ur_queue_flags_v(IntEnum):
    OUT_OF_ORDER_EXEC_MODE_ENABLE = UR_BIT(0)       ## Enable/disable out of order execution
    PROFILING_ENABLE = UR_BIT(1)                    ## Enable/disable profiling
    ON_DEVICE = UR_BIT(2)                           ## Is a device queue
    ON_DEVICE_DEFAULT = UR_BIT(3)                   ## Is the default queue for a device

class ur_queue_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Get sample object information
class ur_sampler_info_v(IntEnum):
    SAMPLER_INFO_REFERENCE_COUNT = 0                ## Sampler reference count info
    SAMPLER_INFO_CONTEXT = 1                        ## Sampler context info
    SAMPLER_INFO_NORMALIZED_COORDS = 2              ## Sampler normalized coordindate setting
    SAMPLER_INFO_ADDRESSING_MODE = 3                ## Sampler addressing mode setting
    SAMPLER_INFO_FILTER_MODE = 4                    ## Sampler filter mode setting
    SAMPLER_INFO_MIP_FILTER_MODE = 5                ## Sampler MIP filter mode setting
    SAMPLER_INFO_LOD_MIN = 6                        ## Sampler LOD Min value
    SAMPLER_INFO_LOD_MAX = 7                        ## Sampler LOD Max value

class ur_sampler_info_t(c_int):
    def __str__(self):
        return str(ur_sampler_info_v(self.value))


###############################################################################
## @brief Sampler properties
class ur_sampler_properties_v(IntEnum):
    SAMPLER_PROPERTIES_NORMALIZED_COORDS = 0        ## Sampler normalized coordinates
    SAMPLER_PROPERTIES_ADDRESSING_MODE = 1          ## Sampler addressing mode
    SAMPLER_PROPERTIES_FILTER_MODE = 2              ## Sampler filter mode

class ur_sampler_properties_t(c_int):
    def __str__(self):
        return str(ur_sampler_properties_v(self.value))


###############################################################################
## @brief Sampler properties <name, value> pair
class ur_sampler_property_value_t(Structure):
    _fields_ = [
        ("propName", ur_sampler_properties_t),                          ## [in] Sampler property
        ("propValue", c_ulong)                                          ## [in] Sampler property value
    ]

###############################################################################
## @brief USM memory property flags
class ur_usm_mem_flags_v(IntEnum):
    MEM_ALLOC_FLAGS_INTEL = UR_BIT(0)               ## The USM memory allocation is from Intel USM

class ur_usm_mem_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief USM memory allocation information type
class ur_mem_info_v(IntEnum):
    MEM_ALLOC_TYPE = 0                              ## Memory allocation type info
    MEM_ALLOC_BASE_PTR = 1                          ## Memory allocation base pointer info
    MEM_ALLOC_SIZE = 2                              ## Memory allocation size info
    MEM_ALLOC_DEVICE = 3                            ## Memory allocation device info

class ur_mem_info_t(c_int):
    def __str__(self):
        return str(ur_mem_info_v(self.value))


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
    TYPE = 0                                        ## ::ur_device_type_t: type of the device
    VENDOR_ID = 1                                   ## uint32_t: vendor Id of the device
    MAX_COMPUTE_UNITS = 2                           ## uint32_t: the number of compute units
    MAX_WORK_ITEM_DIMENSIONS = 3                    ## uint32_t: max work item dimensions
    MAX_WORK_ITEM_SIZES = 4                         ## size_t[]: return an array of max work item sizes
    MAX_WORK_GROUP_SIZE = 5                         ## size_t: max work group size
    SINGLE_FP_CONFIG = 6                            ## Return a bit field of ::ur_fp_capability_flags_t: single precision
                                                    ## floating point capability
    HALF_FP_CONFIG = 7                              ## Return a bit field of ::ur_fp_capability_flags_t: half precsion
                                                    ## floating point capability
    DOUBLE_FP_CONFIG = 8                            ## Return a bit field of ::ur_fp_capability_flags_t: double precision
                                                    ## floating point capability
    QUEUE_PROPERTIES = 9                            ## Return a bit field of ::ur_queue_flags_t: command queue properties
                                                    ## supported by the device
    PREFERRED_VECTOR_WIDTH_CHAR = 10                ## uint32_t: preferred vector width for char
    PREFERRED_VECTOR_WIDTH_SHORT = 11               ## uint32_t: preferred vector width for short
    PREFERRED_VECTOR_WIDTH_INT = 12                 ## uint32_t: preferred vector width for int
    PREFERRED_VECTOR_WIDTH_LONG = 13                ## uint32_t: preferred vector width for long
    PREFERRED_VECTOR_WIDTH_FLOAT = 14               ## uint32_t: preferred vector width for float
    PREFERRED_VECTOR_WIDTH_DOUBLE = 15              ## uint32_t: preferred vector width for double
    PREFERRED_VECTOR_WIDTH_HALF = 16                ## uint32_t: preferred vector width for half float
    NATIVE_VECTOR_WIDTH_CHAR = 17                   ## uint32_t: native vector width for char
    NATIVE_VECTOR_WIDTH_SHORT = 18                  ## uint32_t: native vector width for short
    NATIVE_VECTOR_WIDTH_INT = 19                    ## uint32_t: native vector width for int
    NATIVE_VECTOR_WIDTH_LONG = 20                   ## uint32_t: native vector width for long
    NATIVE_VECTOR_WIDTH_FLOAT = 21                  ## uint32_t: native vector width for float
    NATIVE_VECTOR_WIDTH_DOUBLE = 22                 ## uint32_t: native vector width for double
    NATIVE_VECTOR_WIDTH_HALF = 23                   ## uint32_t: native vector width for half float
    MAX_CLOCK_FREQUENCY = 24                        ## uint32_t: max clock frequency in MHz
    ADDRESS_BITS = 25                               ## uint32_t: address bits
    MAX_MEM_ALLOC_SIZE = 26                         ## uint64_t: max memory allocation size
    IMAGE_SUPPORTED = 27                            ## bool: images are supported
    MAX_READ_IMAGE_ARGS = 28                        ## uint32_t: max number of image objects arguments of a kernel declared
                                                    ## with the read_only qualifier
    MAX_WRITE_IMAGE_ARGS = 29                       ## uint32_t: max number of image objects arguments of a kernel declared
                                                    ## with the write_only qualifier
    MAX_READ_WRITE_IMAGE_ARGS = 30                  ## uint32_t: max number of image objects arguments of a kernel declared
                                                    ## with the read_write qualifier
    IMAGE2D_MAX_WIDTH = 31                          ## size_t: max width of Image2D object
    IMAGE2D_MAX_HEIGHT = 32                         ## size_t: max heigh of Image2D object
    IMAGE3D_MAX_WIDTH = 33                          ## size_t: max width of Image3D object
    IMAGE3D_MAX_HEIGHT = 34                         ## size_t: max height of Image3D object
    IMAGE3D_MAX_DEPTH = 35                          ## size_t: max depth of Image3D object
    IMAGE_MAX_BUFFER_SIZE = 36                      ## size_t: max image buffer size
    IMAGE_MAX_ARRAR_SIZE = 37                       ## size_t: max image array size
    MAX_SAMPLERS = 38                               ## uint32_t: max number of samplers that can be used in a kernel
    MAX_PARAMETER_SIZE = 39                         ## size_t: max size in bytes of all arguments passed to a kernel
    MEM_BASE_ADDR_ALIGN = 40                        ## uint32_t: memory base address alignment
    GLOBAL_MEM_CACHE_TYPE = 41                      ## ::ur_device_mem_cache_type_t: global memory cache type
    GLOBAL_MEM_CACHELINE_SIZE = 42                  ## uint32_t: global memory cache line size in bytes
    GLOBAL_MEM_CACHE_SIZE = 43                      ## uint64_t: size of global memory cache in bytes
    GLOBAL_MEM_SIZE = 44                            ## uint64_t: size of global memory in bytes
    MAX_CONSTANT_BUFFER_SIZE = 45                   ## uint64_t: max constant buffer size in bytes
    MAX_CONSTANT_ARGS = 46                          ## uint32_t: max number of __const declared arguments in a kernel
    LOCAL_MEM_TYPE = 47                             ## ::ur_device_local_mem_type_t: local memory type
    LOCAL_MEM_SIZE = 48                             ## uint64_t: local memory size in bytes
    ERROR_CORRECTION_SUPPORT = 49                   ## bool: support error correction to gloal and local memory
    HOST_UNIFIED_MEMORY = 50                        ## bool: unifed host device memory
    PROFILING_TIMER_RESOLUTION = 51                 ## size_t: profiling timer resolution in nanoseconds
    ENDIAN_LITTLE = 52                              ## bool: little endian byte order
    AVAILABLE = 53                                  ## bool: device is available
    COMPILER_AVAILABLE = 54                         ## bool: device compiler is available
    LINKER_AVAILABLE = 55                           ## bool: device linker is available
    EXECUTION_CAPABILITIES = 56                     ## ::ur_device_exec_capability_flags_t: device kernel execution
                                                    ## capability bit-field
    QUEUE_ON_DEVICE_PROPERTIES = 57                 ## ::ur_queue_flags_t: device command queue property bit-field
    QUEUE_ON_HOST_PROPERTIES = 58                   ## ::ur_queue_flags_t: host queue property bit-field
    BUILT_IN_KERNELS = 59                           ## char[]: a semi-colon separated list of built-in kernels
    PLATFORM = 60                                   ## ::ur_platform_handle_t: the platform associated with the device
    REFERENCE_COUNT = 61                            ## uint32_t: reference count
    IL_VERSION = 62                                 ## char[]: IL version
    NAME = 63                                       ## char[]: Device name
    VENDOR = 64                                     ## char[]: Device vendor
    DRIVER_VERSION = 65                             ## char[]: Driver version
    PROFILE = 66                                    ## char[]: Device profile
    VERSION = 67                                    ## char[]: Device version
    EXTENSIONS = 68                                 ## char[]: Return a space separated list of extension names
    PRINTF_BUFFER_SIZE = 69                         ## size_t: Maximum size in bytes of internal printf buffer
    PREFERRED_INTEROP_USER_SYNC = 70                ## bool: prefer user synchronization when sharing object with other API
    PARENT_DEVICE = 71                              ## ::ur_device_handle_t: return parent device handle
    PARTITION_PROPERTIES = 72                       ## uint32_t: return a bit-field of partition properties
                                                    ## ::ur_device_partition_property_flags_t
    PARTITION_MAX_SUB_DEVICES = 73                  ## uint32_t: maximum number of sub-devices when the device is partitioned
    PARTITION_AFFINITY_DOMAIN = 74                  ## uint32_t: return a bit-field of affinity domain
                                                    ## ::ur_device_affinity_domain_flags_t
    PARTITION_TYPE = 75                             ## uint32_t: return a bit-field of ::ur_device_partition_property_flags_t
                                                    ## for properties specified in ::urDevicePartition
    MAX_NUM_SUB_GROUPS = 76                         ## uint32_t: max number of sub groups
    SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 77     ## bool: support sub group independent forward progress
    SUB_GROUP_SIZES_INTEL = 78                      ## uint32_t[]: return an array of sub group sizes supported on Intel
                                                    ## device
    USM_HOST_SUPPORT = 79                           ## bool: support USM host memory access
    USM_DEVICE_SUPPORT = 80                         ## bool: support USM device memory access
    USM_SINGLE_SHARED_SUPPORT = 81                  ## bool: support USM single device shared memory access
    USM_CROSS_SHARED_SUPPORT = 82                   ## bool: support USM cross device shared memory access
    USM_SYSTEM_SHARED_SUPPORT = 83                  ## bool: support USM system wide shared memory access
    UUID = 84                                       ## char[]: return device UUID
    PCI_ADDRESS = 85                                ## char[]: return device PCI address
    GPU_EU_COUNT = 86                               ## uint32_t: return Intel GPU EU count
    GPU_EU_SIMD_WIDTH = 87                          ## uint32_t: return Intel GPU EU SIMD width
    GPU_EU_SLICES = 88                              ## uint32_t: return Intel GPU number of slices
    GPU_SUBSLICES_PER_SLICE = 89                    ## uint32_t: return Intel GPU number of subslices per slice
    MAX_MEMORY_BANDWIDTH = 90                       ## uint32_t: return max memory bandwidth in Mb/s
    IMAGE_SRGB = 91                                 ## bool: image is SRGB
    ATOMIC_64 = 92                                  ## bool: support 64 bit atomics
    ATOMIC_MEMORY_ORDER_CAPABILITIES = 93           ## uint32_t: atomics memory order capabilities

class ur_device_info_t(c_int):
    def __str__(self):
        return str(ur_device_info_v(self.value))


###############################################################################
## @brief Device partition property
class ur_device_partition_property_flags_v(IntEnum):
    EQUALLY = UR_BIT(0)                             ## Support equal partition
    BY_COUNTS = UR_BIT(1)                           ## Support partition by count
    BY_AFFINITY_DOMAIN = UR_BIT(2)                  ## Support partition by affinity domain

class ur_device_partition_property_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Partition property value
class ur_device_partition_property_value_t(Structure):
    _fields_ = [
        ("property", ur_device_partition_property_flags_t),             ## [in] device partition property flags
        ("value", c_ulong)                                              ## [in] partition value
    ]

###############################################################################
## @brief FP capabilities
class ur_fp_capability_flags_v(IntEnum):
    CORRECTLY_ROUNDED_DIVIDE_SQRT = UR_BIT(0)       ## Support correctly rounded divide and sqrt
    ROUND_TO_NEAREST = UR_BIT(1)                    ## Support round to nearest
    ROUND_TO_ZERO = UR_BIT(2)                       ## Support round to zero
    ROUND_TO_INF = UR_BIT(3)                        ## Support round to infinity
    INF_NAN = UR_BIT(4)                             ## Support INF to NAN
    DENORM = UR_BIT(5)                              ## Support denorm
    FMA = UR_BIT(6)                                 ## Support FMA

class ur_fp_capability_flags_t(c_int):
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
    LOCAL = 0                                       ## Dedicated local memory
    GLOBAL = 1                                      ## Global memory

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
## @brief Device affinity domain
class ur_device_affinity_domain_flags_v(IntEnum):
    NUMA = UR_BIT(0)                                ## By NUMA
    NEXT_PARTITIONABLE = UR_BIT(1)                  ## BY next partitionable

class ur_device_affinity_domain_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Get Kernel object information
class ur_kernel_info_v(IntEnum):
    FUNCTION_NAME = 0                               ## Return Kernel function name, return type char[]
    NUM_ARGS = 1                                    ## Return Kernel number of arguments
    REFERENCE_COUNT = 2                             ## Return Kernel reference count
    CONTEXT = 3                                     ## Return Context object associated with Kernel
    PROGRAM = 4                                     ## Return Program object associated with Kernel
    ATTRIBUTES = 5                                  ## Return Kernel attributes, return type char[]

class ur_kernel_info_t(c_int):
    def __str__(self):
        return str(ur_kernel_info_v(self.value))


###############################################################################
## @brief Get Kernel Work Group information
class ur_kernel_group_info_v(IntEnum):
    GLOBAL_WORK_SIZE = 0                            ## Return Work Group maximum global size, return type size_t[3]
    WORK_GROUP_SIZE = 1                             ## Return maximum Work Group size, return type size_t
    COMPILE_WORK_GROUP_SIZE = 2                     ## Return Work Group size required by the source code, such as
                                                    ## __attribute__((required_work_group_size(X,Y,Z)), return type size_t[3]
    LOCAL_MEM_SIZE = 3                              ## Return local memory required by the Kernel, return type size_t
    PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 4          ## Return preferred multiple of Work Group size for launch, return type
                                                    ## size_t
    PRIVATE_MEM_SIZE = 5                            ## Return minimum amount of private memory in bytes used by each work
                                                    ## item in the Kernel, return type size_t

class ur_kernel_group_info_t(c_int):
    def __str__(self):
        return str(ur_kernel_group_info_v(self.value))


###############################################################################
## @brief Get Kernel SubGroup information
class ur_kernel_sub_group_info_v(IntEnum):
    MAX_SUB_GROUP_SIZE = 0                          ## Return maximum SubGroup size, return type uint32_t
    MAX_NUM_SUB_GROUPS = 1                          ## Return maximum number of SubGroup, return type uint32_t
    COMPILE_NUM_SUB_GROUPS = 2                      ## Return number of SubGroup required by the source code, return type
                                                    ## uint32_t
    SUB_GROUP_SIZE_INTEL = 3                        ## Return SubGroup size required by Intel, return type uint32_t

class ur_kernel_sub_group_info_t(c_int):
    def __str__(self):
        return str(ur_kernel_sub_group_info_v(self.value))


###############################################################################
## @brief Set additional Kernel execution information
class ur_kernel_exec_info_v(IntEnum):
    USM_INDIRECT_ACCESS = 0                         ## Kernel might access data through USM pointer, type bool_t*
    USM_PTRS = 1                                    ## Provide an explicit list of USM pointers that the kernel will access,
                                                    ## type void*[].

class ur_kernel_exec_info_t(c_int):
    def __str__(self):
        return str(ur_kernel_exec_info_v(self.value))


###############################################################################
## @brief Supported platform info
class ur_platform_info_v(IntEnum):
    NAME = 1                                        ## [char*] The string denoting name of the platform. The size of the info
                                                    ## needs to be dynamically queried.
    VENDOR_NAME = 2                                 ## [char*] The string denoting name of the vendor of the platform. The
                                                    ## size of the info needs to be dynamically queried.
    VERSION = 3                                     ## [char*] The string denoting the version of the platform. The size of
                                                    ## the info needs to be dynamically queried.
    EXTENSIONS = 4                                  ## [char*] The string denoting extensions supported by the platform. The
                                                    ## size of the info needs to be dynamically queried.
    PROFILE = 5                                     ## [char*] The string denoting profile of the platform. The size of the
                                                    ## info needs to be dynamically queried.

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
    _0_9 = UR_MAKE_VERSION( 0, 9 )                  ## version 0.9
    CURRENT = UR_MAKE_VERSION( 0, 9 )               ## latest known version

class ur_api_version_t(c_int):
    def __str__(self):
        return str(ur_api_version_v(self.value))


###############################################################################
## @brief Get Program object information
class ur_program_info_v(IntEnum):
    REFERENCE_COUNT = 0                             ## Program reference count info
    CONTEXT = 1                                     ## Program context info
    NUM_DEVICES = 2                                 ## Return number of devices associated with Program
    DEVICES = 3                                     ## Return list of devices associated with Program, return type
                                                    ## uint32_t[].
    SOURCE = 4                                      ## Return program source associated with Program, return type char[].
    BINARY_SIZES = 5                                ## Return program binary sizes for each device, return type size_t[].
    BINARIES = 6                                    ## Return program binaries for all devices for this Program, return type
                                                    ## uchar[].
    NUM_KERNELS = 7                                 ## Number of kernels in Program, return type size_t
    KERNEL_NAMES = 8                                ## Return a semi-colon separated list of kernel names in Program, return
                                                    ## type char[]

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
    STATUS = 0                                      ## Program build status, return type ::ur_program_build_status_t
    OPTIONS = 1                                     ## Program build options, return type char[]
    LOG = 2                                         ## Program build log, return type char[]
    BINARY_TYPE = 3                                 ## Program binary type, return type ::ur_program_binary_type_t

class ur_program_build_info_t(c_int):
    def __str__(self):
        return str(ur_program_build_info_v(self.value))


###############################################################################
## @brief Supported platform initialization flags
class ur_platform_init_flags_v(IntEnum):
    LEVEL_ZERO = UR_BIT(0)                          ## initialize Unified Runtime platform drivers

class ur_platform_init_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Supported device initialization flags
class ur_device_init_flags_v(IntEnum):
    GPU = UR_BIT(0)                                 ## initialize GPU device drivers

class ur_device_init_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
__use_win_types = "Windows" == platform.uname()[0]

###############################################################################
## @brief Function-pointer for urPlatformGet
if __use_win_types:
    _urPlatformGet_t = WINFUNCTYPE( ur_result_t, POINTER(c_ulong), POINTER(ur_platform_handle_t) )
else:
    _urPlatformGet_t = CFUNCTYPE( ur_result_t, POINTER(c_ulong), POINTER(ur_platform_handle_t) )

###############################################################################
## @brief Function-pointer for urPlatformGetInfo
if __use_win_types:
    _urPlatformGetInfo_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_platform_info_t, POINTER(c_size_t), c_void_p )
else:
    _urPlatformGetInfo_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_platform_info_t, POINTER(c_size_t), c_void_p )

###############################################################################
## @brief Function-pointer for urPlatformGetNativeHandle
if __use_win_types:
    _urPlatformGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, POINTER(ur_native_handle_t) )
else:
    _urPlatformGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urPlatformCreateWithNativeHandle
if __use_win_types:
    _urPlatformCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_platform_handle_t) )
else:
    _urPlatformCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_platform_handle_t) )

###############################################################################
## @brief Function-pointer for urPlatformGetApiVersion
if __use_win_types:
    _urPlatformGetApiVersion_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, POINTER(ur_api_version_t) )
else:
    _urPlatformGetApiVersion_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, POINTER(ur_api_version_t) )


###############################################################################
## @brief Table of Platform functions pointers
class _ur_platform_dditable_t(Structure):
    _fields_ = [
        ("pfnGet", c_void_p),                                           ## _urPlatformGet_t
        ("pfnGetInfo", c_void_p),                                       ## _urPlatformGetInfo_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urPlatformGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p),                        ## _urPlatformCreateWithNativeHandle_t
        ("pfnGetApiVersion", c_void_p)                                  ## _urPlatformGetApiVersion_t
    ]

###############################################################################
## @brief Function-pointer for urContextCreate
if __use_win_types:
    _urContextCreate_t = WINFUNCTYPE( ur_result_t, c_ulong, POINTER(ur_device_handle_t), POINTER(ur_context_handle_t) )
else:
    _urContextCreate_t = CFUNCTYPE( ur_result_t, c_ulong, POINTER(ur_device_handle_t), POINTER(ur_context_handle_t) )

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
    _urContextGetInfo_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_context_info_t, POINTER(c_size_t), c_void_p )
else:
    _urContextGetInfo_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_context_info_t, POINTER(c_size_t), c_void_p )

###############################################################################
## @brief Function-pointer for urContextGetNativeHandle
if __use_win_types:
    _urContextGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_native_handle_t) )
else:
    _urContextGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urContextCreateWithNativeHandle
if __use_win_types:
    _urContextCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_context_handle_t) )
else:
    _urContextCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_context_handle_t) )


###############################################################################
## @brief Table of Context functions pointers
class _ur_context_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _urContextCreate_t
        ("pfnRetain", c_void_p),                                        ## _urContextRetain_t
        ("pfnRelease", c_void_p),                                       ## _urContextRelease_t
        ("pfnGetInfo", c_void_p),                                       ## _urContextGetInfo_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urContextGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p)                         ## _urContextCreateWithNativeHandle_t
    ]

###############################################################################
## @brief Function-pointer for urEventCreate
if __use_win_types:
    _urEventCreate_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_event_handle_t) )
else:
    _urEventCreate_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEventGetInfo
if __use_win_types:
    _urEventGetInfo_t = WINFUNCTYPE( ur_result_t, ur_event_handle_t, ur_event_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urEventGetInfo_t = CFUNCTYPE( ur_result_t, ur_event_handle_t, ur_event_info_t, c_size_t, c_void_p, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for urEventGetProfilingInfo
if __use_win_types:
    _urEventGetProfilingInfo_t = WINFUNCTYPE( ur_result_t, ur_event_handle_t, ur_profiling_info_t, c_size_t, c_void_p, c_size_t )
else:
    _urEventGetProfilingInfo_t = CFUNCTYPE( ur_result_t, ur_event_handle_t, ur_profiling_info_t, c_size_t, c_void_p, c_size_t )

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
    _urEventCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_event_handle_t) )
else:
    _urEventCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_event_handle_t) )


###############################################################################
## @brief Table of Event functions pointers
class _ur_event_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _urEventCreate_t
        ("pfnGetInfo", c_void_p),                                       ## _urEventGetInfo_t
        ("pfnGetProfilingInfo", c_void_p),                              ## _urEventGetProfilingInfo_t
        ("pfnWait", c_void_p),                                          ## _urEventWait_t
        ("pfnRetain", c_void_p),                                        ## _urEventRetain_t
        ("pfnRelease", c_void_p),                                       ## _urEventRelease_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urEventGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p)                         ## _urEventCreateWithNativeHandle_t
    ]

###############################################################################
## @brief Function-pointer for urProgramCreate
if __use_win_types:
    _urProgramCreate_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_ulong, POINTER(ur_module_handle_t), c_char_p, POINTER(ur_program_handle_t) )
else:
    _urProgramCreate_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_ulong, POINTER(ur_module_handle_t), c_char_p, POINTER(ur_program_handle_t) )

###############################################################################
## @brief Function-pointer for urProgramCreateWithBinary
if __use_win_types:
    _urProgramCreateWithBinary_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, c_ulong, POINTER(c_ubyte), POINTER(ur_program_handle_t) )
else:
    _urProgramCreateWithBinary_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, c_ulong, POINTER(c_ubyte), POINTER(ur_program_handle_t) )

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
    _urProgramGetInfo_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t, ur_program_info_t, POINTER(c_size_t), c_void_p )
else:
    _urProgramGetInfo_t = CFUNCTYPE( ur_result_t, ur_program_handle_t, ur_program_info_t, POINTER(c_size_t), c_void_p )

###############################################################################
## @brief Function-pointer for urProgramGetBuildInfo
if __use_win_types:
    _urProgramGetBuildInfo_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t, ur_device_handle_t, ur_program_build_info_t, POINTER(c_size_t), c_void_p )
else:
    _urProgramGetBuildInfo_t = CFUNCTYPE( ur_result_t, ur_program_handle_t, ur_device_handle_t, ur_program_build_info_t, POINTER(c_size_t), c_void_p )

###############################################################################
## @brief Function-pointer for urProgramSetSpecializationConstant
if __use_win_types:
    _urProgramSetSpecializationConstant_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t, c_ulong, c_size_t, c_void_p )
else:
    _urProgramSetSpecializationConstant_t = CFUNCTYPE( ur_result_t, ur_program_handle_t, c_ulong, c_size_t, c_void_p )

###############################################################################
## @brief Function-pointer for urProgramGetNativeHandle
if __use_win_types:
    _urProgramGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t, POINTER(ur_native_handle_t) )
else:
    _urProgramGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_program_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urProgramCreateWithNativeHandle
if __use_win_types:
    _urProgramCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_program_handle_t, ur_native_handle_t, POINTER(ur_program_handle_t) )
else:
    _urProgramCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_program_handle_t, ur_native_handle_t, POINTER(ur_program_handle_t) )


###############################################################################
## @brief Table of Program functions pointers
class _ur_program_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _urProgramCreate_t
        ("pfnCreateWithBinary", c_void_p),                              ## _urProgramCreateWithBinary_t
        ("pfnRetain", c_void_p),                                        ## _urProgramRetain_t
        ("pfnRelease", c_void_p),                                       ## _urProgramRelease_t
        ("pfnGetFunctionPointer", c_void_p),                            ## _urProgramGetFunctionPointer_t
        ("pfnGetInfo", c_void_p),                                       ## _urProgramGetInfo_t
        ("pfnGetBuildInfo", c_void_p),                                  ## _urProgramGetBuildInfo_t
        ("pfnSetSpecializationConstant", c_void_p),                     ## _urProgramSetSpecializationConstant_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urProgramGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p)                         ## _urProgramCreateWithNativeHandle_t
    ]

###############################################################################
## @brief Function-pointer for urModuleCreate
if __use_win_types:
    _urModuleCreate_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_ulong, c_char_p, POINTER(c_void_p), c_void_p, POINTER(ur_module_handle_t) )
else:
    _urModuleCreate_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, c_ulong, c_char_p, POINTER(c_void_p), c_void_p, POINTER(ur_module_handle_t) )

###############################################################################
## @brief Function-pointer for urModuleRetain
if __use_win_types:
    _urModuleRetain_t = WINFUNCTYPE( ur_result_t, ur_module_handle_t )
else:
    _urModuleRetain_t = CFUNCTYPE( ur_result_t, ur_module_handle_t )

###############################################################################
## @brief Function-pointer for urModuleRelease
if __use_win_types:
    _urModuleRelease_t = WINFUNCTYPE( ur_result_t, ur_module_handle_t )
else:
    _urModuleRelease_t = CFUNCTYPE( ur_result_t, ur_module_handle_t )

###############################################################################
## @brief Function-pointer for urModuleGetNativeHandle
if __use_win_types:
    _urModuleGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_module_handle_t, POINTER(ur_native_handle_t) )
else:
    _urModuleGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_module_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urModuleCreateWithNativeHandle
if __use_win_types:
    _urModuleCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_module_handle_t) )
else:
    _urModuleCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_module_handle_t) )


###############################################################################
## @brief Table of Module functions pointers
class _ur_module_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _urModuleCreate_t
        ("pfnRetain", c_void_p),                                        ## _urModuleRetain_t
        ("pfnRelease", c_void_p),                                       ## _urModuleRelease_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urModuleGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p)                         ## _urModuleCreateWithNativeHandle_t
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
    _urKernelGetInfo_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_kernel_info_t, POINTER(c_size_t), c_void_p )
else:
    _urKernelGetInfo_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_kernel_info_t, POINTER(c_size_t), c_void_p )

###############################################################################
## @brief Function-pointer for urKernelGetGroupInfo
if __use_win_types:
    _urKernelGetGroupInfo_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_device_handle_t, ur_kernel_group_info_t, c_size_t, c_void_p )
else:
    _urKernelGetGroupInfo_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_device_handle_t, ur_kernel_group_info_t, c_size_t, c_void_p )

###############################################################################
## @brief Function-pointer for urKernelGetSubGroupInfo
if __use_win_types:
    _urKernelGetSubGroupInfo_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_device_handle_t, ur_kernel_sub_group_info_t, c_size_t, c_void_p )
else:
    _urKernelGetSubGroupInfo_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_device_handle_t, ur_kernel_sub_group_info_t, c_size_t, c_void_p )

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
    _urKernelCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_kernel_handle_t) )
else:
    _urKernelCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_kernel_handle_t) )

###############################################################################
## @brief Function-pointer for urKernelSetArg
if __use_win_types:
    _urKernelSetArg_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, c_size_t, c_void_p )
else:
    _urKernelSetArg_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, c_size_t, c_void_p )

###############################################################################
## @brief Function-pointer for urKernelSetArgPointer
if __use_win_types:
    _urKernelSetArgPointer_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, c_size_t, c_void_p )
else:
    _urKernelSetArgPointer_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, c_size_t, c_void_p )

###############################################################################
## @brief Function-pointer for urKernelSetExecInfo
if __use_win_types:
    _urKernelSetExecInfo_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_kernel_exec_info_t, c_size_t, c_void_p )
else:
    _urKernelSetExecInfo_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, ur_kernel_exec_info_t, c_size_t, c_void_p )

###############################################################################
## @brief Function-pointer for urKernelSetArgSampler
if __use_win_types:
    _urKernelSetArgSampler_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, ur_sampler_handle_t )
else:
    _urKernelSetArgSampler_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, ur_sampler_handle_t )

###############################################################################
## @brief Function-pointer for urKernelSetArgMemObj
if __use_win_types:
    _urKernelSetArgMemObj_t = WINFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, ur_mem_handle_t )
else:
    _urKernelSetArgMemObj_t = CFUNCTYPE( ur_result_t, ur_kernel_handle_t, c_ulong, ur_mem_handle_t )


###############################################################################
## @brief Table of Kernel functions pointers
class _ur_kernel_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _urKernelCreate_t
        ("pfnGetInfo", c_void_p),                                       ## _urKernelGetInfo_t
        ("pfnGetGroupInfo", c_void_p),                                  ## _urKernelGetGroupInfo_t
        ("pfnGetSubGroupInfo", c_void_p),                               ## _urKernelGetSubGroupInfo_t
        ("pfnRetain", c_void_p),                                        ## _urKernelRetain_t
        ("pfnRelease", c_void_p),                                       ## _urKernelRelease_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urKernelGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p),                        ## _urKernelCreateWithNativeHandle_t
        ("pfnSetArg", c_void_p),                                        ## _urKernelSetArg_t
        ("pfnSetArgPointer", c_void_p),                                 ## _urKernelSetArgPointer_t
        ("pfnSetExecInfo", c_void_p),                                   ## _urKernelSetExecInfo_t
        ("pfnSetArgSampler", c_void_p),                                 ## _urKernelSetArgSampler_t
        ("pfnSetArgMemObj", c_void_p)                                   ## _urKernelSetArgMemObj_t
    ]

###############################################################################
## @brief Function-pointer for urSamplerCreate
if __use_win_types:
    _urSamplerCreate_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_sampler_property_value_t), POINTER(ur_sampler_handle_t) )
else:
    _urSamplerCreate_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_sampler_property_value_t), POINTER(ur_sampler_handle_t) )

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
    _urSamplerCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_sampler_handle_t, ur_native_handle_t, POINTER(ur_sampler_handle_t) )
else:
    _urSamplerCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_sampler_handle_t, ur_native_handle_t, POINTER(ur_sampler_handle_t) )


###############################################################################
## @brief Table of Sampler functions pointers
class _ur_sampler_dditable_t(Structure):
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
    _urMemBufferCreate_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_mem_flags_t, c_size_t, c_void_p, POINTER(ur_mem_handle_t) )
else:
    _urMemBufferCreate_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_mem_flags_t, c_size_t, c_void_p, POINTER(ur_mem_handle_t) )

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
## @brief Function-pointer for urMemCreateWithNativeHandle
if __use_win_types:
    _urMemCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_mem_handle_t) )
else:
    _urMemCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_mem_handle_t) )

###############################################################################
## @brief Function-pointer for urMemFree
if __use_win_types:
    _urMemFree_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p )
else:
    _urMemFree_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p )

###############################################################################
## @brief Function-pointer for urMemGetMemAllocInfo
if __use_win_types:
    _urMemGetMemAllocInfo_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, ur_mem_info_t, c_size_t, c_void_p, POINTER(c_size_t) )
else:
    _urMemGetMemAllocInfo_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, c_void_p, ur_mem_info_t, c_size_t, c_void_p, POINTER(c_size_t) )


###############################################################################
## @brief Table of Mem functions pointers
class _ur_mem_dditable_t(Structure):
    _fields_ = [
        ("pfnImageCreate", c_void_p),                                   ## _urMemImageCreate_t
        ("pfnBufferCreate", c_void_p),                                  ## _urMemBufferCreate_t
        ("pfnRetain", c_void_p),                                        ## _urMemRetain_t
        ("pfnRelease", c_void_p),                                       ## _urMemRelease_t
        ("pfnBufferPartition", c_void_p),                               ## _urMemBufferPartition_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urMemGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p),                        ## _urMemCreateWithNativeHandle_t
        ("pfnFree", c_void_p),                                          ## _urMemFree_t
        ("pfnGetMemAllocInfo", c_void_p)                                ## _urMemGetMemAllocInfo_t
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
    _urEnqueueMemBufferRead_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemBufferRead_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemBufferWrite
if __use_win_types:
    _urEnqueueMemBufferWrite_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemBufferWrite_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemBufferReadRect
if __use_win_types:
    _urEnqueueMemBufferReadRect_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemBufferReadRect_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemBufferWriteRect
if __use_win_types:
    _urEnqueueMemBufferWriteRect_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemBufferWriteRect_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemBufferCopy
if __use_win_types:
    _urEnqueueMemBufferCopy_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, ur_mem_handle_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemBufferCopy_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, ur_mem_handle_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

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
    _urEnqueueMemImageRead_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemImageRead_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemImageWrite
if __use_win_types:
    _urEnqueueMemImageWrite_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemImageWrite_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, ur_rect_offset_t, ur_rect_region_t, c_size_t, c_size_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemImageCopy
if __use_win_types:
    _urEnqueueMemImageCopy_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, ur_mem_handle_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemImageCopy_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, ur_mem_handle_t, ur_rect_offset_t, ur_rect_offset_t, ur_rect_region_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueMemBufferMap
if __use_win_types:
    _urEnqueueMemBufferMap_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, ur_map_flags_t, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t), POINTER(c_void_p) )
else:
    _urEnqueueMemBufferMap_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, bool, ur_map_flags_t, c_size_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t), POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for urEnqueueMemUnmap
if __use_win_types:
    _urEnqueueMemUnmap_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueMemUnmap_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_mem_handle_t, c_void_p, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueUSMMemset
if __use_win_types:
    _urEnqueueUSMMemset_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_int8_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueUSMMemset_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_int8_t, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueUSMMemcpy
if __use_win_types:
    _urEnqueueUSMMemcpy_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, bool, c_void_p, c_void_p, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueUSMMemcpy_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, bool, c_void_p, c_void_p, c_size_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueUSMPrefetch
if __use_win_types:
    _urEnqueueUSMPrefetch_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_size_t, ur_usm_migration_flags_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )
else:
    _urEnqueueUSMPrefetch_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_size_t, ur_usm_migration_flags_t, c_ulong, POINTER(ur_event_handle_t), POINTER(ur_event_handle_t) )

###############################################################################
## @brief Function-pointer for urEnqueueUSMMemAdvice
if __use_win_types:
    _urEnqueueUSMMemAdvice_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_size_t, ur_mem_advice_t, POINTER(ur_event_handle_t) )
else:
    _urEnqueueUSMMemAdvice_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, c_void_p, c_size_t, ur_mem_advice_t, POINTER(ur_event_handle_t) )


###############################################################################
## @brief Table of Enqueue functions pointers
class _ur_enqueue_dditable_t(Structure):
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
        ("pfnUSMMemset", c_void_p),                                     ## _urEnqueueUSMMemset_t
        ("pfnUSMMemcpy", c_void_p),                                     ## _urEnqueueUSMMemcpy_t
        ("pfnUSMPrefetch", c_void_p),                                   ## _urEnqueueUSMPrefetch_t
        ("pfnUSMMemAdvice", c_void_p)                                   ## _urEnqueueUSMMemAdvice_t
    ]

###############################################################################
## @brief Function-pointer for urUSMHostAlloc
if __use_win_types:
    _urUSMHostAlloc_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_usm_mem_flags_t), c_size_t, c_ulong, POINTER(c_void_p) )
else:
    _urUSMHostAlloc_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, POINTER(ur_usm_mem_flags_t), c_size_t, c_ulong, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for urUSMDeviceAlloc
if __use_win_types:
    _urUSMDeviceAlloc_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_usm_mem_flags_t), c_size_t, c_ulong, POINTER(c_void_p) )
else:
    _urUSMDeviceAlloc_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_usm_mem_flags_t), c_size_t, c_ulong, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for urUSMSharedAlloc
if __use_win_types:
    _urUSMSharedAlloc_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_usm_mem_flags_t), c_size_t, c_ulong, POINTER(c_void_p) )
else:
    _urUSMSharedAlloc_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, POINTER(ur_usm_mem_flags_t), c_size_t, c_ulong, POINTER(c_void_p) )


###############################################################################
## @brief Table of USM functions pointers
class _ur_usm_dditable_t(Structure):
    _fields_ = [
        ("pfnHostAlloc", c_void_p),                                     ## _urUSMHostAlloc_t
        ("pfnDeviceAlloc", c_void_p),                                   ## _urUSMDeviceAlloc_t
        ("pfnSharedAlloc", c_void_p)                                    ## _urUSMSharedAlloc_t
    ]

###############################################################################
## @brief Function-pointer for urTearDown
if __use_win_types:
    _urTearDown_t = WINFUNCTYPE( ur_result_t, c_void_p )
else:
    _urTearDown_t = CFUNCTYPE( ur_result_t, c_void_p )

###############################################################################
## @brief Function-pointer for urInit
if __use_win_types:
    _urInit_t = WINFUNCTYPE( ur_result_t, ur_platform_init_flags_t, ur_device_init_flags_t )
else:
    _urInit_t = CFUNCTYPE( ur_result_t, ur_platform_init_flags_t, ur_device_init_flags_t )


###############################################################################
## @brief Table of Global functions pointers
class _ur_global_dditable_t(Structure):
    _fields_ = [
        ("pfnTearDown", c_void_p),                                      ## _urTearDown_t
        ("pfnInit", c_void_p)                                           ## _urInit_t
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
    _urQueueCreate_t = WINFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_queue_flags_t, POINTER(ur_queue_handle_t) )
else:
    _urQueueCreate_t = CFUNCTYPE( ur_result_t, ur_context_handle_t, ur_device_handle_t, ur_queue_flags_t, POINTER(ur_queue_handle_t) )

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
    _urQueueGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, POINTER(ur_native_handle_t) )
else:
    _urQueueGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urQueueCreateWithNativeHandle
if __use_win_types:
    _urQueueCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_native_handle_t, POINTER(ur_queue_handle_t) )
else:
    _urQueueCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_queue_handle_t, ur_native_handle_t, POINTER(ur_queue_handle_t) )


###############################################################################
## @brief Table of Queue functions pointers
class _ur_queue_dditable_t(Structure):
    _fields_ = [
        ("pfnGetInfo", c_void_p),                                       ## _urQueueGetInfo_t
        ("pfnCreate", c_void_p),                                        ## _urQueueCreate_t
        ("pfnRetain", c_void_p),                                        ## _urQueueRetain_t
        ("pfnRelease", c_void_p),                                       ## _urQueueRelease_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urQueueGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p)                         ## _urQueueCreateWithNativeHandle_t
    ]

###############################################################################
## @brief Function-pointer for urDeviceGet
if __use_win_types:
    _urDeviceGet_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_device_type_t, POINTER(c_ulong), POINTER(ur_device_handle_t) )
else:
    _urDeviceGet_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_device_type_t, POINTER(c_ulong), POINTER(ur_device_handle_t) )

###############################################################################
## @brief Function-pointer for urDeviceGetInfo
if __use_win_types:
    _urDeviceGetInfo_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t, ur_device_info_t, POINTER(c_size_t), c_void_p )
else:
    _urDeviceGetInfo_t = CFUNCTYPE( ur_result_t, ur_device_handle_t, ur_device_info_t, POINTER(c_size_t), c_void_p )

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
    _urDevicePartition_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t, POINTER(ur_device_partition_property_value_t), POINTER(c_ulong), POINTER(ur_device_handle_t) )
else:
    _urDevicePartition_t = CFUNCTYPE( ur_result_t, ur_device_handle_t, POINTER(ur_device_partition_property_value_t), POINTER(c_ulong), POINTER(ur_device_handle_t) )

###############################################################################
## @brief Function-pointer for urDeviceSelectBinary
if __use_win_types:
    _urDeviceSelectBinary_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t, c_ulong, POINTER(c_ubyte*), c_ulong )
else:
    _urDeviceSelectBinary_t = CFUNCTYPE( ur_result_t, ur_device_handle_t, c_ulong, POINTER(c_ubyte*), c_ulong )

###############################################################################
## @brief Function-pointer for urDeviceGetNativeHandle
if __use_win_types:
    _urDeviceGetNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_device_handle_t, POINTER(ur_native_handle_t) )
else:
    _urDeviceGetNativeHandle_t = CFUNCTYPE( ur_result_t, ur_device_handle_t, POINTER(ur_native_handle_t) )

###############################################################################
## @brief Function-pointer for urDeviceCreateWithNativeHandle
if __use_win_types:
    _urDeviceCreateWithNativeHandle_t = WINFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_device_handle_t) )
else:
    _urDeviceCreateWithNativeHandle_t = CFUNCTYPE( ur_result_t, ur_platform_handle_t, ur_native_handle_t, POINTER(ur_device_handle_t) )


###############################################################################
## @brief Table of Device functions pointers
class _ur_device_dditable_t(Structure):
    _fields_ = [
        ("pfnGet", c_void_p),                                           ## _urDeviceGet_t
        ("pfnGetInfo", c_void_p),                                       ## _urDeviceGetInfo_t
        ("pfnRetain", c_void_p),                                        ## _urDeviceRetain_t
        ("pfnRelease", c_void_p),                                       ## _urDeviceRelease_t
        ("pfnPartition", c_void_p),                                     ## _urDevicePartition_t
        ("pfnSelectBinary", c_void_p),                                  ## _urDeviceSelectBinary_t
        ("pfnGetNativeHandle", c_void_p),                               ## _urDeviceGetNativeHandle_t
        ("pfnCreateWithNativeHandle", c_void_p)                         ## _urDeviceCreateWithNativeHandle_t
    ]

###############################################################################
class _ur_dditable_t(Structure):
    _fields_ = [
        ("Platform", _ur_platform_dditable_t),
        ("Context", _ur_context_dditable_t),
        ("Event", _ur_event_dditable_t),
        ("Program", _ur_program_dditable_t),
        ("Module", _ur_module_dditable_t),
        ("Kernel", _ur_kernel_dditable_t),
        ("Sampler", _ur_sampler_dditable_t),
        ("Mem", _ur_mem_dditable_t),
        ("Enqueue", _ur_enqueue_dditable_t),
        ("USM", _ur_usm_dditable_t),
        ("Global", _ur_global_dditable_t),
        ("Queue", _ur_queue_dditable_t),
        ("Device", _ur_device_dditable_t)
    ]

###############################################################################
## @brief ur device-driver interfaces
class UR_DDI:
    def __init__(self, version : ur_api_version_t):
        # load the ur_loader library
        if "Windows" == platform.uname()[0]:
            self.__dll = WinDLL("ur_loader.dll")
        else:
            self.__dll = CDLL("ur_loader.so")

        # fill the ddi tables
        self.__dditable = _ur_dditable_t()

        # call driver to get function pointers
        _Platform = _ur_platform_dditable_t()
        r = ur_result_v(self.__dll.urGetPlatformProcAddrTable(version, byref(_Platform)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Platform = _Platform

        # attach function interface to function address
        self.urPlatformGet = _urPlatformGet_t(self.__dditable.Platform.pfnGet)
        self.urPlatformGetInfo = _urPlatformGetInfo_t(self.__dditable.Platform.pfnGetInfo)
        self.urPlatformGetNativeHandle = _urPlatformGetNativeHandle_t(self.__dditable.Platform.pfnGetNativeHandle)
        self.urPlatformCreateWithNativeHandle = _urPlatformCreateWithNativeHandle_t(self.__dditable.Platform.pfnCreateWithNativeHandle)
        self.urPlatformGetApiVersion = _urPlatformGetApiVersion_t(self.__dditable.Platform.pfnGetApiVersion)

        # call driver to get function pointers
        _Context = _ur_context_dditable_t()
        r = ur_result_v(self.__dll.urGetContextProcAddrTable(version, byref(_Context)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Context = _Context

        # attach function interface to function address
        self.urContextCreate = _urContextCreate_t(self.__dditable.Context.pfnCreate)
        self.urContextRetain = _urContextRetain_t(self.__dditable.Context.pfnRetain)
        self.urContextRelease = _urContextRelease_t(self.__dditable.Context.pfnRelease)
        self.urContextGetInfo = _urContextGetInfo_t(self.__dditable.Context.pfnGetInfo)
        self.urContextGetNativeHandle = _urContextGetNativeHandle_t(self.__dditable.Context.pfnGetNativeHandle)
        self.urContextCreateWithNativeHandle = _urContextCreateWithNativeHandle_t(self.__dditable.Context.pfnCreateWithNativeHandle)

        # call driver to get function pointers
        _Event = _ur_event_dditable_t()
        r = ur_result_v(self.__dll.urGetEventProcAddrTable(version, byref(_Event)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Event = _Event

        # attach function interface to function address
        self.urEventCreate = _urEventCreate_t(self.__dditable.Event.pfnCreate)
        self.urEventGetInfo = _urEventGetInfo_t(self.__dditable.Event.pfnGetInfo)
        self.urEventGetProfilingInfo = _urEventGetProfilingInfo_t(self.__dditable.Event.pfnGetProfilingInfo)
        self.urEventWait = _urEventWait_t(self.__dditable.Event.pfnWait)
        self.urEventRetain = _urEventRetain_t(self.__dditable.Event.pfnRetain)
        self.urEventRelease = _urEventRelease_t(self.__dditable.Event.pfnRelease)
        self.urEventGetNativeHandle = _urEventGetNativeHandle_t(self.__dditable.Event.pfnGetNativeHandle)
        self.urEventCreateWithNativeHandle = _urEventCreateWithNativeHandle_t(self.__dditable.Event.pfnCreateWithNativeHandle)

        # call driver to get function pointers
        _Program = _ur_program_dditable_t()
        r = ur_result_v(self.__dll.urGetProgramProcAddrTable(version, byref(_Program)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Program = _Program

        # attach function interface to function address
        self.urProgramCreate = _urProgramCreate_t(self.__dditable.Program.pfnCreate)
        self.urProgramCreateWithBinary = _urProgramCreateWithBinary_t(self.__dditable.Program.pfnCreateWithBinary)
        self.urProgramRetain = _urProgramRetain_t(self.__dditable.Program.pfnRetain)
        self.urProgramRelease = _urProgramRelease_t(self.__dditable.Program.pfnRelease)
        self.urProgramGetFunctionPointer = _urProgramGetFunctionPointer_t(self.__dditable.Program.pfnGetFunctionPointer)
        self.urProgramGetInfo = _urProgramGetInfo_t(self.__dditable.Program.pfnGetInfo)
        self.urProgramGetBuildInfo = _urProgramGetBuildInfo_t(self.__dditable.Program.pfnGetBuildInfo)
        self.urProgramSetSpecializationConstant = _urProgramSetSpecializationConstant_t(self.__dditable.Program.pfnSetSpecializationConstant)
        self.urProgramGetNativeHandle = _urProgramGetNativeHandle_t(self.__dditable.Program.pfnGetNativeHandle)
        self.urProgramCreateWithNativeHandle = _urProgramCreateWithNativeHandle_t(self.__dditable.Program.pfnCreateWithNativeHandle)

        # call driver to get function pointers
        _Module = _ur_module_dditable_t()
        r = ur_result_v(self.__dll.urGetModuleProcAddrTable(version, byref(_Module)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Module = _Module

        # attach function interface to function address
        self.urModuleCreate = _urModuleCreate_t(self.__dditable.Module.pfnCreate)
        self.urModuleRetain = _urModuleRetain_t(self.__dditable.Module.pfnRetain)
        self.urModuleRelease = _urModuleRelease_t(self.__dditable.Module.pfnRelease)
        self.urModuleGetNativeHandle = _urModuleGetNativeHandle_t(self.__dditable.Module.pfnGetNativeHandle)
        self.urModuleCreateWithNativeHandle = _urModuleCreateWithNativeHandle_t(self.__dditable.Module.pfnCreateWithNativeHandle)

        # call driver to get function pointers
        _Kernel = _ur_kernel_dditable_t()
        r = ur_result_v(self.__dll.urGetKernelProcAddrTable(version, byref(_Kernel)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Kernel = _Kernel

        # attach function interface to function address
        self.urKernelCreate = _urKernelCreate_t(self.__dditable.Kernel.pfnCreate)
        self.urKernelGetInfo = _urKernelGetInfo_t(self.__dditable.Kernel.pfnGetInfo)
        self.urKernelGetGroupInfo = _urKernelGetGroupInfo_t(self.__dditable.Kernel.pfnGetGroupInfo)
        self.urKernelGetSubGroupInfo = _urKernelGetSubGroupInfo_t(self.__dditable.Kernel.pfnGetSubGroupInfo)
        self.urKernelRetain = _urKernelRetain_t(self.__dditable.Kernel.pfnRetain)
        self.urKernelRelease = _urKernelRelease_t(self.__dditable.Kernel.pfnRelease)
        self.urKernelGetNativeHandle = _urKernelGetNativeHandle_t(self.__dditable.Kernel.pfnGetNativeHandle)
        self.urKernelCreateWithNativeHandle = _urKernelCreateWithNativeHandle_t(self.__dditable.Kernel.pfnCreateWithNativeHandle)
        self.urKernelSetArg = _urKernelSetArg_t(self.__dditable.Kernel.pfnSetArg)
        self.urKernelSetArgPointer = _urKernelSetArgPointer_t(self.__dditable.Kernel.pfnSetArgPointer)
        self.urKernelSetExecInfo = _urKernelSetExecInfo_t(self.__dditable.Kernel.pfnSetExecInfo)
        self.urKernelSetArgSampler = _urKernelSetArgSampler_t(self.__dditable.Kernel.pfnSetArgSampler)
        self.urKernelSetArgMemObj = _urKernelSetArgMemObj_t(self.__dditable.Kernel.pfnSetArgMemObj)

        # call driver to get function pointers
        _Sampler = _ur_sampler_dditable_t()
        r = ur_result_v(self.__dll.urGetSamplerProcAddrTable(version, byref(_Sampler)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Sampler = _Sampler

        # attach function interface to function address
        self.urSamplerCreate = _urSamplerCreate_t(self.__dditable.Sampler.pfnCreate)
        self.urSamplerRetain = _urSamplerRetain_t(self.__dditable.Sampler.pfnRetain)
        self.urSamplerRelease = _urSamplerRelease_t(self.__dditable.Sampler.pfnRelease)
        self.urSamplerGetInfo = _urSamplerGetInfo_t(self.__dditable.Sampler.pfnGetInfo)
        self.urSamplerGetNativeHandle = _urSamplerGetNativeHandle_t(self.__dditable.Sampler.pfnGetNativeHandle)
        self.urSamplerCreateWithNativeHandle = _urSamplerCreateWithNativeHandle_t(self.__dditable.Sampler.pfnCreateWithNativeHandle)

        # call driver to get function pointers
        _Mem = _ur_mem_dditable_t()
        r = ur_result_v(self.__dll.urGetMemProcAddrTable(version, byref(_Mem)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Mem = _Mem

        # attach function interface to function address
        self.urMemImageCreate = _urMemImageCreate_t(self.__dditable.Mem.pfnImageCreate)
        self.urMemBufferCreate = _urMemBufferCreate_t(self.__dditable.Mem.pfnBufferCreate)
        self.urMemRetain = _urMemRetain_t(self.__dditable.Mem.pfnRetain)
        self.urMemRelease = _urMemRelease_t(self.__dditable.Mem.pfnRelease)
        self.urMemBufferPartition = _urMemBufferPartition_t(self.__dditable.Mem.pfnBufferPartition)
        self.urMemGetNativeHandle = _urMemGetNativeHandle_t(self.__dditable.Mem.pfnGetNativeHandle)
        self.urMemCreateWithNativeHandle = _urMemCreateWithNativeHandle_t(self.__dditable.Mem.pfnCreateWithNativeHandle)
        self.urMemFree = _urMemFree_t(self.__dditable.Mem.pfnFree)
        self.urMemGetMemAllocInfo = _urMemGetMemAllocInfo_t(self.__dditable.Mem.pfnGetMemAllocInfo)

        # call driver to get function pointers
        _Enqueue = _ur_enqueue_dditable_t()
        r = ur_result_v(self.__dll.urGetEnqueueProcAddrTable(version, byref(_Enqueue)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Enqueue = _Enqueue

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
        self.urEnqueueUSMMemset = _urEnqueueUSMMemset_t(self.__dditable.Enqueue.pfnUSMMemset)
        self.urEnqueueUSMMemcpy = _urEnqueueUSMMemcpy_t(self.__dditable.Enqueue.pfnUSMMemcpy)
        self.urEnqueueUSMPrefetch = _urEnqueueUSMPrefetch_t(self.__dditable.Enqueue.pfnUSMPrefetch)
        self.urEnqueueUSMMemAdvice = _urEnqueueUSMMemAdvice_t(self.__dditable.Enqueue.pfnUSMMemAdvice)

        # call driver to get function pointers
        _USM = _ur_usm_dditable_t()
        r = ur_result_v(self.__dll.urGetUSMProcAddrTable(version, byref(_USM)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.USM = _USM

        # attach function interface to function address
        self.urUSMHostAlloc = _urUSMHostAlloc_t(self.__dditable.USM.pfnHostAlloc)
        self.urUSMDeviceAlloc = _urUSMDeviceAlloc_t(self.__dditable.USM.pfnDeviceAlloc)
        self.urUSMSharedAlloc = _urUSMSharedAlloc_t(self.__dditable.USM.pfnSharedAlloc)

        # call driver to get function pointers
        _Global = _ur_global_dditable_t()
        r = ur_result_v(self.__dll.urGetGlobalProcAddrTable(version, byref(_Global)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Global = _Global

        # attach function interface to function address
        self.urTearDown = _urTearDown_t(self.__dditable.Global.pfnTearDown)
        self.urInit = _urInit_t(self.__dditable.Global.pfnInit)

        # call driver to get function pointers
        _Queue = _ur_queue_dditable_t()
        r = ur_result_v(self.__dll.urGetQueueProcAddrTable(version, byref(_Queue)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Queue = _Queue

        # attach function interface to function address
        self.urQueueGetInfo = _urQueueGetInfo_t(self.__dditable.Queue.pfnGetInfo)
        self.urQueueCreate = _urQueueCreate_t(self.__dditable.Queue.pfnCreate)
        self.urQueueRetain = _urQueueRetain_t(self.__dditable.Queue.pfnRetain)
        self.urQueueRelease = _urQueueRelease_t(self.__dditable.Queue.pfnRelease)
        self.urQueueGetNativeHandle = _urQueueGetNativeHandle_t(self.__dditable.Queue.pfnGetNativeHandle)
        self.urQueueCreateWithNativeHandle = _urQueueCreateWithNativeHandle_t(self.__dditable.Queue.pfnCreateWithNativeHandle)

        # call driver to get function pointers
        _Device = _ur_device_dditable_t()
        r = ur_result_v(self.__dll.urGetDeviceProcAddrTable(version, byref(_Device)))
        if r != ur_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Device = _Device

        # attach function interface to function address
        self.urDeviceGet = _urDeviceGet_t(self.__dditable.Device.pfnGet)
        self.urDeviceGetInfo = _urDeviceGetInfo_t(self.__dditable.Device.pfnGetInfo)
        self.urDeviceRetain = _urDeviceRetain_t(self.__dditable.Device.pfnRetain)
        self.urDeviceRelease = _urDeviceRelease_t(self.__dditable.Device.pfnRelease)
        self.urDevicePartition = _urDevicePartition_t(self.__dditable.Device.pfnPartition)
        self.urDeviceSelectBinary = _urDeviceSelectBinary_t(self.__dditable.Device.pfnSelectBinary)
        self.urDeviceGetNativeHandle = _urDeviceGetNativeHandle_t(self.__dditable.Device.pfnGetNativeHandle)
        self.urDeviceCreateWithNativeHandle = _urDeviceCreateWithNativeHandle_t(self.__dditable.Device.pfnCreateWithNativeHandle)

        # success!
