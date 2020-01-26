//==---------- pi.h - Plugin Interface -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition of a generic offload Plugin Interface (PI), which is
// used by the SYCL implementation to connect to multiple device back-ends,
// e.g. to OpenCL. The interface is intentionally kept C-only for the
// purpose of having full flexibility and interoperability with different
// environments.
//
#ifndef _PI_H_
#define _PI_H_

// Every single change in PI API should be accompanied with the minor
// version increase (+1). In the cases where backward compatibility is not
// maintained there should be a (+1) change to the major version in
// addition to the increase of the minor.
//
#define _PI_H_VERSION_MAJOR 1
#define _PI_H_VERSION_MINOR 1

#define _PI_STRING_HELPER(a) #a
#define _PI_CONCAT(a, b) _PI_STRING_HELPER(a.b)
#define _PI_H_VERSION_STRING                                                   \
  _PI_CONCAT(_PI_H_VERSION_MAJOR, _PI_H_VERSION_MINOR)
// TODO: we need a mapping of PI to OpenCL somewhere, and this can be done
// elsewhere, e.g. in the pi_opencl, but constants/enums mapping is now
// done here, for efficiency and simplicity.
//
#include <CL/opencl.h>
#include <CL/cl_usm_ext.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef int32_t     pi_int32;
typedef uint32_t    pi_uint32;
typedef uint64_t    pi_uint64;
typedef pi_uint32   pi_bool;
typedef pi_uint64   pi_bitfield;

//
// NOTE: prefer to map 1:1 to OpenCL so that no translation is needed
// for PI <-> OpenCL ways. The PI <-> to other BE translation is almost
// always needed anyway.
//
// TODO: populate PI enums.
//
typedef enum {
  PI_SUCCESS = CL_SUCCESS,
  PI_RESULT_INVALID_KERNEL_NAME = CL_INVALID_KERNEL_NAME,
  PI_INVALID_OPERATION = CL_INVALID_OPERATION,
  PI_INVALID_QUEUE_PROPERTIES = CL_INVALID_QUEUE_PROPERTIES,
  PI_INVALID_VALUE = CL_INVALID_VALUE,
  PI_INVALID_CONTEXT = CL_INVALID_CONTEXT,
  PI_INVALID_PLATFORM = CL_INVALID_PLATFORM,
  PI_INVALID_DEVICE = CL_INVALID_DEVICE,
  PI_INVALID_BINARY = CL_INVALID_BINARY,
  PI_MISALIGNED_SUB_BUFFER_OFFSET = CL_MISALIGNED_SUB_BUFFER_OFFSET,
  PI_OUT_OF_HOST_MEMORY = CL_OUT_OF_HOST_MEMORY,
  PI_INVALID_WORK_GROUP_SIZE = CL_INVALID_WORK_GROUP_SIZE
} _pi_result;

typedef enum {
  PI_PLATFORM_INFO_EXTENSIONS = CL_PLATFORM_EXTENSIONS,
  PI_PLATFORM_INFO_NAME       = CL_PLATFORM_NAME,
  PI_PLATFORM_INFO_PROFILE    = CL_PLATFORM_PROFILE,
  PI_PLATFORM_INFO_VENDOR     = CL_PLATFORM_VENDOR,
  PI_PLATFORM_INFO_VERSION    = CL_PLATFORM_VERSION,
} _pi_platform_info;

// NOTE: this is made 64-bit to match the size of cl_device_type to
// make the translation to OpenCL transparent.
//
typedef enum : pi_uint64 {
  PI_DEVICE_TYPE_CPU = CL_DEVICE_TYPE_CPU,
  PI_DEVICE_TYPE_GPU = CL_DEVICE_TYPE_GPU,
  PI_DEVICE_TYPE_ACC = CL_DEVICE_TYPE_ACCELERATOR
} _pi_device_type;

// TODO: populate and sync with cl::sycl::info::device
typedef enum {
  PI_DEVICE_INFO_TYPE                = CL_DEVICE_TYPE,
  PI_DEVICE_INFO_PARENT              = CL_DEVICE_PARENT_DEVICE,
  PI_DEVICE_INFO_PLATFORM            = CL_DEVICE_PLATFORM,
  PI_DEVICE_INFO_PARTITION_TYPE      = CL_DEVICE_PARTITION_TYPE,
  PI_DEVICE_INFO_NAME                = CL_DEVICE_NAME,
  PI_DEVICE_INFO_VERSION             = CL_DEVICE_VERSION,
  PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE = CL_DEVICE_MAX_WORK_GROUP_SIZE,
  PI_DEVICE_INFO_EXTENSIONS          = CL_DEVICE_EXTENSIONS
} _pi_device_info;

// TODO: populate
typedef enum {
  PI_CONTEXT_INFO_DEVICES     = CL_CONTEXT_DEVICES,
  PI_CONTEXT_INFO_NUM_DEVICES = CL_CONTEXT_NUM_DEVICES
} _pi_context_info;

// TODO: populate
typedef enum {
  PI_QUEUE_INFO_DEVICE          = CL_QUEUE_DEVICE,
  PI_QUEUE_INFO_REFERENCE_COUNT = CL_QUEUE_REFERENCE_COUNT
} _pi_queue_info;

typedef enum {
  PI_IMAGE_INFO_FORMAT       = CL_IMAGE_FORMAT,
  PI_IMAGE_INFO_ELEMENT_SIZE = CL_IMAGE_ELEMENT_SIZE,
  PI_IMAGE_INFO_ROW_PITCH    = CL_IMAGE_ROW_PITCH,
  PI_IMAGE_INFO_SLICE_PITCH  = CL_IMAGE_SLICE_PITCH,
  PI_IMAGE_INFO_WIDTH        = CL_IMAGE_WIDTH,
  PI_IMAGE_INFO_HEIGHT       = CL_IMAGE_HEIGHT,
  PI_IMAGE_INFO_DEPTH        = CL_IMAGE_DEPTH
} _pi_image_info;

typedef enum {
  PI_MEM_TYPE_BUFFER         = CL_MEM_OBJECT_BUFFER,
  PI_MEM_TYPE_IMAGE2D        = CL_MEM_OBJECT_IMAGE2D,
  PI_MEM_TYPE_IMAGE3D        = CL_MEM_OBJECT_IMAGE3D,
  PI_MEM_TYPE_IMAGE2D_ARRAY  = CL_MEM_OBJECT_IMAGE2D_ARRAY,
  PI_MEM_TYPE_IMAGE1D        = CL_MEM_OBJECT_IMAGE1D,
  PI_MEM_TYPE_IMAGE1D_ARRAY  = CL_MEM_OBJECT_IMAGE1D_ARRAY,
  PI_MEM_TYPE_IMAGE1D_BUFFER = CL_MEM_OBJECT_IMAGE1D_BUFFER
} _pi_mem_type;

typedef enum {
  PI_IMAGE_CHANNEL_ORDER_A         = CL_A,
  PI_IMAGE_CHANNEL_ORDER_R         = CL_R,
  PI_IMAGE_CHANNEL_ORDER_RG        = CL_RG,
  PI_IMAGE_CHANNEL_ORDER_RA        = CL_RA,
  PI_IMAGE_CHANNEL_ORDER_RGB       = CL_RGB,
  PI_IMAGE_CHANNEL_ORDER_RGBA      = CL_RGBA,
  PI_IMAGE_CHANNEL_ORDER_BGRA      = CL_BGRA,
  PI_IMAGE_CHANNEL_ORDER_ARGB      = CL_ARGB,
  PI_IMAGE_CHANNEL_ORDER_ABGR      = CL_ABGR,
  PI_IMAGE_CHANNEL_ORDER_INTENSITY = CL_INTENSITY,
  PI_IMAGE_CHANNEL_ORDER_LUMINANCE = CL_LUMINANCE,
  PI_IMAGE_CHANNEL_ORDER_Rx        = CL_Rx,
  PI_IMAGE_CHANNEL_ORDER_RGx       = CL_RGx,
  PI_IMAGE_CHANNEL_ORDER_RGBx      = CL_RGBx
} _pi_image_channel_order;

typedef enum {
  PI_IMAGE_CHANNEL_TYPE_SNORM_INT8       = CL_SNORM_INT8,
  PI_IMAGE_CHANNEL_TYPE_SNORM_INT16      = CL_SNORM_INT16,
  PI_IMAGE_CHANNEL_TYPE_UNORM_INT8       = CL_UNORM_INT8,
  PI_IMAGE_CHANNEL_TYPE_UNORM_INT16      = CL_UNORM_INT16,
  PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565  = CL_UNORM_SHORT_565,
  PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555  = CL_UNORM_SHORT_555,
  PI_IMAGE_CHANNEL_TYPE_UNORM_INT_101010 = CL_UNORM_INT_101010,
  PI_IMAGE_CHANNEL_TYPE_SIGNED_INT8      = CL_SIGNED_INT8,
  PI_IMAGE_CHANNEL_TYPE_SIGNED_INT16     = CL_SIGNED_INT16,
  PI_IMAGE_CHANNEL_TYPE_SIGNED_INT32     = CL_SIGNED_INT32,
  PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8    = CL_UNSIGNED_INT8,
  PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16   = CL_UNSIGNED_INT16,
  PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32   = CL_UNSIGNED_INT32,
  PI_IMAGE_CHANNEL_TYPE_HALF_FLOAT       = CL_HALF_FLOAT,
  PI_IMAGE_CHANNEL_TYPE_FLOAT            = CL_FLOAT
} _pi_image_channel_type;

typedef enum {
  PI_BUFFER_CREATE_TYPE_REGION = CL_BUFFER_CREATE_TYPE_REGION
} _pi_buffer_create_type;

typedef pi_bitfield pi_sampler_properties;
const pi_bool PI_TRUE = CL_TRUE;
const pi_bool PI_FALSE = CL_FALSE;

typedef enum {
 PI_SAMPLER_INFO_REFERENCE_COUNT   = CL_SAMPLER_REFERENCE_COUNT,
 PI_SAMPLER_INFO_CONTEXT           = CL_SAMPLER_CONTEXT,
 PI_SAMPLER_INFO_NORMALIZED_COORDS = CL_SAMPLER_NORMALIZED_COORDS,
 PI_SAMPLER_INFO_ADDRESSING_MODE   = CL_SAMPLER_ADDRESSING_MODE,
 PI_SAMPLER_INFO_FILTER_MODE       = CL_SAMPLER_FILTER_MODE,
 PI_SAMPLER_INFO_MIP_FILTER_MODE   = CL_SAMPLER_MIP_FILTER_MODE,
 PI_SAMPLER_INFO_LOD_MIN           = CL_SAMPLER_LOD_MIN,
 PI_SAMPLER_INFO_LOD_MAX           = CL_SAMPLER_LOD_MAX
} _pi_sampler_info;

typedef enum {
  PI_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT = CL_ADDRESS_MIRRORED_REPEAT,
  PI_SAMPLER_ADDRESSING_MODE_REPEAT          = CL_ADDRESS_REPEAT,
  PI_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE   = CL_ADDRESS_CLAMP_TO_EDGE,
  PI_SAMPLER_ADDRESSING_MODE_CLAMP           = CL_ADDRESS_CLAMP,
  PI_SAMPLER_ADDRESSING_MODE_NONE            = CL_ADDRESS_NONE
} _pi_sampler_addressing_mode;

typedef enum {
  PI_SAMPLER_FILTER_MODE_NEAREST = CL_FILTER_NEAREST,
  PI_SAMPLER_FILTER_MODE_LINEAR  = CL_FILTER_LINEAR,
} _pi_sampler_filter_mode;

// NOTE: this is made 64-bit to match the size of cl_mem_flags to
// make the translation to OpenCL transparent.
// TODO: populate
//
typedef pi_bitfield pi_mem_flags;
// Access
const pi_mem_flags PI_MEM_FLAGS_ACCESS_RW     = CL_MEM_READ_WRITE;
// Host pointer
const pi_mem_flags PI_MEM_FLAGS_HOST_PTR_USE  = CL_MEM_USE_HOST_PTR;
const pi_mem_flags PI_MEM_FLAGS_HOST_PTR_COPY = CL_MEM_COPY_HOST_PTR;

// NOTE: queue properties are implemented this way to better support bit
// manipulations
typedef pi_bitfield pi_queue_properties;
const pi_queue_properties PI_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE =
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
const pi_queue_properties PI_QUEUE_PROFILING_ENABLE = CL_QUEUE_PROFILING_ENABLE;
const pi_queue_properties PI_QUEUE_ON_DEVICE = CL_QUEUE_ON_DEVICE;
const pi_queue_properties PI_QUEUE_ON_DEVICE_DEFAULT =
        CL_QUEUE_ON_DEVICE_DEFAULT;

typedef _pi_result                  pi_result;
typedef _pi_platform_info           pi_platform_info;
typedef _pi_device_type             pi_device_type;
typedef _pi_device_info             pi_device_info;
typedef _pi_context_info            pi_context_info;
typedef _pi_queue_info              pi_queue_info;
typedef _pi_image_info              pi_image_info;
typedef _pi_mem_type                pi_mem_type;
typedef _pi_image_channel_order     pi_image_channel_order;
typedef _pi_image_channel_type      pi_image_channel_type;
typedef _pi_buffer_create_type      pi_buffer_create_type;
typedef _pi_sampler_addressing_mode pi_sampler_addressing_mode;
typedef _pi_sampler_filter_mode     pi_sampler_filter_mode;
typedef _pi_sampler_info            pi_sampler_info;

// Entry type, matches OpenMP for compatibility
struct _pi_offload_entry_struct {
  void *addr;
  char *name;
  size_t size;
  int32_t flags;
  int32_t reserved;
};

typedef _pi_offload_entry_struct * _pi_offload_entry;

/// Types of device binary.
typedef uint8_t pi_device_binary_type;
// format is not determined
static const pi_device_binary_type PI_DEVICE_BINARY_TYPE_NONE    = 0;
// specific to a device
static const pi_device_binary_type PI_DEVICE_BINARY_TYPE_NATIVE  = 1;
// portable binary types go next
// SPIR-V
static const pi_device_binary_type PI_DEVICE_BINARY_TYPE_SPIRV   = 2;
// LLVM bitcode
static const pi_device_binary_type PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE = 3;

// Device binary descriptor version supported by this library.
static const uint16_t PI_DEVICE_BINARY_VERSION = 1;

// The kind of offload model the binary employs; must be 4 for SYCL
static const uint8_t PI_DEVICE_BINARY_OFFLOAD_KIND_SYCL = 4;

/// Target identification strings for
/// pi_device_binary_struct.DeviceTargetSpec
///
/// A device type represented by a particular target
/// triple requires specific binary images. We need
/// to map the image type onto the device target triple
///
#define PI_DEVICE_BINARY_TARGET_UNKNOWN "<unknown>"
/// SPIR-V 32-bit image <-> "spir", 32-bit OpenCL device
#define PI_DEVICE_BINARY_TARGET_SPIRV32 "spir"
/// SPIR-V 64-bit image <-> "spir64", 64-bit OpenCL device
#define PI_DEVICE_BINARY_TARGET_SPIRV64 "spir64"
/// Device-specific binary images produced from SPIR-V 64-bit <->
/// various "spir64_*" triples for specific 64-bit OpenCL devices
#define PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64 "spir64_x86_64"
#define PI_DEVICE_BINARY_TARGET_SPIRV64_GEN "spir64_gen"
#define PI_DEVICE_BINARY_TARGET_SPIRV64_FPGA "spir64_fpga"

/// This struct is a record of the device binary information. If the Kind field
/// denotes a portable binary type (SPIR-V or LLVM IR), the DeviceTargetSpec field
/// can still be specific and denote e.g. FPGA target.
/// It must match the __tgt_device_image structure generated by
/// the clang-offload-wrapper tool when their Version field match.
struct pi_device_binary_struct {
  /// version of this structure - for backward compatibility;
  /// all modifications which change order/type/offsets of existing fields
  /// should increment the version.
  uint16_t Version;
  /// the type of offload model the binary employs; must be 4 for SYCL
  uint8_t Kind;
  /// format of the binary data - SPIR-V, LLVM IR bitcode,...
  uint8_t Format;
  /// null-terminated string representation of the device's target architecture
  /// which holds one of:
  /// PI_DEVICE_BINARY_TARGET_UNKNOWN - unknown
  /// PI_DEVICE_BINARY_TARGET_SPIRV32 - general value for 32-bit OpenCL devices
  /// PI_DEVICE_BINARY_TARGET_SPIRV64 - general value for 64-bit OpenCL devices
  /// PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64 - 64-bit OpenCL CPU device
  /// PI_DEVICE_BINARY_TARGET_SPIRV64_GEN - GEN GPU device (64-bit OpenCL)
  /// PI_DEVICE_BINARY_TARGET_SPIRV64_FPGA - 64-bit OpenCL FPGA device
  const char *DeviceTargetSpec;
  /// a null-terminated string; target- and compiler-specific options
  /// which are suggested to use to "build" program at runtime
  const char *BuildOptions;
  /// Pointer to the manifest data start
  const char *ManifestStart;
  /// Pointer to the manifest data end
  const char *ManifestEnd;
  /// Pointer to the target code start
  const unsigned char *BinaryStart;
  /// Pointer to the target code end
  const unsigned char *BinaryEnd;
  /// the offload entry table
  _pi_offload_entry EntriesBegin;
  _pi_offload_entry EntriesEnd;
};
typedef pi_device_binary_struct * pi_device_binary;

// Offload binaries descriptor version supported by this library.
static const uint16_t PI_DEVICE_BINARIES_VERSION = 1;

/// This struct is a record of all the device code that may be offloaded.
/// It must match the __tgt_bin_desc structure generated by
/// the clang-offload-wrapper tool when their Version field match.
struct pi_device_binaries_struct {
  /// version of this structure - for backward compatibility;
  /// all modifications which change order/type/offsets of existing fields
  /// should increment the version.
  uint16_t Version;
  /// Number of device binaries in this descriptor
  uint16_t NumDeviceBinaries;
  /// Device binaries data
  pi_device_binary DeviceBinaries;
  /// the offload entry table (not used, for compatibility with OpenMP)
  _pi_offload_entry *HostEntriesBegin;
  _pi_offload_entry *HostEntriesEnd;
};
typedef pi_device_binaries_struct *  pi_device_binaries;

// Opaque types that make reading build log errors easier.
struct _pi_platform;
struct _pi_device;
struct _pi_context;
struct _pi_queue;
struct _pi_mem;
struct _pi_program;
struct _pi_kernel;
struct _pi_event;
struct _pi_sampler;

typedef _pi_platform *    pi_platform;
typedef _pi_device *      pi_device;
typedef _pi_context *     pi_context;
typedef _pi_queue *       pi_queue;
typedef _pi_mem *         pi_mem;
typedef _pi_program *     pi_program;
typedef _pi_kernel *      pi_kernel;
typedef _pi_event *       pi_event;
typedef _pi_sampler *     pi_sampler;

typedef struct {
  pi_image_channel_order image_channel_order;
  pi_image_channel_type image_channel_data_type;
} _pi_image_format;

typedef struct {
  pi_mem_type image_type;
  size_t image_width;
  size_t image_height;
  size_t image_depth;
  size_t image_array_size;
  size_t image_row_pitch;
  size_t image_slice_pitch;
  pi_uint32 num_mip_levels;
  pi_uint32 num_samples;
  pi_mem buffer;
} _pi_image_desc;

typedef _pi_image_format   pi_image_format;
typedef _pi_image_desc     pi_image_desc;

//
// Following section contains SYCL RT Plugin Interface (PI) functions.
// They are 3 distinct categories:
//
// 1) Ones having direct analogy in OpenCL and needed for the core SYCL
//    functionality are started with just "pi" prefix in their names.
// 2) Those having direct analogy in OpenCL but only needed for SYCL
//    interoperability with OpenCL are started with "picl" prefix.
// 3) Functions having no direct analogy in OpenCL, started with "piext".
//
// TODO: describe interfaces in Doxygen format
//

struct _pi_plugin;
typedef _pi_plugin pi_plugin;

// PI Plugin Initialise.
// Plugin will check the PI version of Plugin Interface,
// populate the PI Version it supports, update targets field and populate
// PiFunctionTable with Supported APIs. The pointers are in a predetermined
// order in pi.def file.
pi_result piPluginInit(pi_plugin *plugin_info);

//
// Platform
//
pi_result piPlatformsGet(
  pi_uint32      num_entries,
  pi_platform *  platforms,
  pi_uint32 *    num_platforms);

pi_result piPlatformGetInfo(
  pi_platform       platform,
  pi_platform_info  param_name,
  size_t            param_value_size,
  void *            param_value,
  size_t *          param_value_size_ret);

//
// Device
//
pi_result piDevicesGet(
  pi_platform      platform,
  pi_device_type   device_type,
  pi_uint32        num_entries,
  pi_device *      devices,
  pi_uint32 *      num_devices);

pi_result piDeviceGetInfo(
  pi_device       device,
  pi_device_info  param_name,
  size_t          param_value_size,
  void *          param_value,
  size_t *        param_value_size_ret);

pi_result piDeviceRetain(pi_device device);

pi_result piDeviceRelease(pi_device device);

pi_result piDevicePartition(
  pi_device     device,
  const cl_device_partition_property * properties, // TODO: untie from OpenCL
  pi_uint32     num_devices,
  pi_device *   out_devices,
  pi_uint32 *   out_num_devices);

/// Selects the most appropriate device binary based on runtime information
/// and the IR characteristics.
///
pi_result piextDeviceSelectBinary(
  pi_device           device,
  pi_device_binary *  binaries,
  pi_uint32           num_binaries,
  pi_device_binary *  selected_binary);

/// Retrieves a device function pointer to a user-defined function
/// \arg \c function_name. \arg \c function_pointer_ret is set to 0 if query
/// failed.
///
/// \arg \c program must be built before calling this API. \arg \c device
/// must present in the list of devices returned by \c get_device method for
/// \arg \c program.
///
pi_result piextGetDeviceFunctionPointer(
  pi_device        device,
  pi_program       program,
  const char *     function_name,
  pi_uint64 *      function_pointer_ret);

//
// Context
//
pi_result piContextCreate(
  const cl_context_properties * properties, // TODO: untie from OpenCL
  pi_uint32         num_devices,
  const pi_device * devices,
  void (*           pfn_notify)(
    const char * errinfo,
    const void * private_info,
    size_t       cb,
    void *       user_data),
  void *            user_data,
  pi_context *      ret_context);

pi_result piContextGetInfo(
  pi_context         context,
  pi_context_info    param_name,
  size_t             param_value_size,
  void *             param_value,
  size_t *           param_value_size_ret);

pi_result piContextRetain(pi_context context);

pi_result piContextRelease(pi_context context);

//
// Queue
//
pi_result piQueueCreate(
  pi_context                  context,
  pi_device                   device,
  pi_queue_properties         properties,
  pi_queue *                  queue);

pi_result piQueueGetInfo(
  pi_queue            command_queue,
  pi_queue_info       param_name,
  size_t              param_value_size,
  void *              param_value,
  size_t *            param_value_size_ret);

pi_result piQueueRetain(pi_queue command_queue);

pi_result piQueueRelease(pi_queue command_queue);

pi_result piQueueFinish(pi_queue command_queue);

//
// Memory
//
pi_result piMemBufferCreate(
  pi_context   context,
  pi_mem_flags flags,
  size_t       size,
  void *       host_ptr,
  pi_mem *     ret_mem);

pi_result piMemImageCreate(
  pi_context              context,
  pi_mem_flags            flags,
  const pi_image_format * image_format,
  const pi_image_desc *   image_desc,
  void *                  host_ptr,
  pi_mem *                ret_mem);

pi_result piMemGetInfo(
  pi_mem           mem,
  cl_mem_info      param_name, // TODO: untie from OpenCL
  size_t           param_value_size,
  void *           param_value,
  size_t *         param_value_size_ret);

pi_result piMemImageGetInfo (
  pi_mem          image,
  pi_image_info   param_name,
  size_t          param_value_size,
  void *          param_value ,
  size_t *        param_value_size_ret);

pi_result piMemRetain(
  pi_mem mem);
 
pi_result piMemRelease(
  pi_mem mem);

pi_result piMemBufferPartition(
    pi_mem                    buffer,
    pi_mem_flags              flags,
    pi_buffer_create_type     buffer_create_type,
    void *                    buffer_create_info,
    pi_mem *                  ret_mem);
//
// Program
//
pi_result piProgramCreate(
  pi_context    context,
  const void *  il,
  size_t        length,
  pi_program *  res_program);

pi_result piclProgramCreateWithSource(
  pi_context        context,
  pi_uint32         count,
  const char **     strings,
  const size_t *    lengths,
  pi_program *      ret_program);

pi_result piclProgramCreateWithBinary(
  pi_context                     context,
  pi_uint32                      num_devices,
  const pi_device *              device_list,
  const size_t *                 lengths,
  const unsigned char **         binaries,
  pi_int32 *                     binary_status,
  pi_program *                   ret_program);

pi_result piProgramGetInfo(
  pi_program          program,
  cl_program_info     param_name, // TODO: untie from OpenCL
  size_t              param_value_size,
  void *              param_value,
  size_t *            param_value_size_ret);

pi_result piProgramLink(
  pi_context          context,
  pi_uint32           num_devices,
  const pi_device *   device_list,
  const char *        options,
  pi_uint32           num_input_programs,
  const pi_program *  input_programs,
  void (*  pfn_notify)(pi_program program,
                       void * user_data),
  void *              user_data,
  pi_program *        ret_program);

pi_result piProgramCompile(
  pi_program           program,
  pi_uint32            num_devices,
  const pi_device *    device_list,
  const char *         options,
  pi_uint32            num_input_headers,
  const pi_program *   input_headers,
  const char **        header_include_names,
  void (*  pfn_notify)(pi_program program, void * user_data),
  void *               user_data);

pi_result piProgramBuild(
  pi_program           program,
  pi_uint32            num_devices,
  const pi_device *    device_list,
  const char *         options,
  void (*  pfn_notify)(pi_program program, void * user_data),
  void *               user_data);

pi_result piProgramGetBuildInfo(
  pi_program              program,
  pi_device               device,
  cl_program_build_info   param_name, // TODO: untie from OpenCL
  size_t                  param_value_size,
  void *                  param_value,
  size_t *                param_value_size_ret);

pi_result piProgramRetain(pi_program program);

pi_result piProgramRelease(pi_program program);

//
// Kernel
//

typedef enum {
  /// indicates that the kernel might access data through USM ptrs
  PI_USM_INDIRECT_ACCESS,
  /// provides an explicit list of pointers that the kernel will access
  PI_USM_PTRS               = CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL
} _pi_kernel_exec_info;

typedef _pi_kernel_exec_info      pi_kernel_exec_info;

pi_result piKernelCreate(
  pi_program      program,
  const char *    kernel_name,
  pi_kernel *     ret_kernel);

pi_result piKernelSetArg(
  pi_kernel    kernel,
  pi_uint32    arg_index,
  size_t       arg_size,
  const void * arg_value);

pi_result piKernelGetInfo(
  pi_kernel       kernel,
  cl_kernel_info  param_name, // TODO: change to pi_kernel_info
  size_t          param_value_size,
  void *          param_value,
  size_t *        param_value_size_ret);

pi_result piKernelGetGroupInfo(
  pi_kernel                  kernel,
  pi_device                  device,
  cl_kernel_work_group_info  param_name, // TODO: untie from OpenCL
  size_t                     param_value_size,
  void *                     param_value,
  size_t *                   param_value_size_ret);

pi_result piKernelGetSubGroupInfo(
  pi_kernel                   kernel,
  pi_device                   device,
  cl_kernel_sub_group_info    param_name, // TODO: untie from OpenCL
  size_t                      input_value_size,
  const void*                 input_value,
  size_t                      param_value_size,
  void*                       param_value,
  size_t*                     param_value_size_ret);

pi_result piKernelRetain(pi_kernel    kernel);

pi_result piKernelRelease(pi_kernel    kernel);

/// Sets up pointer arguments for CL kernels. An extra indirection
/// is required due to CL argument conventions.
///
/// @param kernel is the kernel to be launched
/// @param arg_index is the index of the kernel argument
/// @param arg_size is the size in bytes of the argument (ignored in CL)
/// @param arg_value is the pointer argument
pi_result piextKernelSetArgPointer(
  pi_kernel    kernel,
  pi_uint32    arg_index,
  size_t       arg_size,
  const void * arg_value);

/// API to set attributes controlling kernel execution
///
/// @param kernel is the pi kernel to execute
/// @param param_name is a pi_kernel_exec_info value that specifies the info
///        passed to the kernel
/// @param param_value_size is the size of the value in bytes
/// @param param_value is a pointer to the value to set for the kernel
///
/// If param_name is PI_USM_INDIRECT_ACCESS, the value will be a ptr to
///    the pi_bool value PI_TRUE
/// If param_name is PI_USM_PTRS, the value will be an array of ptrs
pi_result piKernelSetExecInfo(pi_kernel kernel, pi_kernel_exec_info value_name,
                              size_t param_value_size, const void *param_value);

//
// Events
//
pi_result piEventCreate(
  pi_context    context,
  pi_event *    ret_event);

pi_result piEventGetInfo(
  pi_event         event,
  cl_event_info    param_name, // TODO: untie from OpenCL
  size_t           param_value_size,
  void *           param_value,
  size_t *         param_value_size_ret);

pi_result piEventGetProfilingInfo(
  pi_event            event,
  cl_profiling_info   param_name, // TODO: untie from OpenCL
  size_t              param_value_size,
  void *              param_value,
  size_t *            param_value_size_ret);

pi_result piEventsWait(
  pi_uint32           num_events,
  const pi_event *    event_list);

pi_result piEventSetCallback(
  pi_event    event,
  pi_int32    command_exec_callback_type,
  void (*     pfn_notify)(pi_event event,
                          pi_int32 event_command_status,
                          void *   user_data),
  void *      user_data);

pi_result piEventSetStatus(
  pi_event   event,
  pi_int32   execution_status);

pi_result piEventRetain(pi_event event);

pi_result piEventRelease(pi_event event);

//
// Sampler
//
pi_result piSamplerCreate(
  pi_context                     context,
  const pi_sampler_properties *  sampler_properties,
  pi_sampler *                   result_sampler);

pi_result piSamplerGetInfo(
  pi_sampler         sampler,
  pi_sampler_info    param_name,
  size_t             param_value_size,
  void *             param_value,
  size_t *           param_value_size_ret);

pi_result piSamplerRetain(pi_sampler sampler);

pi_result piSamplerRelease(pi_sampler sampler);

//
// Queue Commands
//
pi_result piEnqueueKernelLaunch(
  pi_queue          queue,
  pi_kernel         kernel,
  pi_uint32         work_dim,
  const size_t *    global_work_offset,
  const size_t *    global_work_size,
  const size_t *    local_work_size,
  pi_uint32         num_events_in_wait_list,
  const pi_event *  event_wait_list,
  pi_event *        event);

pi_result piEnqueueNativeKernel(
  pi_queue         queue,
  void             (*user_func)(void *),
  void *           args,
  size_t           cb_args,
  pi_uint32        num_mem_objects,
  const pi_mem *   mem_list,
  const void **    args_mem_loc,
  pi_uint32        num_events_in_wait_list,
  const pi_event * event_wait_list,
  pi_event *       event);

pi_result piEnqueueEventsWait(
  pi_queue          command_queue,
  pi_uint32         num_events_in_wait_list,
  const pi_event *  event_wait_list,
  pi_event *        event);

pi_result piEnqueueMemBufferRead(
  pi_queue            queue,
  pi_mem              buffer,
  pi_bool             blocking_read,
  size_t              offset,
  size_t              size,
  void *              ptr,
  pi_uint32           num_events_in_wait_list,
  const pi_event *    event_wait_list,
  pi_event *          event);

pi_result piEnqueueMemBufferReadRect(
  pi_queue            command_queue,
  pi_mem              buffer,
  pi_bool             blocking_read,
  const size_t *      buffer_offset,
  const size_t *      host_offset,
  const size_t *      region,
  size_t              buffer_row_pitch,
  size_t              buffer_slice_pitch,
  size_t              host_row_pitch,
  size_t              host_slice_pitch,
  void *              ptr,
  pi_uint32           num_events_in_wait_list,
  const pi_event *    event_wait_list,
  pi_event *          event);

pi_result piEnqueueMemBufferWrite(
  pi_queue           command_queue,
  pi_mem             buffer,
  pi_bool            blocking_write,
  size_t             offset,
  size_t             size,
  const void *       ptr,
  pi_uint32          num_events_in_wait_list,
  const pi_event *   event_wait_list,
  pi_event *         event);

pi_result piEnqueueMemBufferWriteRect(
  pi_queue            command_queue,
  pi_mem              buffer,
  pi_bool             blocking_write,
  const size_t *      buffer_offset,
  const size_t *      host_offset,
  const size_t *      region,
  size_t              buffer_row_pitch,
  size_t              buffer_slice_pitch,
  size_t              host_row_pitch,
  size_t              host_slice_pitch,
  const void *        ptr,
  pi_uint32           num_events_in_wait_list,
  const pi_event *    event_wait_list,
  pi_event *          event);

pi_result piEnqueueMemBufferCopy(
  pi_queue            command_queue,
  pi_mem              src_buffer,
  pi_mem              dst_buffer,
  size_t              src_offset,
  size_t              dst_offset,
  size_t              size,
  pi_uint32           num_events_in_wait_list,
  const pi_event *    event_wait_list,
  pi_event *          event);

pi_result piEnqueueMemBufferCopyRect(
  pi_queue            command_queue,
  pi_mem              src_buffer,
  pi_mem              dst_buffer,
  const size_t *      src_origin,
  const size_t *      dst_origin,
  const size_t *      region,
  size_t              src_row_pitch,
  size_t              src_slice_pitch,
  size_t              dst_row_pitch,
  size_t              dst_slice_pitch,
  pi_uint32           num_events_in_wait_list,
  const pi_event *    event_wait_list,
  pi_event *          event);

pi_result piEnqueueMemBufferFill(
  pi_queue           command_queue,
  pi_mem             buffer,
  const void *       pattern,
  size_t             pattern_size,
  size_t             offset,
  size_t             size,
  pi_uint32          num_events_in_wait_list,
  const pi_event *   event_wait_list,
  pi_event *         event);

pi_result piEnqueueMemImageRead(
  pi_queue          command_queue,
  pi_mem            image,
  pi_bool           blocking_read,
  const size_t *    origin,
  const size_t *    region,
  size_t            row_pitch,
  size_t            slice_pitch,
  void *            ptr,
  pi_uint32         num_events_in_wait_list,
  const pi_event *  event_wait_list,
  pi_event *        event);

pi_result piEnqueueMemImageWrite(
  pi_queue          command_queue,
  pi_mem            image,
  pi_bool           blocking_write,
  const size_t *    origin,
  const size_t *    region,
  size_t            input_row_pitch,
  size_t            input_slice_pitch,
  const void *      ptr,
  pi_uint32         num_events_in_wait_list,
  const pi_event *  event_wait_list,
  pi_event *        event);

pi_result piEnqueueMemImageCopy(
  pi_queue          command_queue,
  pi_mem            src_image,
  pi_mem            dst_image,
  const size_t *    src_origin,
  const size_t *    dst_origin,
  const size_t *    region,
  pi_uint32         num_events_in_wait_list,
  const pi_event *  event_wait_list,
  pi_event *        event);

pi_result piEnqueueMemImageFill(
  pi_queue          command_queue,
  pi_mem            image,
  const void *      fill_color,
  const size_t *    origin,
  const size_t *    region,
  pi_uint32         num_events_in_wait_list,
  const pi_event *  event_wait_list,
  pi_event *        event);

pi_result piEnqueueMemBufferMap(
  pi_queue          command_queue,
  pi_mem            buffer,
  pi_bool           blocking_map,
  cl_map_flags      map_flags,  // TODO: untie from OpenCL
  size_t            offset,
  size_t            size,
  pi_uint32         num_events_in_wait_list,
  const pi_event *  event_wait_list,
  pi_event *        event,
  void* *           ret_map);

pi_result piEnqueueMemUnmap(
  pi_queue         command_queue,
  pi_mem           memobj,
  void *           mapped_ptr,
  pi_uint32        num_events_in_wait_list,
  const pi_event * event_wait_list,
  pi_event *       event);

///
// USM
///
typedef enum {
  PI_USM_HOST_SUPPORT          = CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL,
  PI_USM_DEVICE_SUPPORT        = CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL,
  PI_USM_SINGLE_SHARED_SUPPORT = CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
  PI_USM_CROSS_SHARED_SUPPORT  = CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
  PI_USM_SYSTEM_SHARED_SUPPORT = CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL
} _pi_usm_capability_query;

typedef enum : pi_bitfield {
  PI_USM_ACCESS                   = CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL,
  PI_USM_ATOMIC_ACCESS            = CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL,
  PI_USM_CONCURRENT_ACCESS        = CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL,
  PI_USM_CONCURRENT_ATOMIC_ACCESS = CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL
} _pi_usm_capabilities;

typedef enum {
  PI_MEM_ALLOC_TYPE        = CL_MEM_ALLOC_TYPE_INTEL,
  PI_MEM_ALLOC_BASE_PTR    = CL_MEM_ALLOC_BASE_PTR_INTEL,
  PI_MEM_ALLOC_SIZE        = CL_MEM_ALLOC_SIZE_INTEL,
  PI_MEM_ALLOC_DEVICE      = CL_MEM_ALLOC_DEVICE_INTEL,
  PI_MEM_ALLOC_INFO_TBD0   = CL_MEM_ALLOC_INFO_TBD0_INTEL,
  PI_MEM_ALLOC_INFO_TBD1   = CL_MEM_ALLOC_INFO_TBD1_INTEL,
} _pi_mem_info;

typedef enum {
  PI_MEM_TYPE_UNKNOWN = CL_MEM_TYPE_UNKNOWN_INTEL,
  PI_MEM_TYPE_HOST    = CL_MEM_TYPE_HOST_INTEL,
  PI_MEM_TYPE_DEVICE  = CL_MEM_TYPE_DEVICE_INTEL,
  PI_MEM_TYPE_SHARED  = CL_MEM_TYPE_SHARED_INTEL
} _pi_usm_type;

typedef enum : pi_bitfield  {
  PI_MEM_ALLOC_FLAGS = CL_MEM_ALLOC_FLAGS_INTEL
} _pi_usm_mem_properties;

typedef enum : pi_bitfield {
  PI_USM_MIGRATION_TBD0 = (1 << 0)
} _pi_usm_migration_flags;

typedef _pi_usm_capability_query  pi_usm_capability_query;
typedef _pi_usm_capabilities      pi_usm_capabilities;
typedef _pi_mem_info              pi_mem_info;
typedef _pi_usm_type              pi_usm_type;
typedef _pi_usm_mem_properties    pi_usm_mem_properties;
typedef _pi_usm_migration_flags   pi_usm_migration_flags;

/// Allocates host memory accessible by the device.
///
/// @param result_ptr contains the allocated memory
/// @param context is the pi_context
/// @param pi_usm_mem_properties are optional allocation properties
/// @param size_t is the size of the allocation
/// @param alignment is the desired alignment of the allocation
pi_result piextUSMHostAlloc(
  void **                 result_ptr,
  pi_context              context,
  pi_usm_mem_properties * properties,
  size_t                  size,
  pi_uint32               alignment);

/// Allocates device memory
///
/// @param result_ptr contains the allocated memory
/// @param context is the pi_context
/// @param device is the device the memory will be allocated on
/// @param pi_usm_mem_properties are optional allocation properties
/// @param size_t is the size of the allocation
/// @param alignment is the desired alignment of the allocation
pi_result piextUSMDeviceAlloc(
  void **                 result_ptr,
  pi_context              context,
  pi_device               device,
  pi_usm_mem_properties * properties,
  size_t                  size,
  pi_uint32               alignment);

/// Allocates memory accessible on both host and device
///
/// @param result_ptr contains the allocated memory
/// @param context is the pi_context
/// @param device is the device the memory will be allocated on
/// @param pi_usm_mem_properties are optional allocation properties
/// @param size_t is the size of the allocation
/// @param alignment is the desired alignment of the allocation
pi_result piextUSMSharedAlloc(
  void **                 result_ptr,
  pi_context              context,
  pi_device               device,
  pi_usm_mem_properties * properties,
  size_t                  size,
  pi_uint32               alignment);

/// Frees allocated USM memory
///
/// @param context is the pi_context of the allocation
/// @param ptr is the memory to be freed
pi_result piextUSMFree(
  pi_context context,
  void *     ptr);

/// USM Memset API
///
/// @param queue is the queue to submit to
/// @param ptr is the ptr to memset
/// @param value is value to set.  It is interpreted as an 8-bit value and the upper
///        24 bits are ignored
/// @param count is the size in bytes to memset
/// @param num_events_in_waitlist is the number of events to wait on
/// @param events_waitlist is an array of events to wait on
/// @param event is the event that represents this operation
pi_result piextUSMEnqueueMemset(
  pi_queue         queue,
  void *           ptr,
  pi_int32         value,
  size_t           count,
  pi_uint32        num_events_in_waitlist,
  const pi_event * events_waitlist,
  pi_event *       event);

/// USM Memcpy API
///
/// @param queue is the queue to submit to
/// @param blocking is whether this operation should block the host
/// @param src_ptr is the data to be copied
/// @param dst_ptr is the location the data will be copied
/// @param size is number of bytes to copy
/// @param num_events_in_waitlist is the number of events to wait on
/// @param events_waitlist is an array of events to wait on
/// @param event is the event that represents this operation
pi_result piextUSMEnqueueMemcpy(
  pi_queue         queue,
  pi_bool          blocking,
  void *           dst_ptr,
  const void *     src_ptr,
  size_t           size,
  pi_uint32        num_events_in_waitlist,
  const pi_event * events_waitlist,
  pi_event *       event);

/// Hint to migrate memory to the device
///
/// @param queue is the queue to submit to
/// @param ptr points to the memory to migrate
/// @param size is the number of bytes to migrate
/// @param flags is a bitfield used to specify memory migration options
/// @param num_events_in_waitlist is the number of events to wait on
/// @param events_waitlist is an array of events to wait on
/// @param event is the event that represents this operation
pi_result piextUSMEnqueuePrefetch(
  pi_queue               queue,
  const void *           ptr,
  size_t                 size,
  pi_usm_migration_flags flags,
  pi_uint32              num_events_in_waitlist,
  const pi_event *       events_waitlist,
  pi_event *             event);

/// USM Memadvise API
///
/// @param queue is the queue to submit to
/// @param ptr is the data to be advised
/// @param length is the size in bytes of the memory to advise
/// @param advice is device specific advice
/// @param event is the event that represents this operation
// USM memadvise API to govern behavior of automatic migration mechanisms
pi_result piextUSMEnqueueMemAdvise(
  pi_queue     queue,
  const void * ptr,
  size_t       length,
  int          advice,
  pi_event *   event);

/// API to query information about USM allocated pointers
/// Valid Queries:
///   PI_MEM_ALLOC_TYPE returns host/device/shared pi_host_usm value
///   PI_MEM_ALLOC_BASE_PTR returns the base ptr of an allocation if
///                         the queried pointer fell inside an allocation.
///                         Result must fit in void *
///   PI_MEM_ALLOC_SIZE returns how big the queried pointer's
///                     allocation is in bytes. Result is a size_t.
///   PI_MEM_ALLOC_DEVICE returns the pi_device this was allocated against
///
/// @param context is the pi_context
/// @param ptr is the pointer to query
/// @param param_name is the type of query to perform
/// @param param_value_size is the size of the result in bytes
/// @param param_value is the result
/// @param param_value_ret is how many bytes were written
pi_result piextUSMGetMemAllocInfo(
  pi_context   context,
  const void * ptr,
  pi_mem_info  param_name,
  size_t       param_value_size,
  void *       param_value,
  size_t *     param_value_size_ret);

struct _pi_plugin {
  // PI version supported by host passed to the plugin. The Plugin
  // checks and writes the appropriate Function Pointers in
  // PiFunctionTable.
  // TODO: Work on version fields and their handshaking mechanism.
  // Some choices are:
  // - Use of integers to keep major and minor version.
  // - Keeping char* Versions.
  const char PiVersion[4] = _PI_H_VERSION_STRING;
  // Plugin edits this.
  char PluginVersion[4] = _PI_H_VERSION_STRING;
  char *Targets;
  struct FunctionPointers {
#define _PI_API(api) decltype(::api) *api;
#include <CL/sycl/detail/pi.def>
  } PiFunctionTable;
};

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // _PI_H_
