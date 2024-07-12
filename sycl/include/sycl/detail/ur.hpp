//==---------- ur.hpp - Unified Runtime integration helpers ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
///
/// C++ utilities for Unified Runtime integration.
///
/// \ingroup sycl_ur

#pragma once

#include <sycl/backend_types.hpp>
#include <sycl/detail/export.hpp>
#include <ur_api.h>

#include <memory>
#include <type_traits>
#include <vector>

// Entry type, matches OpenMP for compatibility
struct _pi_offload_entry_struct {
  void *addr;
  char *name;
  size_t size;
  int32_t flags;
  int32_t reserved;
};

using _pi_offload_entry = _pi_offload_entry_struct *;

// A type of a binary image property.
typedef enum {
  PI_PROPERTY_TYPE_UNKNOWN,
  PI_PROPERTY_TYPE_UINT32,     // 32-bit integer
  PI_PROPERTY_TYPE_BYTE_ARRAY, // byte array
  PI_PROPERTY_TYPE_STRING      // null-terminated string
} pi_property_type;

// Device binary image property.
// If the type size of the property value is fixed and is no greater than
// 64 bits, then ValAddr is 0 and the value is stored in the ValSize field.
// Example - PI_PROPERTY_TYPE_UINT32, which is 32-bit
struct _pi_device_binary_property_struct {
  char *Name;       // null-terminated property name
  void *ValAddr;    // address of property value
  uint32_t Type;    // _pi_property_type
  uint64_t ValSize; // size of property value in bytes
};

typedef _pi_device_binary_property_struct *pi_device_binary_property;

// Named array of properties.
struct _pi_device_binary_property_set_struct {
  char *Name;                                // the name
  pi_device_binary_property PropertiesBegin; // array start
  pi_device_binary_property PropertiesEnd;   // array end
};

typedef _pi_device_binary_property_set_struct *pi_device_binary_property_set;

/// Types of device binary.
using pi_device_binary_type = uint8_t;
// format is not determined
static constexpr pi_device_binary_type PI_DEVICE_BINARY_TYPE_NONE = 0;
// specific to a device
static constexpr pi_device_binary_type PI_DEVICE_BINARY_TYPE_NATIVE = 1;
// portable binary types go next
// SPIR-V
static constexpr pi_device_binary_type PI_DEVICE_BINARY_TYPE_SPIRV = 2;
// LLVM bitcode
static constexpr pi_device_binary_type PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE = 3;

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
#define __SYCL_PI_DEVICE_BINARY_TARGET_UNKNOWN "<unknown>"
/// SPIR-V 32-bit image <-> "spir", 32-bit OpenCL device
#define __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV32 "spir"
/// SPIR-V 64-bit image <-> "spir64", 64-bit OpenCL device
#define __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64 "spir64"
/// Device-specific binary images produced from SPIR-V 64-bit <->
/// various "spir64_*" triples for specific 64-bit OpenCL devices
#define __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64 "spir64_x86_64"
#define __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_GEN "spir64_gen"
#define __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_FPGA "spir64_fpga"
/// PTX 64-bit image <-> "nvptx64", 64-bit NVIDIA PTX device
#define __SYCL_PI_DEVICE_BINARY_TARGET_NVPTX64 "nvptx64"
#define __SYCL_PI_DEVICE_BINARY_TARGET_AMDGCN "amdgcn"
#define __SYCL_PI_DEVICE_BINARY_TARGET_NATIVE_CPU "native_cpu"

/// Extension to denote native support of assert feature by an arbitrary device
/// piDeviceGetInfo call should return this extension when the device supports
/// native asserts if supported extensions' names are requested
#define PI_DEVICE_INFO_EXTENSION_DEVICELIB_ASSERT "cl_intel_devicelib_assert"

/// Device binary image property set names recognized by the SYCL runtime.
/// Name must be consistent with
/// PropertySetRegistry::SYCL_SPECIALIZATION_CONSTANTS defined in
/// PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SPEC_CONST_MAP "SYCL/specialization constants"
/// PropertySetRegistry::SYCL_SPEC_CONSTANTS_DEFAULT_VALUES defined in
/// PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SPEC_CONST_DEFAULT_VALUES_MAP                   \
  "SYCL/specialization constants default values"
/// PropertySetRegistry::SYCL_DEVICELIB_REQ_MASK defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_DEVICELIB_REQ_MASK "SYCL/devicelib req mask"
/// PropertySetRegistry::SYCL_KERNEL_PARAM_OPT_INFO defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_KERNEL_PARAM_OPT_INFO "SYCL/kernel param opt"
/// PropertySetRegistry::SYCL_KERNEL_PROGRAM_METADATA defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_PROGRAM_METADATA "SYCL/program metadata"
/// PropertySetRegistry::SYCL_MISC_PROP defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SYCL_MISC_PROP "SYCL/misc properties"
/// PropertySetRegistry::SYCL_ASSERT_USED defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SYCL_ASSERT_USED "SYCL/assert used"
/// PropertySetRegistry::SYCL_EXPORTED_SYMBOLS defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS "SYCL/exported symbols"
/// PropertySetRegistry::SYCL_DEVICE_GLOBALS defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SYCL_DEVICE_GLOBALS "SYCL/device globals"
/// PropertySetRegistry::SYCL_DEVICE_REQUIREMENTS defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS                        \
  "SYCL/device requirements"
/// PropertySetRegistry::SYCL_HOST_PIPES defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SYCL_HOST_PIPES "SYCL/host pipes"
/// PropertySetRegistry::SYCL_VIRTUAL_FUNCTIONS defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SYCL_VIRTUAL_FUNCTIONS "SYCL/virtual functions"

/// Program metadata tags recognized by the PI backends. For kernels the tag
/// must appear after the kernel name.
#define __SYCL_PI_PROGRAM_METADATA_TAG_REQD_WORK_GROUP_SIZE                    \
  "@reqd_work_group_size"
#define __SYCL_PI_PROGRAM_METADATA_GLOBAL_ID_MAPPING "@global_id_mapping"

#define __SYCL_PI_PROGRAM_METADATA_TAG_NEED_FINALIZATION "Requires finalization"

/// This struct is a record of the device binary information. If the Kind field
/// denotes a portable binary type (SPIR-V or LLVM IR), the DeviceTargetSpec
/// field can still be specific and denote e.g. FPGA target. It must match the
/// __tgt_device_image structure generated by the clang-offload-wrapper tool
/// when their Version field match.
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
  /// __SYCL_PI_DEVICE_BINARY_TARGET_UNKNOWN - unknown
  /// __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV32 - general value for 32-bit OpenCL
  /// devices
  /// __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64 - general value for 64-bit OpenCL
  /// devices
  /// __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64 - 64-bit OpenCL CPU device
  /// __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_GEN - GEN GPU device (64-bit
  /// OpenCL)
  /// __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_FPGA - 64-bit OpenCL FPGA device
  const char *DeviceTargetSpec;
  /// a null-terminated string; target- and compiler-specific options
  /// which are suggested to use to "compile" program at runtime
  const char *CompileOptions;
  /// a null-terminated string; target- and compiler-specific options
  /// which are suggested to use to "link" program at runtime
  const char *LinkOptions;
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
  // Array of preperty sets; e.g. specialization constants symbol-int ID map is
  // propagated to runtime with this mechanism.
  pi_device_binary_property_set PropertySetsBegin;
  pi_device_binary_property_set PropertySetsEnd;
  // TODO Other fields like entries, link options can be propagated using
  // the property set infrastructure. This will improve binary compatibility and
  // add flexibility.
};
using pi_device_binary = pi_device_binary_struct *;

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
using pi_device_binaries = pi_device_binaries_struct *;

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Forward declarations
namespace xpti {
struct trace_event_data_t;
}
#endif

namespace sycl {
inline namespace _V1 {

class context;

namespace detail {

class plugin;
using PluginPtr = std::shared_ptr<plugin>;

template <sycl::backend BE>
__SYCL_EXPORT void *getPluginOpaqueData(void *opaquedata_arg);

namespace ur {

__SYCL_EXPORT void contextSetExtendedDeleter(const sycl::context &constext,
                                             ur_context_extended_deleter_t func,
                                             void *user_data);

// Function to load a shared library
// Implementation is OS dependent
void *loadOsLibrary(const std::string &Library);

// Function to unload a shared library
// Implementation is OS dependent (see posix-pi.cpp and windows-pi.cpp)
int unloadOsLibrary(void *Library);

// Function to get Address of a symbol defined in the shared
// library, implementation is OS dependent.
void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName);

// Performs UR one-time initialization.
std::vector<PluginPtr> &initializeUr();

// Get the plugin serving given backend.
template <backend BE> __SYCL_EXPORT const PluginPtr &getPlugin();

/// Tries to determine the device binary image foramat. Returns
/// PI_DEVICE_BINARY_TYPE_NONE if unsuccessful.
pi_device_binary_type getBinaryImageFormat(const unsigned char *ImgData,
                                           size_t ImgSize);

// Return true if we want to trace UR related activities.
bool trace();

// Report error and no return (keeps compiler happy about no return statements).
[[noreturn]] __SYCL_EXPORT void die(const char *Message);

__SYCL_EXPORT void assertion(bool Condition, const char *Message = nullptr);

// Want all the needed casts be explicit, do not define conversion operators.
template <class To, class From> To cast(From value);

// Want all the needed casts be explicit, do not define conversion
// operators.
template <class To, class From> inline To cast(From value) {
  // TODO: see if more sanity checks are possible.
  assertion(sizeof(From) == sizeof(To), "assert: cast failed size check");
  return (To)(value);
}

// Helper traits for identifying std::vector with arbitrary element type.
template <typename T> struct IsStdVector : std::false_type {};
template <typename T> struct IsStdVector<std::vector<T>> : std::true_type {};

// Overload for vectors that applies the cast to all elements. This
// creates a new vector.
template <class To, class FromE> To cast(std::vector<FromE> Values) {
  static_assert(IsStdVector<To>::value, "Return type must be a vector.");
  To ResultVec;
  ResultVec.reserve(Values.size());
  for (FromE &Val : Values) {
    ResultVec.push_back(cast<typename To::value_type>(Val));
  }
  return ResultVec;
}

} // namespace ur
} // namespace detail
} // namespace _V1
} // namespace sycl
