#pragma once

#include <CL/sycl/detail/common.hpp>

#include <xpti/xpti_data_types.h>
#include <xpti/xpti_trace_framework.hpp>

#include <string_view>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
struct MetadataAccessMode {
  using type = int;
  static constexpr auto name = "access_mode";
};
struct MetadataAccessRangeStart {
  using type = size_t;
  static constexpr auto name = "access_range_start";
};
struct MetadataAccessRangeEnd {
  using type = size_t;
  static constexpr auto name = "access_range_end";
};
struct MetadataAllocationType {
  using type = const std::string_view;
  static constexpr auto name = "allocation_type";
};
struct MetadataCopyFrom {
  using type = size_t;
  static constexpr auto name = "copy_from";
};
struct MetadataCopyTo {
  using type = size_t;
  static constexpr auto name = "copy_to";
};
struct MetadataEvent {
  using type = size_t;
  static constexpr auto name = "event";
};
struct MetadataFromSource {
  using type = bool;
  static constexpr auto name = "from_source";
};
struct MetadataKernelName {
  using type = const std::string_view;
  static constexpr auto name = "kernel_name";
};
struct MetadataMemoryObject {
  using type = size_t;
  static constexpr auto name = "memory_object";
};
struct MetadataOffset {
  using type = size_t;
  static constexpr auto name = "offset";
};
struct MetadataDevice {
  using type = size_t;
  static constexpr auto name = "sycl_device";
};

struct MetadataDeviceType {
  using type = const std::string_view;
  static constexpr auto name = "sycl_device_type";
};

struct MetadataDeviceName {
  using type = const std::string_view;
  static constexpr auto name = "sycl_device_name";
};
struct MetadataFunctionName {
  using type = const std::string_view;
  static constexpr auto name = "sym_function_name";
};
struct MetadataSourceFileName {
  using type = const std::string_view;
  static constexpr auto name = "sym_source_file_name";
};
struct MetadataLineNo {
  using type = int32_t;
  static constexpr auto name = "sym_line_no";
};
struct MetadataColumnNo {
  using type = int32_t;
  static constexpr auto name = "sym_column_no";
};
} // namespace detail

namespace ext {
namespace oneapi {
namespace experimental {
namespace tracing {
inline constexpr sycl::detail::MetadataAccessMode metadata_access_mode;
inline constexpr sycl::detail::MetadataAccessRangeStart
    metadata_access_range_start;
inline constexpr sycl::detail::MetadataAccessRangeEnd metadata_access_range_end;
inline constexpr sycl::detail::MetadataAllocationType metadata_allocation_type;
inline constexpr sycl::detail::MetadataCopyFrom metadata_copy_from;
inline constexpr sycl::detail::MetadataCopyTo metadata_copy_to;
inline constexpr sycl::detail::MetadataEvent metadata_event;
inline constexpr sycl::detail::MetadataFromSource metadata_from_source;
inline constexpr sycl::detail::MetadataKernelName metadata_kernel_name;
inline constexpr sycl::detail::MetadataMemoryObject metadata_memory_object;
inline constexpr sycl::detail::MetadataOffset metadata_offset;
inline constexpr sycl::detail::MetadataDevice metadata_device;
inline constexpr sycl::detail::MetadataDeviceType metadata_device_type;
inline constexpr sycl::detail::MetadataDeviceName metadata_device_name;
inline constexpr sycl::detail::MetadataFunctionName metadata_function_name;
inline constexpr sycl::detail::MetadataSourceFileName metadata_source_file_name;
inline constexpr sycl::detail::MetadataLineNo metadata_line_no;
inline constexpr sycl::detail::MetadataColumnNo metadata_column_no;
} // namespace tracing
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
