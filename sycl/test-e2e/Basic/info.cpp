// RUN: %{build} -D__SYCL_INTERNAL_API -o %t.out
// RUN: %{run} %t.out

//==----------------info.cpp - SYCL objects get_info() test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/detail/core.hpp>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

using namespace sycl;

// Handle unknown info, e.g., from non-standard extensions.
template <typename T> std::string unknown_info_to_string(T info) {
  std::stringstream stream;
  stream << std::hex << static_cast<std::uint64_t>(info) << " (unknown value)"
         << "\n";
  return stream.str();
}

template <typename T> std::string info_to_string(T info) {
  return std::to_string(info);
}

template <> std::string info_to_string(std::string info) {
  if (info.empty()) {
    return "none";
  }
  return info;
}

template <> std::string info_to_string(bool info) {
  if (info) {
    return "true";
  }
  return "false";
}
template <> std::string info_to_string(info::device_type info) {
  switch (info) {
  case info::device_type::cpu:
    return "cpu";
  case info::device_type::gpu:
    return "gpu";
  case info::device_type::accelerator:
    return "accelerator";
  case info::device_type::custom:
    return "custom";
  case info::device_type::automatic:
    return "automatic";
  case info::device_type::all:
    return "all";
  default:
    return unknown_info_to_string(info);
  }
}

template <> std::string info_to_string(info::fp_config info) {
  switch (info) {
  case info::fp_config::denorm:
    return "denorm";
  case info::fp_config::inf_nan:
    return "inf_nan";
  case info::fp_config::round_to_nearest:
    return "round_to_nearest";
  case info::fp_config::round_to_zero:
    return "round_to_zero";
  case info::fp_config::round_to_inf:
    return "round_to_inf";
  case info::fp_config::fma:
    return "fma";
  case info::fp_config::correctly_rounded_divide_sqrt:
    return "correctly_rounded_divide_sqrt";
  case info::fp_config::soft_float:
    return "soft_float";
  default:
    return unknown_info_to_string(info);
  }
}

template <> std::string info_to_string(info::global_mem_cache_type info) {
  switch (info) {
  case info::global_mem_cache_type::none:
    return "none";
  case info::global_mem_cache_type::read_only:
    return "read_only";
  case info::global_mem_cache_type::read_write:
    return "read_write";
  default:
    return unknown_info_to_string(info);
  }
}

template <> std::string info_to_string(info::local_mem_type info) {
  switch (info) {
  case info::local_mem_type::none:
    return "none";
  case info::local_mem_type::local:
    return "local";
  case info::local_mem_type::global:
    return "global";
  default:
    return unknown_info_to_string(info);
  }
}

template <> std::string info_to_string(info::execution_capability info) {
  switch (info) {
  case info::execution_capability::exec_kernel:
    return "exec_kernel";
  case info::execution_capability::exec_native_kernel:
    return "exec_native_kernel";
  default:
    return unknown_info_to_string(info);
  }
}

template <> std::string info_to_string(info::partition_property info) {
  switch (info) {
  case info::partition_property::no_partition:
    return "no_partition";
  case info::partition_property::partition_equally:
    return "partition_equally";
  case info::partition_property::partition_by_counts:
    return "partition_by_counts";
  case info::partition_property::partition_by_affinity_domain:
    return "partition_by_affinity_domain";
  default:
    return unknown_info_to_string(info);
  }
}

template <> std::string info_to_string(info::partition_affinity_domain info) {
  switch (info) {
  case info::partition_affinity_domain::not_applicable:
    return "not_applicable";
  case info::partition_affinity_domain::numa:
    return "numa";
  case info::partition_affinity_domain::L4_cache:
    return "L4_cache";
  case info::partition_affinity_domain::L3_cache:
    return "L3_cache";
  case info::partition_affinity_domain::L2_cache:
    return "L2_cache";
  case info::partition_affinity_domain::L1_cache:
    return "L1_cache";
  case info::partition_affinity_domain::next_partitionable:
    return "next_partitionable";
  default:
    return unknown_info_to_string(info);
  }
}

template <> std::string info_to_string(platform info) {
  return "SYCL OpenCL platform";
}

template <> std::string info_to_string(device info) {
  return "SYCL OpenCL device";
}

template <int Dim> std::string info_to_string(range<Dim> info) {
  std::string str;
  for (size_t i = 0; i < Dim; ++i) {
    str += info_to_string(info[i]) + " ";
  }
  return str;
}

template <typename T> std::string info_to_string(std::vector<T> info) {
  if (info.empty()) {
    return "none";
  }
  std::string str;
  for (const auto &x : info) {
    str += info_to_string(x) + " ";
  }
  return str;
}

template <typename Param, typename ExpectedReturnT>
void print_info(const device &dev, const std::string &name) {
  static_assert(
      std::is_same<typename Param::return_type, ExpectedReturnT>::value,
      "Unexpected info query return type");
  ExpectedReturnT result = dev.get_info<Param>();
  std::cout << name << ": " << info_to_string(result) << std::endl;
}

template <typename Param, typename ExpectedReturnT>
void print_info(const platform &plt, const std::string &name) {
  static_assert(
      std::is_same<typename Param::return_type, ExpectedReturnT>::value,
      "Unexpected info query return type");
  ExpectedReturnT result(plt.get_info<Param>());
  std::cout << name << ": " << info_to_string(result) << std::endl;
}

int main() {
  std::string separator(std::string(80, '-') + "\n");
  std::cout << separator << "Device information\n" << separator;
  device dev(default_selector_v);

  print_info<info::device::device_type, info::device_type>(dev, "Device type");
  print_info<info::device::vendor_id, std::uint32_t>(dev, "Vendor ID");
  print_info<info::device::max_compute_units, std::uint32_t>(
      dev, "Max compute units");
  print_info<info::device::max_work_item_dimensions, std::uint32_t>(
      dev, "Max work item dimensions");
  print_info<info::device::max_work_item_sizes<1>, range<1>>(
      dev, "Max work item sizes 1D");
  print_info<info::device::max_work_item_sizes<2>, range<2>>(
      dev, "Max work item sizes 2D");
  print_info<info::device::max_work_item_sizes<3>, range<3>>(
      dev, "Max work item sizes 3D");
  print_info<info::device::max_work_group_size, size_t>(dev,
                                                        "Max work group size");
  print_info<info::device::preferred_vector_width_char, std::uint32_t>(
      dev, "Preferred vector width char");
  print_info<info::device::preferred_vector_width_short, std::uint32_t>(
      dev, "Preferred vector width short");
  print_info<info::device::preferred_vector_width_int, std::uint32_t>(
      dev, "Preferred vector width int");
  print_info<info::device::preferred_vector_width_long, std::uint32_t>(
      dev, "Preferred vector width long");
  print_info<info::device::preferred_vector_width_float, std::uint32_t>(
      dev, "Preferred vector width float");
  print_info<info::device::preferred_vector_width_double, std::uint32_t>(
      dev, "Preferred vector width double");
  print_info<info::device::preferred_vector_width_half, std::uint32_t>(
      dev, "Preferred vector width half");
  print_info<info::device::native_vector_width_char, std::uint32_t>(
      dev, "Native vector width char");
  print_info<info::device::native_vector_width_short, std::uint32_t>(
      dev, "Native vector width short");
  print_info<info::device::native_vector_width_int, std::uint32_t>(
      dev, "Native vector width int");
  print_info<info::device::native_vector_width_long, std::uint32_t>(
      dev, "Native vector width long");
  print_info<info::device::native_vector_width_float, std::uint32_t>(
      dev, "Native vector width float");
  print_info<info::device::native_vector_width_double, std::uint32_t>(
      dev, "Native vector width double");
  print_info<info::device::native_vector_width_half, std::uint32_t>(
      dev, "Native vector width half");
  /*TODO: uncomment when problem with frequency detection is fixed
  print_info<info::device::max_clock_frequency, std::uint32_t>(
      dev, "Max clock frequency");*/
  print_info<info::device::address_bits, std::uint32_t>(dev, "Address bits");
  print_info<info::device::max_mem_alloc_size, std::uint64_t>(
      dev, "Max mem alloc size");
  print_info<info::device::image_support, bool>(dev, "Image support");
  print_info<info::device::max_read_image_args, std::uint32_t>(
      dev, "Max read image args");
  print_info<info::device::max_write_image_args, std::uint32_t>(
      dev, "Max write image args");
  print_info<info::device::image2d_max_width, size_t>(dev, "Image2D max width");
  print_info<info::device::image2d_max_height, size_t>(dev,
                                                       "Image2D max height");
  print_info<info::device::image3d_max_width, size_t>(dev, "Image3D max width");
  print_info<info::device::image3d_max_height, size_t>(dev,
                                                       "Image3D max height");
  print_info<info::device::image3d_max_depth, size_t>(dev, "Image3D max depth");
  print_info<info::device::image_max_buffer_size, size_t>(
      dev, "Image max buffer size");
  print_info<info::device::image_max_array_size, size_t>(
      dev, "Image max array size");
  print_info<info::device::max_samplers, std::uint32_t>(dev, "Max samplers");
  print_info<info::device::max_parameter_size, size_t>(dev,
                                                       "Max parameter size");
  print_info<info::device::mem_base_addr_align, std::uint32_t>(
      dev, "Mem base addr align");
  print_info<info::device::half_fp_config, std::vector<info::fp_config>>(
      dev, "Half fp config");
  print_info<info::device::single_fp_config, std::vector<info::fp_config>>(
      dev, "Single fp config");
  print_info<info::device::double_fp_config, std::vector<info::fp_config>>(
      dev, "Double fp config");
  print_info<info::device::global_mem_cache_type, info::global_mem_cache_type>(
      dev, "Global mem cache type");
  print_info<info::device::global_mem_cache_line_size, std::uint32_t>(
      dev, "Global mem cache line size");
  print_info<info::device::global_mem_cache_size, std::uint64_t>(
      dev, "Global mem cache size");
  print_info<info::device::global_mem_size, std::uint64_t>(dev,
                                                           "Global mem size");
  print_info<info::device::max_constant_buffer_size, std::uint64_t>(
      dev, "Max constant buffer size");
  print_info<info::device::max_constant_args, std::uint32_t>(
      dev, "Max constant args");
  print_info<info::device::local_mem_type, info::local_mem_type>(
      dev, "Local mem type");
  print_info<info::device::local_mem_size, std::uint64_t>(dev,
                                                          "Local mem size");
  print_info<info::device::error_correction_support, bool>(
      dev, "Error correction support");
  print_info<info::device::host_unified_memory, bool>(dev,
                                                      "Host unified memory");
  print_info<info::device::profiling_timer_resolution, size_t>(
      dev, "Profiling timer resolution");
  print_info<info::device::is_endian_little, bool>(dev, "Is endian little");
  print_info<info::device::is_available, bool>(dev, "Is available");
  print_info<info::device::is_compiler_available, bool>(
      dev, "Is compiler available");
  print_info<info::device::is_linker_available, bool>(dev,
                                                      "Is linker available");
  print_info<info::device::execution_capabilities,
             std::vector<info::execution_capability>>(dev,
                                                      "Execution capabilities");
  print_info<info::device::queue_profiling, bool>(dev, "Queue profiling");
  print_info<info::device::built_in_kernels, std::vector<std::string>>(
      dev, "Built in kernels");
  print_info<info::device::platform, platform>(dev, "Platform");
  print_info<info::device::name, std::string>(dev, "Name");
  print_info<info::device::vendor, std::string>(dev, "Vendor");
  print_info<info::device::driver_version, std::string>(dev, "Driver version");
  print_info<info::device::profile, std::string>(dev, "Profile");
  print_info<info::device::version, std::string>(dev, "Version");
  print_info<info::device::backend_version, std::string>(dev,
                                                         "Backend version");
  print_info<info::device::opencl_c_version, std::string>(dev,
                                                          "OpenCL C version");
  print_info<info::device::extensions, std::vector<std::string>>(dev,
                                                                 "Extensions");
  print_info<info::device::printf_buffer_size, size_t>(dev,
                                                       "Printf buffer size");
  print_info<info::device::preferred_interop_user_sync, bool>(
      dev, "Preferred interop user sync");
  try {
    print_info<info::device::parent_device, device>(dev, "Parent device");
  } catch (sycl::exception e) {
    std::cout << "Expected exception has been caught: " << e.what()
              << std::endl;
  }
  print_info<info::device::partition_max_sub_devices, std::uint32_t>(
      dev, "Partition max sub devices");
  print_info<info::device::partition_properties,
             std::vector<info::partition_property>>(dev,
                                                    "Partition properties");
  print_info<info::device::partition_affinity_domains,
             std::vector<info::partition_affinity_domain>>(
      dev, "Partition affinity domains");
  // TODO test once subdevice creation is enabled
  // print_info<info::device::partition_type_property,
  // info::partition_property>(dev, "Partition type property");
  print_info<info::device::partition_type_affinity_domain,
             info::partition_affinity_domain>(dev,
                                              "Partition type affinity domain");
  print_info<info::device::reference_count, sycl::opencl::cl_uint>(
      dev, "Reference count");

  std::cout << separator << "Platform information\n" << separator;
  platform plt(dev.get_platform());
  print_info<info::platform::profile, std::string>(plt, "Profile");
  print_info<info::platform::version, std::string>(plt, "Version");
  print_info<info::platform::name, std::string>(plt, "Name");
  print_info<info::platform::vendor, std::string>(plt, "Vendor");
  print_info<info::platform::extensions, std::vector<std::string>>(
      plt, "Extensions");

  std::cout << separator << "Queue information\n" << separator;
  queue q(default_selector_v);
  auto qdev = q.get_info<sycl::info::queue::device>();
  std::cout << "Device from queue information\n";
  print_info<info::device::name, std::string>(qdev, "Name");
  auto ctx = q.get_info<sycl::info::queue::context>();

  std::cout << separator << "Context information\n" << separator;
  std::cout << "Devices from context information\n";
  auto cdevs = ctx.get_info<sycl::info::context::devices>();
  for (auto cdev : cdevs) {
    print_info<info::device::name, std::string>(cdev, "Name");
  }
  std::cout << separator << "Platform from context information\n" << separator;
  auto cplt = ctx.get_info<sycl::info::context::platform>();
  print_info<info::platform::name, std::string>(cplt, "Name");
}
