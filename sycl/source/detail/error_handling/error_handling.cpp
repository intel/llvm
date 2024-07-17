//===------------------- enqueue_kernel.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SYCL error handling of enqueue kernel operations
//
//===----------------------------------------------------------------------===//

#include "error_handling.hpp"

#include <detail/plugin.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/detail/pi.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail::enqueue_kernel_launch {

void handleOutOfResources(const device_impl &DeviceImpl, pi_kernel Kernel,
                          const NDRDescT &NDRDesc) {
  sycl::platform Platform = DeviceImpl.get_platform();
  sycl::backend Backend = Platform.get_backend();
  if (Backend == sycl::backend::ext_oneapi_cuda) {
    // PI_ERROR_OUT_OF_RESOURCES is returned when the kernel registers
    // required for the launch config exceeds the maximum number of registers
    // per block (PI_EXT_CODEPLAY_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP).
    // This is if local_work_size[0] * ... * local_work_size[work_dim - 1]
    // multiplied by PI_KERNEL_GROUP_INFO_NUM_REGS is greater than the value
    // of PI_KERNEL_MAX_NUM_REGISTERS_PER_BLOCK. See Table 15: Technical
    // Specifications per Compute Capability, for limitations.
    const size_t TotalNumberOfWIs =
        NDRDesc.LocalSize[0] * NDRDesc.LocalSize[1] * NDRDesc.LocalSize[2];

    const uint32_t MaxRegistersPerBlock =
        DeviceImpl.get_info<ext::codeplay::experimental::info::device::
                                max_registers_per_work_group>();

    const PluginPtr &Plugin = DeviceImpl.getPlugin();
    sycl::detail::pi::PiDevice Device = DeviceImpl.getHandleRef();

    uint32_t NumRegisters = 0;
    Plugin->call<PiApiKind::piKernelGetGroupInfo>(
        Kernel, Device, PI_KERNEL_GROUP_INFO_NUM_REGS, sizeof(NumRegisters),
        &NumRegisters, nullptr);

    const bool HasExceededAvailableRegisters =
        TotalNumberOfWIs * NumRegisters > MaxRegistersPerBlock;

    if (HasExceededAvailableRegisters) {
      std::string message(
          "Exceeded the number of registers available on the hardware.\n");
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::nd_range),
          // Additional information which can be helpful to the user.
          message.append(
              "\tThe number registers per work-group cannot exceed " +
              std::to_string(MaxRegistersPerBlock) +
              " for this kernel on this device.\n"
              "\tThe kernel uses " +
              std::to_string(NumRegisters) +
              " registers per work-item for a total of " +
              std::to_string(TotalNumberOfWIs) +
              " work-items per work-group.\n"));
    }
  }
  // Fallback
  constexpr pi_result Error = PI_ERROR_OUT_OF_RESOURCES;
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "PI backend failed. PI backend returns:" +
                            codeToString(Error));
}

void handleInvalidWorkGroupSize(const device_impl &DeviceImpl, pi_kernel Kernel,
                                const NDRDescT &NDRDesc) {
  sycl::platform Platform = DeviceImpl.get_platform();

  // Some of the error handling below is special for particular OpenCL
  // versions.  If this is an OpenCL backend, get the version.
  bool IsOpenCL = false;      // Backend is any OpenCL version
  bool IsOpenCLV1x = false;   // Backend is OpenCL 1.x
  bool IsOpenCLVGE20 = false; // Backend is Greater or Equal to OpenCL 2.0
  bool IsLevelZero = false;   // Backend is any OneAPI Level 0 version
  auto Backend = Platform.get_backend();
  if (Backend == sycl::backend::opencl) {
    std::string VersionString =
        DeviceImpl.get_info<info::device::version>().substr(7, 3);
    IsOpenCL = true;
    IsOpenCLV1x = (VersionString.find("1.") == 0);
    IsOpenCLVGE20 =
        (VersionString.find("2.") == 0) || (VersionString.find("3.") == 0);
  } else if (Backend == sycl::backend::ext_oneapi_level_zero) {
    IsLevelZero = true;
  }

  const PluginPtr &Plugin = DeviceImpl.getPlugin();
  sycl::detail::pi::PiDevice Device = DeviceImpl.getHandleRef();

  size_t CompileWGSize[3] = {0};
  Plugin->call<PiApiKind::piKernelGetGroupInfo>(
      Kernel, Device, PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
      sizeof(size_t) * 3, CompileWGSize, nullptr);

  size_t MaxWGSize = 0;
  Plugin->call<PiApiKind::piDeviceGetInfo>(Device,
                                           PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
                                           sizeof(size_t), &MaxWGSize, nullptr);

  const bool HasLocalSize = (NDRDesc.LocalSize[0] != 0);

  if (CompileWGSize[0] != 0) {
    if (CompileWGSize[0] > MaxWGSize || CompileWGSize[1] > MaxWGSize ||
        CompileWGSize[2] > MaxWGSize)
      throw sycl::exception(
          make_error_code(errc::kernel_not_supported),
          "Submitting a kernel decorated with reqd_work_group_size attribute "
          "to a device that does not support this work group size is invalid.");

    // OpenCL 1.x && 2.0:
    // PI_ERROR_INVALID_WORK_GROUP_SIZE if local_work_size is NULL and the
    // reqd_work_group_size attribute is used to declare the work-group size
    // for kernel in the program source.
    if (!HasLocalSize && (IsOpenCLV1x || IsOpenCLVGE20)) {
      throw sycl::exception(
          make_error_code(errc::nd_range),
          "OpenCL 1.x and 2.0 requires to pass local size argument even if "
          "required work-group size was specified in the program source");
    }
    // PI_ERROR_INVALID_WORK_GROUP_SIZE if local_work_size is specified and does
    // not match the required work-group size for kernel in the program source.
    if (NDRDesc.LocalSize[0] != CompileWGSize[0] ||
        NDRDesc.LocalSize[1] != CompileWGSize[1] ||
        NDRDesc.LocalSize[2] != CompileWGSize[2])
      throw sycl::exception(
          make_error_code(errc::nd_range),
          "The specified local size {" + std::to_string(NDRDesc.LocalSize[2]) +
              ", " + std::to_string(NDRDesc.LocalSize[1]) + ", " +
              std::to_string(NDRDesc.LocalSize[0]) +
              "} doesn't match the required work-group size specified "
              "in the program source {" +
              std::to_string(CompileWGSize[2]) + ", " +
              std::to_string(CompileWGSize[1]) + ", " +
              std::to_string(CompileWGSize[0]) + "}");
  }

  if (HasLocalSize) {
    size_t MaxThreadsPerBlock[3] = {};
    Plugin->call<PiApiKind::piDeviceGetInfo>(
        Device, PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES, sizeof(MaxThreadsPerBlock),
        MaxThreadsPerBlock, nullptr);

    for (size_t I = 0; I < 3; ++I) {
      if (MaxThreadsPerBlock[I] < NDRDesc.LocalSize[I]) {
        throw sycl::exception(make_error_code(errc::nd_range),
                              "The number of work-items in each dimension of a "
                              "work-group cannot exceed {" +
                                  std::to_string(MaxThreadsPerBlock[0]) + ", " +
                                  std::to_string(MaxThreadsPerBlock[1]) + ", " +
                                  std::to_string(MaxThreadsPerBlock[2]) +
                                  "} for this device");
      }
    }
  }

  if (IsOpenCLV1x) {
    // OpenCL 1.x:
    // PI_ERROR_INVALID_WORK_GROUP_SIZE if local_work_size is specified and
    // the total number of work-items in the work-group computed as
    // local_work_size[0] * ... * local_work_size[work_dim - 1] is greater
    // than the value specified by PI_DEVICE_MAX_WORK_GROUP_SIZE in
    // table 4.3
    const size_t TotalNumberOfWIs =
        NDRDesc.LocalSize[0] * NDRDesc.LocalSize[1] * NDRDesc.LocalSize[2];
    if (TotalNumberOfWIs > MaxWGSize)
      throw sycl::exception(
          make_error_code(errc::nd_range),
          "Total number of work-items in a work-group cannot exceed " +
              std::to_string(MaxWGSize));
  } else if (IsOpenCLVGE20 || IsLevelZero) {
    // OpenCL 2.x or OneAPI Level Zero:
    // PI_ERROR_INVALID_WORK_GROUP_SIZE if local_work_size is specified and
    // the total number of work-items in the work-group computed as
    // local_work_size[0] * ... * local_work_size[work_dim - 1] is greater
    // than the value specified by PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE in
    // table 5.21.
    size_t KernelWGSize = 0;
    Plugin->call<PiApiKind::piKernelGetGroupInfo>(
        Kernel, Device, PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE, sizeof(size_t),
        &KernelWGSize, nullptr);
    const size_t TotalNumberOfWIs =
        NDRDesc.LocalSize[0] * NDRDesc.LocalSize[1] * NDRDesc.LocalSize[2];
    if (TotalNumberOfWIs > KernelWGSize)
      throw sycl::exception(
          make_error_code(errc::nd_range),
          "Total number of work-items in a work-group cannot exceed " +
              std::to_string(KernelWGSize) + " for this kernel");
  } else {
    // TODO: Should probably have something similar for the other backends
  }

  if (HasLocalSize) {
    // Is the global range size evenly divisible by the local workgroup size?
    const bool NonUniformWGs =
        (NDRDesc.LocalSize[0] != 0 &&
         NDRDesc.GlobalSize[0] % NDRDesc.LocalSize[0] != 0) ||
        (NDRDesc.LocalSize[1] != 0 &&
         NDRDesc.GlobalSize[1] % NDRDesc.LocalSize[1] != 0) ||
        (NDRDesc.LocalSize[2] != 0 &&
         NDRDesc.GlobalSize[2] % NDRDesc.LocalSize[2] != 0);
    // Is the local size of the workgroup greater than the global range size in
    // any dimension?
    if (IsOpenCL) {
      const bool LocalExceedsGlobal =
          NonUniformWGs && (NDRDesc.LocalSize[0] > NDRDesc.GlobalSize[0] ||
                            NDRDesc.LocalSize[1] > NDRDesc.GlobalSize[1] ||
                            NDRDesc.LocalSize[2] > NDRDesc.GlobalSize[2]);

      if (NonUniformWGs) {
        if (IsOpenCLV1x) {
          // OpenCL 1.x:
          // PI_ERROR_INVALID_WORK_GROUP_SIZE if local_work_size is specified
          // and number of workitems specified by global_work_size is not evenly
          // divisible by size of work-group given by local_work_size
          if (LocalExceedsGlobal)
            throw sycl::exception(make_error_code(errc::nd_range),
                                  "Local workgroup size cannot be greater than "
                                  "global range in any dimension");
          else
            throw sycl::exception(make_error_code(errc::nd_range),
                                  "Global_work_size must be evenly divisible "
                                  "by local_work_size. Non-uniform work-groups "
                                  "are not supported by the target device");
        } else {
          // OpenCL 2.x:
          // PI_ERROR_INVALID_WORK_GROUP_SIZE if the program was compiled with
          // –cl-uniform-work-group-size and the number of work-items specified
          // by global_work_size is not evenly divisible by size of work-group
          // given by local_work_size

          pi_program Program = nullptr;
          Plugin->call<PiApiKind::piKernelGetInfo>(
              Kernel, PI_KERNEL_INFO_PROGRAM, sizeof(pi_program), &Program,
              nullptr);
          size_t OptsSize = 0;
          Plugin->call<PiApiKind::piProgramGetBuildInfo>(
              Program, Device, PI_PROGRAM_BUILD_INFO_OPTIONS, 0, nullptr,
              &OptsSize);
          std::string Opts(OptsSize, '\0');
          Plugin->call<PiApiKind::piProgramGetBuildInfo>(
              Program, Device, PI_PROGRAM_BUILD_INFO_OPTIONS, OptsSize,
              &Opts.front(), nullptr);
          const bool HasStd20 = Opts.find("-cl-std=CL2.0") != std::string::npos;
          const bool RequiresUniformWGSize =
              Opts.find("-cl-uniform-work-group-size") != std::string::npos;
          std::string LocalWGSize = std::to_string(NDRDesc.LocalSize[0]) +
                                    ", " +
                                    std::to_string(NDRDesc.LocalSize[1]) +
                                    ", " + std::to_string(NDRDesc.LocalSize[2]);
          std::string GlobalWGSize =
              std::to_string(NDRDesc.GlobalSize[0]) + ", " +
              std::to_string(NDRDesc.GlobalSize[1]) + ", " +
              std::to_string(NDRDesc.GlobalSize[2]);
          std::string message =
              LocalExceedsGlobal
                  ? "Local work-group size {" + LocalWGSize +
                        "} is greater than global range size {" + GlobalWGSize +
                        "}. "
                  : "Global work size {" + GlobalWGSize +
                        "} is not evenly divisible by local work-group size {" +
                        LocalWGSize + "}. ";
          if (!HasStd20)
            throw sycl::exception(
                make_error_code(errc::nd_range),
                message.append("Non-uniform work-groups are not allowed by "
                               "default. Underlying OpenCL 2.x implementation "
                               "supports this feature and to enable it, build "
                               "device program with -cl-std=CL2.0"));
          else if (RequiresUniformWGSize)
            throw sycl::exception(
                make_error_code(errc::nd_range),
                message.append("Non-uniform work-groups are not allowed when "
                               "-cl-uniform-work-group-size flag is used. "
                               "Underlying OpenCL 2.x implementation supports "
                               "this feature, but it is being disabled by "
                               "-cl-uniform-work-group-size build flag"));
          // else unknown.  fallback (below)
        }
      }
    } else {
      // TODO: Decide what checks (if any) we need for the other backends
    }
    throw sycl::exception(
        make_error_code(errc::nd_range),
        "Non-uniform work-groups are not supported by the target device");
  }
  // TODO: required number of sub-groups, OpenCL 2.1:
  // PI_ERROR_INVALID_WORK_GROUP_SIZE if local_work_size is specified and is not
  // consistent with the required number of sub-groups for kernel in the
  // program source.

  throw exception(make_error_code(errc::nd_range),
                  "internal error: expected HasLocalSize");
}

void handleInvalidWorkItemSize(const device_impl &DeviceImpl,
                               const NDRDescT &NDRDesc) {

  const PluginPtr &Plugin = DeviceImpl.getPlugin();
  sycl::detail::pi::PiDevice Device = DeviceImpl.getHandleRef();

  size_t MaxWISize[] = {0, 0, 0};

  Plugin->call<PiApiKind::piDeviceGetInfo>(
      Device, PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES, sizeof(MaxWISize), &MaxWISize,
      nullptr);
  for (unsigned I = 0; I < NDRDesc.Dims; I++) {
    if (NDRDesc.LocalSize[I] > MaxWISize[I])
      throw sycl::exception(
          make_error_code(errc::nd_range),
          "Number of work-items in a work-group exceed limit for dimension " +
              std::to_string(I) + " : " + std::to_string(NDRDesc.LocalSize[I]) +
              " > " + std::to_string(MaxWISize[I]));
  }
}

void handleInvalidValue(const device_impl &DeviceImpl,
                        const NDRDescT &NDRDesc) {
  const PluginPtr &Plugin = DeviceImpl.getPlugin();
  sycl::detail::pi::PiDevice Device = DeviceImpl.getHandleRef();

  size_t MaxNWGs[] = {0, 0, 0};
  Plugin->call<PiApiKind::piDeviceGetInfo>(
      Device, PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_3D, sizeof(MaxNWGs),
      &MaxNWGs, nullptr);
  for (unsigned int I = 0; I < NDRDesc.Dims; I++) {
    size_t NWgs = NDRDesc.GlobalSize[I] / NDRDesc.LocalSize[I];
    if (NWgs > MaxNWGs[I])
      throw sycl::exception(
          make_error_code(errc::nd_range),
          "Number of work-groups exceed limit for dimension " +
              std::to_string(I) + " : " + std::to_string(NWgs) + " > " +
              std::to_string(MaxNWGs[I]));
  }

  // fallback
  throw exception(make_error_code(errc::nd_range), "unknown internal error");
}

void handleErrorOrWarning(pi_result Error, const device_impl &DeviceImpl,
                          pi_kernel Kernel, const NDRDescT &NDRDesc) {
  assert(Error != PI_SUCCESS &&
         "Success is expected to be handled on caller side");
  switch (Error) {
  case PI_ERROR_OUT_OF_RESOURCES:
    return handleOutOfResources(DeviceImpl, Kernel, NDRDesc);

  case PI_ERROR_INVALID_WORK_GROUP_SIZE:
    return handleInvalidWorkGroupSize(DeviceImpl, Kernel, NDRDesc);

  case PI_ERROR_INVALID_KERNEL_ARGS:
    throw detail::set_pi_error(
        sycl::exception(
            make_error_code(errc::kernel_argument),
            "The kernel argument values have not been specified OR a kernel "
            "argument declared to be a pointer to a type."),
        PI_ERROR_INVALID_KERNEL_ARGS);

  case PI_ERROR_INVALID_WORK_ITEM_SIZE:
    return handleInvalidWorkItemSize(DeviceImpl, NDRDesc);

  case PI_ERROR_IMAGE_FORMAT_NOT_SUPPORTED:
    throw detail::set_pi_error(
        sycl::exception(
            make_error_code(errc::feature_not_supported),
            "image object is specified as an argument value and the image "
            "format is not supported by device associated with queue"),
        PI_ERROR_IMAGE_FORMAT_NOT_SUPPORTED);

  case PI_ERROR_MISALIGNED_SUB_BUFFER_OFFSET:
    throw detail::set_pi_error(
        sycl::exception(make_error_code(errc::invalid),
                        "a sub-buffer object is specified as the value for an "
                        "argument that is a buffer object and the offset "
                        "specified when the sub-buffer object is created is "
                        "not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value "
                        "for device associated with queue"),
        PI_ERROR_MISALIGNED_SUB_BUFFER_OFFSET);

  case PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE:
    throw detail::set_pi_error(
        sycl::exception(
            make_error_code(errc::memory_allocation),
            "failure to allocate memory for data store associated with image "
            "or buffer objects specified as arguments to kernel"),
        PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE);

  case PI_ERROR_INVALID_IMAGE_SIZE:
    throw detail::set_pi_error(
        sycl::exception(
            make_error_code(errc::invalid),
            "image object is specified as an argument value and the image "
            "dimensions (image width, height, specified or compute row and/or "
            "slice pitch) are not supported by device associated with queue"),
        PI_ERROR_INVALID_IMAGE_SIZE);

  case PI_ERROR_INVALID_VALUE:
    return handleInvalidValue(DeviceImpl, NDRDesc);

  case PI_ERROR_PLUGIN_SPECIFIC_ERROR:
    // checkPiResult does all the necessary handling for
    // PI_ERROR_PLUGIN_SPECIFIC_ERROR, making sure an error is thrown or not,
    // depending on whether PI_ERROR_PLUGIN_SPECIFIC_ERROR contains an error or
    // a warning. It also ensures that the contents of the error message buffer
    // (used only by PI_ERROR_PLUGIN_SPECIFIC_ERROR) get handled correctly.
    return DeviceImpl.getPlugin()->checkPiResult(Error);

    // TODO: Handle other error codes

  default:
    throw detail::set_pi_error(
        exception(make_error_code(errc::runtime), "PI error"), Error);
  }
}

} // namespace detail::enqueue_kernel_launch

namespace detail::kernel_get_group_info {
void handleErrorOrWarning(pi_result Error, pi_kernel_group_info Descriptor,
                          const PluginPtr &Plugin) {
  assert(Error != PI_SUCCESS &&
         "Success is expected to be handled on caller side");
  switch (Error) {
  case PI_ERROR_INVALID_VALUE:
    if (Descriptor == CL_KERNEL_GLOBAL_WORK_SIZE)
      throw sycl::exception(
          sycl::make_error_code(errc::invalid),
          "info::kernel_device_specific::global_work_size descriptor may only "
          "be used if the device type is device_type::custom or if the kernel "
          "is a built-in kernel.");
    break;
  // TODO: Handle other error codes
  default:
    Plugin->checkPiResult(Error);
    break;
  }
}
} // namespace detail::kernel_get_group_info

} // namespace _V1
} // namespace sycl
