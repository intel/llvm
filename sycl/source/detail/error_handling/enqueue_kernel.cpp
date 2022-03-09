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

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

namespace enqueue_kernel_launch {

bool handleInvalidWorkGroupSize(const device_impl &DeviceImpl, pi_kernel Kernel,
                                const NDRDescT &NDRDesc) {
  const bool HasLocalSize = (NDRDesc.LocalSize[0] != 0);

  const plugin &Plugin = DeviceImpl.getPlugin();
  RT::PiDevice Device = DeviceImpl.getHandleRef();
  cl::sycl::platform Platform = DeviceImpl.get_platform();

  if (HasLocalSize) {
    size_t MaxThreadsPerBlock[3] = {};
    Plugin.call<PiApiKind::piDeviceGetInfo>(
        Device, PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES, sizeof(MaxThreadsPerBlock),
        MaxThreadsPerBlock, nullptr);

    for (size_t I = 0; I < 3; ++I) {
      if (MaxThreadsPerBlock[I] < NDRDesc.LocalSize[I]) {
        throw sycl::nd_range_error(
            "The number of work-items in each dimension of a work-group cannot "
            "exceed {" +
                std::to_string(MaxThreadsPerBlock[0]) + ", " +
                std::to_string(MaxThreadsPerBlock[1]) + ", " +
                std::to_string(MaxThreadsPerBlock[2]) + "} for this device",
            PI_INVALID_WORK_GROUP_SIZE);
      }
    }
  }

  // Some of the error handling below is special for particular OpenCL
  // versions.  If this is an OpenCL backend, get the version.
  bool IsOpenCL = false;    // Backend is any OpenCL version
  bool IsOpenCLV1x = false; // Backend is OpenCL 1.x
  bool IsOpenCLV20 = false; // Backend is OpenCL 2.0
  if (Platform.get_backend() == cl::sycl::backend::opencl) {
    std::string VersionString = DeviceImpl.get_info<info::device::version>();
    IsOpenCL = true;
    IsOpenCLV1x = (VersionString.find("1.") == 0);
    IsOpenCLV20 = (VersionString.find("2.0") == 0);
  }

  size_t CompileWGSize[3] = {0};
  Plugin.call<PiApiKind::piKernelGetGroupInfo>(
      Kernel, Device, PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
      sizeof(size_t) * 3, CompileWGSize, nullptr);

  if (CompileWGSize[0] != 0) {
    // OpenCL 1.x && 2.0:
    // PI_INVALID_WORK_GROUP_SIZE if local_work_size is NULL and the
    // reqd_work_group_size attribute is used to declare the work-group size
    // for kernel in the program source.
    if (!HasLocalSize && (IsOpenCLV1x || IsOpenCLV20)) {
      throw sycl::nd_range_error(
          "OpenCL 1.x and 2.0 requires to pass local size argument even if "
          "required work-group size was specified in the program source",
          PI_INVALID_WORK_GROUP_SIZE);
    }
    // PI_INVALID_WORK_GROUP_SIZE if local_work_size is specified and does not
    // match the required work-group size for kernel in the program source.
    if (NDRDesc.LocalSize[0] != CompileWGSize[0] ||
        NDRDesc.LocalSize[1] != CompileWGSize[1] ||
        NDRDesc.LocalSize[2] != CompileWGSize[2])
      throw sycl::nd_range_error(
          "The specified local size {" + std::to_string(NDRDesc.LocalSize[0]) +
              ", " + std::to_string(NDRDesc.LocalSize[1]) + ", " +
              std::to_string(NDRDesc.LocalSize[2]) +
              "} doesn't match the required work-group size specified "
              "in the program source {" +
              std::to_string(CompileWGSize[0]) + ", " +
              std::to_string(CompileWGSize[1]) + ", " +
              std::to_string(CompileWGSize[2]) + "}",
          PI_INVALID_WORK_GROUP_SIZE);
  }
  if (IsOpenCL) {
    if (IsOpenCLV1x) {
      // OpenCL 1.x:
      // PI_INVALID_WORK_GROUP_SIZE if local_work_size is specified and the
      // total number of work-items in the work-group computed as
      // local_work_size[0] * ... * local_work_size[work_dim - 1] is greater
      // than the value specified by PI_DEVICE_MAX_WORK_GROUP_SIZE in
      // table 4.3
      size_t MaxWGSize = 0;
      Plugin.call<PiApiKind::piDeviceGetInfo>(
          Device, PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE, sizeof(size_t),
          &MaxWGSize, nullptr);
      const size_t TotalNumberOfWIs =
          NDRDesc.LocalSize[0] * NDRDesc.LocalSize[1] * NDRDesc.LocalSize[2];
      if (TotalNumberOfWIs > MaxWGSize)
        throw sycl::nd_range_error(
            "Total number of work-items in a work-group cannot exceed " +
                std::to_string(MaxWGSize),
            PI_INVALID_WORK_GROUP_SIZE);
    } else {
      // OpenCL 2.x:
      // PI_INVALID_WORK_GROUP_SIZE if local_work_size is specified and the
      // total number of work-items in the work-group computed as
      // local_work_size[0] * ... * local_work_size[work_dim - 1] is greater
      // than the value specified by PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE in
      // table 5.21.
      size_t KernelWGSize = 0;
      Plugin.call<PiApiKind::piKernelGetGroupInfo>(
          Kernel, Device, PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE, sizeof(size_t),
          &KernelWGSize, nullptr);
      const size_t TotalNumberOfWIs =
          NDRDesc.LocalSize[0] * NDRDesc.LocalSize[1] * NDRDesc.LocalSize[2];
      if (TotalNumberOfWIs > KernelWGSize)
        throw sycl::nd_range_error(
            "Total number of work-items in a work-group cannot exceed " +
                std::to_string(KernelWGSize) + " for this kernel",
            PI_INVALID_WORK_GROUP_SIZE);
    }
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
          // PI_INVALID_WORK_GROUP_SIZE if local_work_size is specified and
          // number of workitems specified by global_work_size is not evenly
          // divisible by size of work-group given by local_work_size
          if (LocalExceedsGlobal)
            throw sycl::nd_range_error("Local workgroup size cannot be greater "
                                       "than global range in any dimension",
                                       PI_INVALID_WORK_GROUP_SIZE);
          else
            throw sycl::nd_range_error(
                "Global_work_size must be evenly divisible by local_work_size. "
                "Non-uniform work-groups are not supported by the target "
                "device",
                PI_INVALID_WORK_GROUP_SIZE);
        } else {
          // OpenCL 2.x:
          // PI_INVALID_WORK_GROUP_SIZE if the program was compiled with
          // –cl-uniform-work-group-size and the number of work-items specified
          // by global_work_size is not evenly divisible by size of work-group
          // given by local_work_size

          pi_program Program = nullptr;
          Plugin.call<PiApiKind::piKernelGetInfo>(
              Kernel, PI_KERNEL_INFO_PROGRAM, sizeof(pi_program), &Program,
              nullptr);
          size_t OptsSize = 0;
          Plugin.call<PiApiKind::piProgramGetBuildInfo>(
              Program, Device, PI_PROGRAM_BUILD_INFO_OPTIONS, 0, nullptr,
              &OptsSize);
          std::string Opts(OptsSize, '\0');
          Plugin.call<PiApiKind::piProgramGetBuildInfo>(
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
            throw sycl::nd_range_error(
                message.append(
                    "Non-uniform work-groups are not allowed by "
                    "default. Underlying "
                    "OpenCL 2.x implementation supports this feature "
                    "and to enable "
                    "it, build device program with -cl-std=CL2.0"),
                PI_INVALID_WORK_GROUP_SIZE);
          else if (RequiresUniformWGSize)
            throw sycl::nd_range_error(
                message.append(
                    "Non-uniform work-groups are not allowed by when "
                    "-cl-uniform-work-group-size flag is used. Underlying "
                    "OpenCL 2.x implementation supports this feature, but it "
                    "is "
                    "being "
                    "disabled by -cl-uniform-work-group-size build flag"),
                PI_INVALID_WORK_GROUP_SIZE);
          // else unknown.  fallback (below)
        }
      }
    } else {
      // TODO: Decide what checks (if any) we need for the other backends
    }
    throw sycl::nd_range_error(
        "Non-uniform work-groups are not supported by the target device",
        PI_INVALID_WORK_GROUP_SIZE);
  }
  // TODO: required number of sub-groups, OpenCL 2.1:
  // PI_INVALID_WORK_GROUP_SIZE if local_work_size is specified and is not
  // consistent with the required number of sub-groups for kernel in the
  // program source.

  // Fallback
  constexpr pi_result Error = PI_INVALID_WORK_GROUP_SIZE;
  throw runtime_error(
      "PI backend failed. PI backend returns: " + codeToString(Error), Error);
}

bool handleInvalidWorkItemSize(const device_impl &DeviceImpl,
                               const NDRDescT &NDRDesc) {

  const plugin &Plugin = DeviceImpl.getPlugin();
  RT::PiDevice Device = DeviceImpl.getHandleRef();

  size_t MaxWISize[] = {0, 0, 0};

  Plugin.call<PiApiKind::piDeviceGetInfo>(
      Device, PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES, sizeof(MaxWISize), &MaxWISize,
      nullptr);
  for (unsigned I = 0; I < NDRDesc.Dims; I++) {
    if (NDRDesc.LocalSize[I] > MaxWISize[I])
      throw sycl::nd_range_error(
          "Number of work-items in a work-group exceed limit for dimension " +
              std::to_string(I) + " : " + std::to_string(NDRDesc.LocalSize[I]) +
              " > " + std::to_string(MaxWISize[I]),
          PI_INVALID_WORK_ITEM_SIZE);
  }
  return 0;
}

bool handleInvalidValue(const device_impl &DeviceImpl,
                        const NDRDescT &NDRDesc) {
  const plugin &Plugin = DeviceImpl.getPlugin();
  RT::PiDevice Device = DeviceImpl.getHandleRef();

  size_t MaxNWGs[] = {0, 0, 0};
  Plugin.call<PiApiKind::piDeviceGetInfo>(Device,
                                          PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES,
                                          sizeof(MaxNWGs), &MaxNWGs, nullptr);
  for (unsigned int I = 0; I < NDRDesc.Dims; I++) {
    size_t n_wgs = NDRDesc.GlobalSize[I] / NDRDesc.LocalSize[I];
    if (n_wgs > MaxNWGs[I])
      throw sycl::nd_range_error(
          "Number of work-groups exceed limit for dimension " +
              std::to_string(I) + " : " + std::to_string(n_wgs) + " > " +
              std::to_string(MaxNWGs[I]),
          PI_INVALID_VALUE);
  }

  // fallback
  constexpr pi_result Error = PI_INVALID_VALUE;
  throw runtime_error(
      "Native API failed. Native API returns: " + codeToString(Error), Error);
}

bool handleError(pi_result Error, const device_impl &DeviceImpl,
                 pi_kernel Kernel, const NDRDescT &NDRDesc) {
  assert(Error != PI_SUCCESS &&
         "Success is expected to be handled on caller side");
  switch (Error) {
  case PI_INVALID_WORK_GROUP_SIZE:
    return handleInvalidWorkGroupSize(DeviceImpl, Kernel, NDRDesc);

  case PI_INVALID_KERNEL_ARGS:
    throw sycl::nd_range_error(
        "The kernel argument values have not been specified "
        " OR "
        "a kernel argument declared to be a pointer to a type.",
        PI_INVALID_KERNEL_ARGS);

  case PI_INVALID_WORK_ITEM_SIZE:
    return handleInvalidWorkItemSize(DeviceImpl, NDRDesc);

  case PI_IMAGE_FORMAT_NOT_SUPPORTED:
    throw sycl::nd_range_error(
        "image object is specified as an argument value"
        " and the image format is not supported by device associated"
        " with queue",
        PI_IMAGE_FORMAT_NOT_SUPPORTED);

  case PI_MISALIGNED_SUB_BUFFER_OFFSET:
    throw sycl::nd_range_error(
        "a sub-buffer object is specified as the value for an argument "
        " that is a buffer object and the offset specified "
        "when the sub-buffer object is created is not aligned "
        "to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated"
        " with queue",
        PI_MISALIGNED_SUB_BUFFER_OFFSET);

  case PI_MEM_OBJECT_ALLOCATION_FAILURE:
    throw sycl::nd_range_error(
        "failure to allocate memory for data store associated with image"
        " or buffer objects specified as arguments to kernel",
        PI_MEM_OBJECT_ALLOCATION_FAILURE);

  case PI_INVALID_IMAGE_SIZE:
    throw sycl::nd_range_error(
        "image object is specified as an argument value and the image "
        "dimensions (image width, height, specified or compute row and/or "
        "slice pitch) are not supported by device associated with queue",
        PI_INVALID_IMAGE_SIZE);

  case PI_INVALID_VALUE:
    return handleInvalidValue(DeviceImpl, NDRDesc);

    // TODO: Handle other error codes

  default:
    throw runtime_error(
        "Native API failed. Native API returns: " + codeToString(Error), Error);
  }
}

} // namespace enqueue_kernel_launch

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
