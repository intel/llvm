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

#include <CL/sycl/detail/pi.hpp>

__SYCL_INLINE namespace cl {
namespace sycl {
namespace detail {

namespace enqueue_kernel_launch {

bool handleInvalidWorkGroupSize(pi_device Device, pi_kernel Kernel,
                                const NDRDescT &NDRDesc) {
  const bool HasLocalSize = (NDRDesc.LocalSize[0] != 0);

  size_t VerSize = 0;
  PI_CALL(piDeviceGetInfo)(Device, PI_DEVICE_INFO_VERSION, 0, nullptr,
                           &VerSize);
  assert(VerSize >= 10 &&
         "Unexpected device version string"); // strlen("OpenCL X.Y")
  string_class VerStr(VerSize, '\0');
  PI_CALL(piDeviceGetInfo)(Device, PI_DEVICE_INFO_VERSION, VerSize,
                           &VerStr.front(), nullptr);
  const char *Ver = &VerStr[7]; // strlen("OpenCL ")

  size_t CompileWGSize[3] = {0};
  PI_CALL(piKernelGetGroupInfo)(Kernel, Device,
                                CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                sizeof(size_t) * 3, CompileWGSize, nullptr);

  if (CompileWGSize[0] != 0) {
    // OpenCL 1.x && 2.0:
    // CL_INVALID_WORK_GROUP_SIZE if local_work_size is NULL and the
    // reqd_work_group_size attribute is used to declare the work-group size
    // for kernel in the program source.
    if (!HasLocalSize && (Ver[0] == '1' || (Ver[0] == '2' && Ver[2] == '0')))
      throw sycl::nd_range_error(
          "OpenCL 1.x and 2.0 requires to pass local size argument even if "
          "required work-group size was specified in the program source",
          PI_INVALID_WORK_GROUP_SIZE);

    // Any OpenCL version:
    // CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified and does not
    // match the required work-group size for kernel in the program source.
    if (NDRDesc.LocalSize[0] != CompileWGSize[0] ||
        NDRDesc.LocalSize[1] != CompileWGSize[1] ||
        NDRDesc.LocalSize[2] != CompileWGSize[2])
      throw sycl::nd_range_error(
          "Specified local size doesn't match the required work-group size "
          "specified in the program source",
          PI_INVALID_WORK_GROUP_SIZE);
  }

  if (Ver[0] == '1') {
    // OpenCL 1.x:
    // CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified and the
    // total number of work-items in the work-group computed as
    // local_work_size[0] * ... * local_work_size[work_dim – 1] is greater
    // than the value specified by CL_DEVICE_MAX_WORK_GROUP_SIZE in
    // table 4.3
    size_t MaxWGSize = 0;
    PI_CALL(piDeviceGetInfo)(Device, PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
                             sizeof(size_t), &MaxWGSize, nullptr);
    const size_t TotalNumberOfWIs =
        NDRDesc.LocalSize[0] * NDRDesc.LocalSize[1] * NDRDesc.LocalSize[2];
    if (TotalNumberOfWIs > MaxWGSize)
      throw sycl::nd_range_error(
          "Total number of work-items in a work-group cannot exceed "
          "info::device::max_work_group_size which is equal to " +
              std::to_string(MaxWGSize),
          PI_INVALID_WORK_GROUP_SIZE);
  } else {
    // OpenCL 2.x:
    // CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified and the
    // total number of work-items in the work-group computed as
    // local_work_size[0] * ... * local_work_size[work_dim – 1] is greater
    // than the value specified by CL_KERNEL_WORK_GROUP_SIZE in table 5.21.
    size_t KernelWGSize = 0;
    PI_CALL(piKernelGetGroupInfo)(Kernel, Device, CL_KERNEL_WORK_GROUP_SIZE,
                                  sizeof(size_t), &KernelWGSize, nullptr);
    const size_t TotalNumberOfWIs =
        NDRDesc.LocalSize[0] * NDRDesc.LocalSize[1] * NDRDesc.LocalSize[2];
    if (TotalNumberOfWIs > KernelWGSize)
      throw sycl::nd_range_error(
          "Total number of work-items in a work-group cannot exceed "
          "info::kernel_work_group::work_group_size which is equal to " +
              std::to_string(KernelWGSize) + " for this kernel",
          PI_INVALID_WORK_GROUP_SIZE);
  }

  if (HasLocalSize) {
    const bool NonUniformWGs =
        (NDRDesc.LocalSize[0] != 0 &&
         NDRDesc.GlobalSize[0] % NDRDesc.LocalSize[0] != 0) ||
        (NDRDesc.LocalSize[1] != 0 &&
         NDRDesc.GlobalSize[1] % NDRDesc.LocalSize[1] != 0) ||
        (NDRDesc.LocalSize[2] != 0 &&
         NDRDesc.GlobalSize[2] % NDRDesc.LocalSize[2] != 0);

    if (Ver[0] == '1') {
      // OpenCL 1.x:
      // CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified and
      // number of workitems specified by global_work_size is not evenly
      // divisible by size of work-group given by local_work_size

      if (NonUniformWGs)
        throw sycl::nd_range_error(
            "Non-uniform work-groups are not supported by the target device",
            PI_INVALID_WORK_GROUP_SIZE);
    } else {
      // OpenCL 2.x:
      // CL_INVALID_WORK_GROUP_SIZE if the program was compiled with
      // –cl-uniform-work-group-size and the number of work-items specified
      // by global_work_size is not evenly divisible by size of work-group
      // given by local_work_size

      pi_program Program = nullptr;
      PI_CALL(piKernelGetInfo)(Kernel, CL_KERNEL_PROGRAM, sizeof(pi_program),
                               &Program, nullptr);
      size_t OptsSize = 0;
      PI_CALL(piProgramGetBuildInfo)(Program, Device, CL_PROGRAM_BUILD_OPTIONS,
                                     0, nullptr, &OptsSize);
      string_class Opts(OptsSize, '\0');
      PI_CALL(piProgramGetBuildInfo)(Program, Device, CL_PROGRAM_BUILD_OPTIONS,
                                     OptsSize, &Opts.front(), nullptr);
      if (NonUniformWGs) {
        const bool HasStd20 = Opts.find("-cl-std=CL2.0") != string_class::npos;
        if (!HasStd20)
          throw sycl::nd_range_error(
              "Non-uniform work-groups are not allowed by default. Underlying "
              "OpenCL 2.x implementation supports this feature and to enable "
              "it, build device program with -cl-std=CL2.0",
              PI_INVALID_WORK_GROUP_SIZE);
        else
          throw sycl::nd_range_error(
              "Non-uniform work-groups are not allowed by default. Underlying "
              "OpenCL 2.x implementation supports this feature, but it is "
              "disabled by -cl-uniform-work-group-size build flag",
              PI_INVALID_WORK_GROUP_SIZE);
      }
    }
  }

  // TODO: required number of sub-groups, OpenCL 2.1:
  // CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified and is not
  // consistent with the required number of sub-groups for kernel in the
  // program source.

  // Fallback
  constexpr pi_result Error = PI_INVALID_WORK_GROUP_SIZE;
  throw runtime_error(
      "OpenCL API failed. OpenCL API returns: " + codeToString(Error), Error);
}

bool handleError(pi_result Error, pi_device Device, pi_kernel Kernel,
                 const NDRDescT &NDRDesc) {
  assert(Error != PI_SUCCESS &&
         "Success is expected to be handled on caller side");
  switch (Error) {
  case PI_INVALID_WORK_GROUP_SIZE:
    return handleInvalidWorkGroupSize(Device, Kernel, NDRDesc);
  // TODO: Handle other error codes
  default:
    throw runtime_error(
        "OpenCL API failed. OpenCL API returns: " + codeToString(Error), Error);
  }
}

} // namespace enqueue_kernel_launch

} // namespace detail
} // namespace sycl
} // namespace cl
