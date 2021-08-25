//==---- CommonRedefinitions.hpp --- Header with common PI redefinitions ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

inline pi_result redefinedProgramCreateCommon(pi_context, const void *, size_t,
                                              pi_program *ret_program) {
  *ret_program = reinterpret_cast<pi_program>(1);
  return PI_SUCCESS;
}

inline pi_result redefinedProgramBuildCommon(
    pi_program prog, pi_uint32, const pi_device *, const char *,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  if (pfn_notify) {
    pfn_notify(prog, user_data);
  }
  return PI_SUCCESS;
}

inline pi_result redefinedProgramCompileCommon(
    pi_program, pi_uint32, const pi_device *, const char *, pi_uint32,
    const pi_program *, const char **, void (*)(pi_program, void *), void *) {
  return PI_SUCCESS;
}

inline pi_result redefinedProgramLinkCommon(pi_context, pi_uint32,
                                            const pi_device *, const char *,
                                            pi_uint32, const pi_program *,
                                            void (*)(pi_program, void *),
                                            void *, pi_program *ret_program) {
  *ret_program = reinterpret_cast<pi_program>(1);
  return PI_SUCCESS;
}

inline pi_result redefinedProgramGetInfoCommon(pi_program program,
                                               pi_program_info param_name,
                                               size_t param_value_size,
                                               void *param_value,
                                               size_t *param_value_size_ret) {
  if (param_name == PI_PROGRAM_INFO_NUM_DEVICES) {
    auto value = reinterpret_cast<unsigned int *>(param_value);
    *value = 1;
  }

  if (param_name == PI_PROGRAM_INFO_BINARY_SIZES) {
    auto value = reinterpret_cast<size_t *>(param_value);
    value[0] = 1;
  }

  if (param_name == PI_PROGRAM_INFO_BINARIES) {
    auto value = reinterpret_cast<unsigned char *>(param_value);
    value[0] = 1;
  }

  return PI_SUCCESS;
}

inline pi_result redefinedProgramRetainCommon(pi_program program) {
  return PI_SUCCESS;
}

inline pi_result redefinedProgramReleaseCommon(pi_program program) {
  return PI_SUCCESS;
}

inline pi_result redefinedKernelCreateCommon(pi_program program,
                                             const char *kernel_name,
                                             pi_kernel *ret_kernel) {
  *ret_kernel = reinterpret_cast<pi_kernel>(1);
  return PI_SUCCESS;
}

inline pi_result redefinedKernelRetainCommon(pi_kernel kernel) {
  return PI_SUCCESS;
}

inline pi_result redefinedKernelReleaseCommon(pi_kernel kernel) {
  return PI_SUCCESS;
}

inline pi_result redefinedKernelGetInfoCommon(pi_kernel kernel,
                                              pi_kernel_info param_name,
                                              size_t param_value_size,
                                              void *param_value,
                                              size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

inline pi_result redefinedKernelSetExecInfoCommon(
    pi_kernel kernel, pi_kernel_exec_info value_name, size_t param_value_size,
    const void *param_value) {
  return PI_SUCCESS;
}

inline pi_result redefinedEventsWaitCommon(pi_uint32 num_events,
                                           const pi_event *event_list) {
  return PI_SUCCESS;
}

inline pi_result redefinedEventReleaseCommon(pi_event event) {
  if (event != nullptr)
    delete reinterpret_cast<int *>(event);
  return PI_SUCCESS;
}

inline pi_result redefinedEnqueueKernelLaunchCommon(
    pi_queue, pi_kernel, pi_uint32, const size_t *, const size_t *,
    const size_t *, pi_uint32, const pi_event *, pi_event *event) {
  *event = reinterpret_cast<pi_event>(new int{});
  return PI_SUCCESS;
}

inline pi_result redefinedKernelGetGroupInfoCommon(
    pi_kernel kernel, pi_device device, pi_kernel_group_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  if (param_name == PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE && param_value) {
    auto RealVal = reinterpret_cast<size_t *>(param_value);
    RealVal[0] = 0;
    RealVal[1] = 0;
    RealVal[2] = 0;
  }
  return PI_SUCCESS;
}

inline void setupDefaultMockAPIs(sycl::unittest::PiMock &Mock) {
  using namespace sycl::detail;
  Mock.redefine<PiApiKind::piProgramCreate>(redefinedProgramCreateCommon);
  Mock.redefine<PiApiKind::piProgramCompile>(redefinedProgramCompileCommon);
  Mock.redefine<PiApiKind::piProgramLink>(redefinedProgramLinkCommon);
  Mock.redefine<PiApiKind::piProgramBuild>(redefinedProgramBuildCommon);
  Mock.redefine<PiApiKind::piProgramGetInfo>(redefinedProgramGetInfoCommon);
  Mock.redefine<PiApiKind::piProgramRetain>(redefinedProgramRetainCommon);
  Mock.redefine<PiApiKind::piProgramRelease>(redefinedProgramReleaseCommon);
  Mock.redefine<PiApiKind::piKernelCreate>(redefinedKernelCreateCommon);
  Mock.redefine<PiApiKind::piKernelRetain>(redefinedKernelRetainCommon);
  Mock.redefine<PiApiKind::piKernelRelease>(redefinedKernelReleaseCommon);
  Mock.redefine<PiApiKind::piKernelGetInfo>(redefinedKernelGetInfoCommon);
  Mock.redefine<PiApiKind::piKernelGetGroupInfo>(
      redefinedKernelGetGroupInfoCommon);
  Mock.redefine<PiApiKind::piKernelSetExecInfo>(
      redefinedKernelSetExecInfoCommon);
  Mock.redefine<PiApiKind::piEventsWait>(redefinedEventsWaitCommon);
  Mock.redefine<PiApiKind::piEventRelease>(redefinedEventReleaseCommon);
  Mock.redefine<PiApiKind::piEnqueueKernelLaunch>(
      redefinedEnqueueKernelLaunchCommon);
}
