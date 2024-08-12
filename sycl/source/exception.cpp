//==---------------- exception.cpp - SYCL exception ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 4.9.2 Exception Class Interface
#include <detail/global_handler.hpp>
#include <sycl/context.hpp>
#include <sycl/exception.hpp>

#include <cstring>
#include <sstream>

namespace sycl {
inline namespace _V1 {

exception::exception(std::error_code EC, const char *Msg)
    : exception(EC, nullptr, Msg) {}

// new SYCL 2020 constructors
exception::exception(std::error_code EC) : exception(EC, nullptr, "") {}

exception::exception(int EV, const std::error_category &ECat,
                     const char *WhatArg)
    : exception({EV, ECat}, nullptr, std::string(WhatArg)) {}

exception::exception(int EV, const std::error_category &ECat)
    : exception({EV, ECat}, nullptr, "") {}

// protected base constructor for all SYCL 2020 constructors
exception::exception(std::error_code EC, std::shared_ptr<context> SharedPtrCtx,
                     const char *WhatArg)
    : MMsg(std::make_shared<detail::string>(WhatArg)),
      MErr(UR_RESULT_ERROR_INVALID_VALUE), MContext(SharedPtrCtx), MErrC(EC) {
  detail::GlobalHandler::instance().TraceEventXPTI(MMsg->c_str());
}

exception::~exception() {}

const std::error_code &exception::code() const noexcept { return MErrC; }

const std::error_category &exception::category() const noexcept {
  return code().category();
}

const char *exception::what() const noexcept { return MMsg->c_str(); }

bool exception::has_context() const noexcept { return (MContext != nullptr); }

context exception::get_context() const {
  if (!has_context())
    throw sycl::exception(sycl::errc::invalid);

  return *MContext;
}

const std::error_category &sycl_category() noexcept {
  static const detail::SYCLCategory SYCLCategoryObj;
  return SYCLCategoryObj;
}

std::error_code make_error_code(sycl::errc Err) noexcept {
  return {static_cast<int>(Err), sycl_category()};
}

namespace detail {
__SYCL_EXPORT const char *stringifyErrorCode(int32_t error) {
  switch (error) {
#define _UR_ERRC(NAME)                                                         \
  case NAME:                                                                   \
    return #NAME;
    // TODO: bring back old code specific messages?
#define _UR_ERRC_WITH_MSG(NAME, MSG)                                           \
  case NAME:                                                                   \
    return MSG;
    _UR_ERRC(UR_RESULT_SUCCESS)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_OPERATION)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_QUEUE)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_VALUE)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_CONTEXT)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_PLATFORM)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_BINARY)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_PROGRAM)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_SAMPLER)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_BUFFER_SIZE)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_MEM_OBJECT)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_EVENT)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST)
    _UR_ERRC(UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE)
    _UR_ERRC(UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE)
    _UR_ERRC(UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE)
    _UR_ERRC(UR_RESULT_ERROR_DEVICE_NOT_FOUND)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_DEVICE)
    _UR_ERRC(UR_RESULT_ERROR_DEVICE_LOST)
    _UR_ERRC(UR_RESULT_ERROR_DEVICE_REQUIRES_RESET)
    _UR_ERRC(UR_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE)
    _UR_ERRC(UR_RESULT_ERROR_DEVICE_PARTITION_FAILED)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_WORK_DIMENSION)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_KERNEL_ARGS)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_KERNEL)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_KERNEL_NAME)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_IMAGE_SIZE)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR)
    _UR_ERRC(UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE)
    _UR_ERRC(UR_RESULT_ERROR_UNINITIALIZED)
    _UR_ERRC(UR_RESULT_ERROR_OUT_OF_HOST_MEMORY)
    _UR_ERRC(UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
    _UR_ERRC(UR_RESULT_ERROR_OUT_OF_RESOURCES)
    _UR_ERRC(UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE)
    _UR_ERRC(UR_RESULT_ERROR_PROGRAM_LINK_FAILURE)
    _UR_ERRC(UR_RESULT_ERROR_UNSUPPORTED_VERSION)
    _UR_ERRC(UR_RESULT_ERROR_UNSUPPORTED_FEATURE)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_ARGUMENT)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_NULL_HANDLE)
    _UR_ERRC(UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_NULL_POINTER)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_SIZE)
    _UR_ERRC(UR_RESULT_ERROR_UNSUPPORTED_SIZE)
    _UR_ERRC(UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_ENUMERATION)
    _UR_ERRC(UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION)
    _UR_ERRC(UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_NATIVE_BINARY)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_GLOBAL_NAME)
    _UR_ERRC(UR_RESULT_ERROR_FUNCTION_ADDRESS_NOT_AVAILABLE)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION)
    _UR_ERRC(UR_RESULT_ERROR_PROGRAM_UNLINKED)
    _UR_ERRC(UR_RESULT_ERROR_OVERLAPPING_REGIONS)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_HOST_PTR)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_USM_SIZE)
    _UR_ERRC(UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE)
    _UR_ERRC(UR_RESULT_ERROR_ADAPTER_SPECIFIC)
    _UR_ERRC(UR_RESULT_ERROR_LAYER_NOT_PRESENT)
    _UR_ERRC(UR_RESULT_ERROR_IN_EVENT_LIST_EXEC_STATUS)
    _UR_ERRC(UR_RESULT_ERROR_DEVICE_NOT_AVAILABLE)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP)
    _UR_ERRC(UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP)
    _UR_ERRC(UR_RESULT_ERROR_UNKNOWN)
#undef _UR_ERRC
#undef _UR_ERRC_WITH_MSG

  default:
    return "Unknown error code";
  }
}
} // namespace detail

} // namespace _V1
} // namespace sycl
