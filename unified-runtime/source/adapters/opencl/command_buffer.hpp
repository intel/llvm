//===--------- command_buffer.hpp - OpenCL Adapter ---------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "common/ur_ref_count.hpp"
#include <ur/ur.hpp>

namespace ur::opencl {

/// Handle to a kernel command.
struct ur_exp_command_buffer_command_handle_t_ : handle_base {
  /// Command-buffer this command belongs to.
  ur_exp_command_buffer_handle_t_ *hCommandBuffer;
  /// OpenCL command-handle.
  cl_mutable_command_khr CLMutableCommand;
  /// Kernel associated with this command handle
  ur_kernel_handle_t_ *Kernel;
  /// Work-dimension the command was originally created with.
  cl_uint WorkDim;
  /// Set to true if the user set the local work size on command creation.
  bool UserDefinedLocalSize;

  ur_exp_command_buffer_command_handle_t_(
      ur_exp_command_buffer_handle_t_ *hCommandBuffer,
      cl_mutable_command_khr CLMutableCommand, ur_kernel_handle_t_ *Kernel,
      cl_uint WorkDim, bool UserDefinedLocalSize)
      : handle_base(), hCommandBuffer(hCommandBuffer),
        CLMutableCommand(CLMutableCommand), Kernel(Kernel), WorkDim(WorkDim),
        UserDefinedLocalSize(UserDefinedLocalSize) {}
};

/// Handle to a command-buffer object.
struct ur_exp_command_buffer_handle_t_ : handle_base {
  /// UR queue belonging to the command-buffer, required for OpenCL creation.
  ur_queue_handle_t_ *hInternalQueue;
  /// Context the command-buffer is created for.
  ur_context_handle_t_ *hContext;
  /// Device the command-buffer is created for.
  ur_device_handle_t_ *hDevice;
  /// OpenCL command-buffer object.
  cl_command_buffer_khr CLCommandBuffer;
  /// Set to true if the kernel commands in the command-buffer can be updated,
  /// false otherwise
  bool IsUpdatable;
  /// Set to true if the command-buffer was created with an in-order queue.
  bool IsInOrder;
  /// Set to true if the command-buffer has been finalized, false otherwise
  bool IsFinalized;
  /// List of commands in the command-buffer.
  std::vector<std::unique_ptr<ur_exp_command_buffer_command_handle_t_>>
      CommandHandles;
  /// Track last submission of the command-buffer
  cl_event LastSubmission;

  ur::RefCount RefCount;

  ur_exp_command_buffer_handle_t_(ur_queue_handle_t_ *hQueue,
                                  ur_context_handle_t_ *hContext,
                                  ur_device_handle_t_ *hDevice,
                                  cl_command_buffer_khr CLCommandBuffer,
                                  bool IsUpdatable, bool IsInOrder)
      : handle_base(), hInternalQueue(hQueue), hContext(hContext),
        hDevice(hDevice), CLCommandBuffer(CLCommandBuffer),
        IsUpdatable(IsUpdatable), IsInOrder(IsInOrder), IsFinalized(false),
        LastSubmission(nullptr) {}

  ~ur_exp_command_buffer_handle_t_();
};

} // namespace ur::opencl
