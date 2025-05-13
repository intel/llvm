//===--------- command_buffer.hpp - OpenCL Adapter ---------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/cl_ext.h>
#include <ur/ur.hpp>

/// Handle to a kernel command.
struct ur_exp_command_buffer_command_handle_t_ {
  /// Command-buffer this command belongs to.
  ur_exp_command_buffer_handle_t hCommandBuffer;
  /// OpenCL command-handle.
  cl_mutable_command_khr CLMutableCommand;
  /// Kernel associated with this command handle
  ur_kernel_handle_t Kernel;
  /// Work-dimension the command was originally created with.
  cl_uint WorkDim;
  /// Set to true if the user set the local work size on command creation.
  bool UserDefinedLocalSize;

  ur_exp_command_buffer_command_handle_t_(
      ur_exp_command_buffer_handle_t hCommandBuffer,
      cl_mutable_command_khr CLMutableCommand, ur_kernel_handle_t Kernel,
      cl_uint WorkDim, bool UserDefinedLocalSize)
      : hCommandBuffer(hCommandBuffer), CLMutableCommand(CLMutableCommand),
        Kernel(Kernel), WorkDim(WorkDim),
        UserDefinedLocalSize(UserDefinedLocalSize) {}
};

/// Handle to a command-buffer object.
struct ur_exp_command_buffer_handle_t_ {
  /// UR queue belonging to the command-buffer, required for OpenCL creation.
  ur_queue_handle_t hInternalQueue;
  /// Context the command-buffer is created for.
  ur_context_handle_t hContext;
  /// Device the command-buffer is created for.
  ur_device_handle_t hDevice;
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
  /// Object reference count
  std::atomic_uint32_t RefCount;
  /// Track last submission of the command-buffer
  cl_event LastSubmission;

  ur_exp_command_buffer_handle_t_(ur_queue_handle_t hQueue,
                                  ur_context_handle_t hContext,
                                  ur_device_handle_t hDevice,
                                  cl_command_buffer_khr CLCommandBuffer,
                                  bool IsUpdatable, bool IsInOrder)
      : hInternalQueue(hQueue), hContext(hContext), hDevice(hDevice),
        CLCommandBuffer(CLCommandBuffer), IsUpdatable(IsUpdatable),
        IsInOrder(IsInOrder), IsFinalized(false), RefCount(0),
        LastSubmission(nullptr) {}

  ~ur_exp_command_buffer_handle_t_();

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }
  uint32_t decrementReferenceCount() noexcept { return --RefCount; }
  uint32_t getReferenceCount() const noexcept { return RefCount; }
};
