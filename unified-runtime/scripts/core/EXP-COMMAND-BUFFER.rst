<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-command-buffer:

================================================================================
Command-Buffer
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------
A command-buffer represents a series of commands for execution on a command
queue. Many adapters support this kind of construct either natively or through
extensions, but they are not available to use directly. Typically their use is
abstracted through the existing Core APIs, for example when calling
${x}EnqueueKernelLaunch the adapter may both append the kernel command to a
command-buffer-like construct and also submit that command-buffer to a queue for
execution. These types of structures allow for batching of commands to improve
host launch latency, but without direct control it falls to the adapter
implementation to implement automatic batching of commands.

This experimental feature exposes command-buffers in the Unified Runtime API
directly, allowing applications explicit control over the enqueue and execution
of commands to batch commands as required for optimal performance.

Querying Command-Buffer Support
--------------------------------------------------------------------------------

Support for command-buffers can be queried for a given device/adapter by using
the device info query with ${X}_DEVICE_INFO_EXTENSIONS. Adapters supporting this
experimental feature will report the string "ur_exp_command_buffer" in the
returned list of supported extensions.

.. hint::
    The macro ${X}_COMMAND_BUFFER_EXTENSION_STRING_EXP is defined for the string
    returned from extension queries for this feature. Since the actual string
    may be subject to change it is safer to use this macro when querying for
    support for this experimental feature.

.. parsed-literal::

    // Retrieve length of extension string
    size_t returnedSize;
    ${x}DeviceGetInfo(hDevice, ${X}_DEVICE_INFO_EXTENSIONS, 0, nullptr,
                    &returnedSize);

    // Retrieve extension string
    std::unique_ptr<char[]> returnedExtensions(new char[returnedSize]);
    ${x}DeviceGetInfo(hDevice, ${X}_DEVICE_INFO_EXTENSIONS, returnedSize,
                      returnedExtensions.get(), nullptr);

    std::string_view ExtensionsString(returnedExtensions.get());
    bool CmdBufferSupport =
        ExtensionsString.find(${X}_COMMAND_BUFFER_EXTENSION_STRING_EXP)
            != std::string::npos;

.. note::
    The ${X}_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP device info query exists to
    serve the same purpose as ${X}_COMMAND_BUFFER_EXTENSION_STRING_EXP.

Command-Buffer Creation
--------------------------------------------------------------------------------

Command-Buffers are tied to a specific ${x}_context_handle_t and
${x}_device_handle_t. ${x}CommandBufferCreateExp optionally takes a descriptor
to provide additional properties for how the command-buffer should be
constructed. The members defined in ${x}_exp_command_buffer_desc_t are:

* ``isUpdatable``, which should be set to ``true`` to support :ref:`updating
command-buffer commands`.
* ``isInOrder``, which should be set to ``true`` to enable commands enqueued to
a command-buffer to be executed in an in-order fashion where possible.
* ``enableProfiling``, which should be set to ``true`` to enable profiling of
the command-buffer.

Command-buffers are reference counted and can be retained and released by
calling ${x}CommandBufferRetainExp and ${x}CommandBufferReleaseExp respectively.

Appending Commands
--------------------------------------------------------------------------------

Commands can be appended to a command-buffer by calling any of the
command-buffer append functions. Typically these closely mimic the existing
enqueue functions in the Core API in terms of their command-specific parameters.
However, they differ in that they take a command-buffer handle instead of a
queue handle. Dependencies are also expressed differently, in that internal
command-buffer dependencies are expressed with sync-points. While event handles
are used to express synchronization external to the command-buffer.

The entry-points for appending commands also return an optional handle to the
command being appended. This handle can be used to update the command
configuration between command-buffer executions, see the section on
:ref:`updating command-buffer commands`.

Currently only the following commands are supported:

* ${x}CommandBufferAppendKernelLaunchExp
* ${x}CommandBufferAppendUSMMemcpyExp
* ${x}CommandBufferAppendUSMFillExp
* ${x}CommandBufferAppendMemBufferCopyExp
* ${x}CommandBufferAppendMemBufferCopyRectExp
* ${x}CommandBufferAppendMemBufferReadExp
* ${x}CommandBufferAppendMemBufferReadRectExp
* ${x}CommandBufferAppendMemBufferWriteExp
* ${x}CommandBufferAppendMemBufferWriteRectExp
* ${x}CommandBufferAppendMemBufferFillExp
* ${x}CommandBufferAppendUSMPrefetchExp
* ${x}CommandBufferAppendUSMAdviseExp

It is planned to eventually support any command type from the Core API which can
actually be appended to the equivalent adapter native constructs.

Sync-Points
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A sync-point is a value which represents a command inside of a command-buffer
which is returned from command-buffer append function calls. These can be
optionally passed to these functions to define execution dependencies on other
commands within the command-buffer. Sync-points passed to functions may be
ignored if the command-buffer was created in-order.

Sync-points are unique and valid for use only within the command-buffer they
were obtained from.

.. parsed-literal::
    // Append a memcpy with no sync-point dependencies
    ${x}_exp_command_buffer_sync_point_t syncPoint;

    ${x}CommandBufferAppendUSMMemcpyExp(hCommandBuffer, pDst, pSrc, size, 0,
                                        nullptr, 0, nullptr, &syncPoint, nullptr,
                                        nullptr);

    // Append a kernel launch with syncPoint as a dependency, ignore returned
    // sync-point
    ${x}CommandBufferAppendKernelLaunchExp(hCommandBuffer, hKernel, workDim,
                                           pGlobalWorkOffset, pGlobalWorkSize,
                                           pLocalWorkSize, 0, nullptr, 1,
                                           &syncPoint, 0, nullptr,
                                           nullptr, nullptr, nullptr);

Command Synchronization With Events
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

When appending commands to a command-buffer an optional ``phEventWaitList``
input parameter is available for passing a list of ${x}_event_handle_t objects
the command should wait on. As well as an optional ``phEvent`` output parameter
to get a ${x}_event_handle_t object that will be signaled on completion of the
command execution. It is the users responsibility to release the returned
``phEvent`` with ${x}EventRelease.

The wait event parameter allows commands in a command-buffer to depend on the
completion of UR commands which are external to a command-buffer. While the
output signal event parameter allows individual commands in a command-buffer to
trigger external queue commands. Using returned signal events as wait events
inside the same command-buffer is also valid usage.

.. important::
   Support for using ``phEventWaitList`` & ``phEvent`` parameters requires a device
   to support ${X}_DEVICE_INFO_COMMAND_BUFFER_EVENT_SUPPORT_EXP.

Signal Event Valid Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A returned signal event represents only the status of the command in the
current execution of a given command-buffer on a device. Command signal events
are not unique per execution of a command-buffer. If a command-buffer is
enqueued multiple times before using one of these events (for example as a
dependency to an eager queue operation), it is undefined which specific
execution of the command-buffer the event will represent. If a dependency on a
specific graph command-buffer execution is required this ordering must be
enforced by the user to ensure there is only a single command-buffer execution
in flight when using these command signal events.

When a user calls ${x}CommandBufferEnqueueExp all the signal events returned
from the individual commands in the command-buffer are synchronously reset to
a non-complete state prior to the asynchronous commands beginning.

Inter-Graph Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible for commands in different command-buffer objects to synchronize
using the event mechanism. This is only guaranteed to behave correctly in the one
directional synchronization case, where the signal events of one
command-buffer's commands are used as a wait events of another command-buffer's
commands. Such a relationship defines a permanent dependency between the
command-buffers which does not need to be updated using
:ref:`command event update` to preserve synchronization on future enqueues of
the command-buffer.

Bi-directional sync between individual commands in two separate command-buffers
is however not guaranteed to behave correctly. This is due to the completion
state of the command events only being reset when a command-buffer is enqueued.
It is therefore possible for the first command-buffer enqueued to execute its
wait node that needs to have its event reset by the enqueue of the second
command-buffer, before the code path returns to user code for the user to
enqueue the second command-buffer. Resulting in the first command-buffer's
wait node completing too early for the intended overall executing ordering.

Enqueueing Command-Buffers
--------------------------------------------------------------------------------

Command-buffers are submitted for execution on a ${x}_queue_handle_t with an
optional list of dependent events. An event is returned which tracks the
execution of the command-buffer, and will be complete when all appended commands
have finished executing. It is adapter specific whether command-buffers can be
enqueued or executed simultaneously, and submissions may be serialized.

.. parsed-literal::
    ${x}_event_handle_t executionEvent;

    ${x}CommandBufferEnqueueExp(hCommandBuffer, hQueue, 0, nullptr,
                              &executionEvent);


Updating Command-Buffer Commands
--------------------------------------------------------------------------------

An adapter implementing the command-buffer experimental feature can optionally
support updating the configuration of kernel commands recorded to a
command-buffer. The attributes of kernel commands that can be updated are
device specific and can be queried using the
${X}_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP query.

All update entry-points are synchronous and may block if the command-buffer is
executing when the entry-point is called.

Kernel Argument Update
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Kernel commands can have the ND-Range & parameter arguments of the command
updated when a device supports the relevant bits in
${X}_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP.

Updating kernel commands is done by passing the new kernel configuration
to ${x}CommandBufferUpdateKernelLaunchExp along with the command handle of
the kernel command to update. Configurations that can be changed are the
kernel handle, the parameters to the kernel and the execution ND-Range.

Kernel handles that might be used to update the kernel of a command, need
to be registered when the command is created. This can be done
using the ``phKernelAlternatives`` parameter of
${x}CommandBufferAppendKernelLaunchExp. The command can then be updated
to use the new kernel handle by passing it to
${x}CommandBufferUpdateKernelLaunchExp.

.. important::
    When updating the kernel handle of a command all required arguments to the
    new kernel must be provided in the update descriptor. Failure to do so will
    result in undefined behavior.

.. parsed-literal::

    // Create a command-buffer with update enabled.
    ${x}_exp_command_buffer_desc_t desc {
      ${X}_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC,
      nullptr,
      true // isUpdatable
    };
    ${x}_exp_command_buffer_handle_t hCommandBuffer;
    ${x}CommandBufferCreateExp(hContext, hDevice, &desc, &hCommandBuffer);

    // Append a kernel command which has two buffer parameters, an input
    // and an output. Register hNewKernel as an alternative kernel handle
    // which can later be used to change the kernel handle associated
    // with this command.
    ${x}_exp_command_buffer_command_handle_t hCommand;
    ${x}CommandBufferAppendKernelLaunchExp(hCommandBuffer, hKernel, workDim,
                                           pGlobalWorkOffset, pGlobalWorkSize,
                                           pLocalWorkSize, 1, &hNewKernel,
                                           0, nullptr, 0, nullptr, nullptr,
                                           nullptr, &hCommand);

    // Close the command-buffer before updating
    ${x}CommandBufferFinalizeExp(hCommandBuffer);

    // Define kernel argument at index 0 to be a new input buffer object
    ${x}_exp_command_buffer_update_memobj_arg_desc_t newInputArg {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_MEMOBJ_ARG_DESC, // stype
        nullptr, // pNext
        0, // argIndex
        nullptr, // pProperties
        newInputBuffer, // hNewMemObjArg
    };

    // Define kernel argument at index 1 to be a new output buffer object
    ${x}_exp_command_buffer_update_memobj_arg_desc_t newOutputArg {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_MEMOBJ_ARG_DESC, // stype
        nullptr, // pNext
        1, // argIndex
        nullptr, // pProperties
        newOutputBuffer, // hNewMemObjArg
    };

    // Define the new configuration of the kernel command
    ${x}_exp_command_buffer_update_memobj_arg_desc_t updatedArgs[2] = {newInputArg, newOutputArg};
    ${x}_exp_command_buffer_update_kernel_launch_desc_t update {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr, // pNext
        hCommand, // hCommand
        hNewKernel,  // hNewKernel
        2, // numNewMemobjArgs
        0, // numNewPointerArgs
        0, // numNewValueArgs
        0, // numNewExecInfos
        0, // newWorkDim
        new_args, // pNewMemObjArgList
        nullptr, // pNewPointerArgList
        nullptr, // pNewValueArgList
        nullptr, // pNewExecInfoList
        nullptr, // pNewGlobalWorkOffset
        nullptr, // pNewGlobalWorkSize
        nullptr, // pNewLocalWorkSize
    };

    // Perform the update
    ${x}CommandBufferUpdateKernelLaunchExp(hCommandBuffer, 1, &update);

Command Event Update
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Once a command-buffer has been finalized the wait-list parameter of the command
can be updated with ${x}CommandBufferUpdateWaitEventsExp. The number of wait
events for a command must stay consistent, therefore the number of events
passed to ${x}CommandBufferUpdateWaitEventsExp must be the same as when the
command was created.

The ${x}CommandBufferUpdateSignalEventExp entry-points can be used to update
the signal event of a command. This returns a new event that will be signaled
on the next execution of the command in the command-buffer. It may be that
this is backed by the same native event object as the original signal event,
provided that the backend provides a way to reset or reuse events between
command-buffer executions.

As ${x}_event_handle_t objects for queue submissions can only be signaled once,
and not reset, this update mechanism allows command synchronization to be
refreshed between command-buffer executions with regular command-queue events
that haven't yet been signaled.

It is the users responsibility to release the returned ``phEvent`` with
${x}EventRelease. To update a command signal event with
${x}CommandBufferUpdateSignalEventExp there must also have been a non-null
``phEvent`` parameter passed on command creation.

.. important::
   Support for updating ``phEventWaitList`` & ``phEvent`` parameters requires a
   device to support the `EVENTS` bit in
   ${X}_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP.

.. parsed-literal::

    // Create a command-buffer with update enabled.
    ${x}_exp_command_buffer_desc_t desc {
      ${X}_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC,
      nullptr,
      true // isUpdatable
    };
    ${x}_exp_command_buffer_handle_t hCommandBuffer;
    ${x}CommandBufferCreateExp(hContext, hDevice, &desc, &hCommandBuffer);

    // Append a kernel command with 2 events to wait on, and returning an
    // event that will be signaled.
    ${x}_event_handle_t hSignalEvent;
    ${x}_event_handle_t hWaitEvents[2] = {...};
    ${x}_exp_command_buffer_command_handle_t hCommand;
    ${x}CommandBufferAppendKernelLaunchExp(hCommandBuffer, hKernel, workDim,
                                           pGlobalWorkOffset, pGlobalWorkSize,
                                           pLocalWorkSize, 0, nullptr, 0, nullptr,
                                           2, hWaitEvents, nullptr, &hSignalEvent,
                                           &hCommand);

    // Close the command-buffer before updating
    ${x}CommandBufferFinalizeExp(hCommandBuffer);

    // Enqueue command-buffer
    ${x}CommandBufferEnqueueExp(hCommandBuffer, hQueue, 0, nullptr, nullptr);

    // Wait for command-buffer to finish
    ${x}QueueFinish(hQueue);

    // Update signal event
    ${x}_event_handle_t hNewSignalEvent;
    ${x}CommandBufferUpdateSignalEventExp(hCommand, &hNewSignalEvent);

    // Update wait events to a new event
    ${x}_event_handle_t hNewWaitEvents[2] = ...;
    {x}CommandBufferUpdateWaitEventsExp(hCommand, 2, &hNewWaitEvents);


API
--------------------------------------------------------------------------------

Macros
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${X}_COMMAND_BUFFER_EXTENSION_STRING_EXP

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP
    * ${X}_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP
    * ${X}_DEVICE_INFO_COMMAND_BUFFER_EVENT_SUPPORT_EXP
* ${x}_device_command_buffer_update_capability_flags_t
    * UPDATE_KERNEL_ARGUMENTS
    * LOCAL_WORK_SIZE
    * GLOBAL_WORK_SIZE
    * GLOBAL_WORK_OFFSET
    * KERNEL_HANDLE
    * EVENTS
* ${x}_result_t
    * ${X}_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
    * ${X}_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
    * ${X}_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
    * ${X}_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP
* ${x}_structure_type_t
    * ${X}_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC
    * ${X}_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC
    * ${X}_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_MEMOBJ_ARG_DESC
    * ${X}_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC
    * ${X}_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC
* ${x}_command_t
    * ${X}_COMMAND_COMMAND_BUFFER_ENQUEUE_EXP
* ${x}_function_t
    * ${X}_FUNCTION_COMMAND_BUFFER_CREATE_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_RETAIN_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_RELEASE_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_FINALIZE_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_APPEND_KERNEL_LAUNCH_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_ENQUEUE_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_APPEND_USM_MEMCPY_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_APPEND_USM_FILL_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_COPY_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_COPY_RECT_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_READ_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_READ_RECT_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_WRITE_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_WRITE_RECT_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_APPEND_MEM_BUFFER_FILL_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_APPEND_USM_PREFETCH_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_APPEND_USM_ADVISE_EXP
    * ${X}_FUNCTION_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_EXP
* ${x}_exp_command_buffer_info_t
    * ${X}_EXP_COMMAND_BUFFER_INFO_REFERENCE_COUNT
    * ${X}_EXP_COMMAND_BUFFER_INFO_DESCRIPTOR
* ${x}_exp_command_buffer_command_info_t
    * ${X}_EXP_COMMAND_BUFFER_COMMAND_INFO_REFERENCE_COUNT

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_exp_command_buffer_desc_t
* ${x}_exp_command_buffer_update_kernel_launch_desc_t
* ${x}_exp_command_buffer_update_memobj_arg_desc_t
* ${x}_exp_command_buffer_update_pointer_arg_desc_t
* ${x}_exp_command_buffer_update_value_arg_desc_t
* ${x}_exp_command_buffer_sync_point_t
* ${x}_exp_command_buffer_handle_t
* ${x}_exp_command_buffer_command_handle_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}CommandBufferCreateExp
* ${x}CommandBufferRetainExp
* ${x}CommandBufferReleaseExp
* ${x}CommandBufferFinalizeExp
* ${x}CommandBufferAppendKernelLaunchExp
* ${x}CommandBufferAppendUSMMemcpyExp
* ${x}CommandBufferAppendUSMFillExp
* ${x}CommandBufferAppendMemBufferCopyExp
* ${x}CommandBufferAppendMemBufferCopyRectExp
* ${x}CommandBufferAppendMemBufferReadExp
* ${x}CommandBufferAppendMemBufferReadRectExp
* ${x}CommandBufferAppendMemBufferWriteExp
* ${x}CommandBufferAppendMemBufferWriteRectExp
* ${x}CommandBufferAppendMemBufferFillExp
* ${x}CommandBufferAppendUSMPrefetchExp
* ${x}CommandBufferAppendUSMAdviseExp
* ${x}CommandBufferEnqueueExp
* ${x}CommandBufferUpdateKernelLaunchExp
* ${x}CommandBufferUpdateSignalEventExp
* ${x}CommandBufferUpdateWaitEventsExp
* ${x}CommandBufferGetInfoExp

Changelog
--------------------------------------------------------------------------------

+-----------+-------------------------------------------------------+
| Revision  | Changes                                               |
+===========+=======================================================+
| 1.0       | Initial Draft                                         |
+-----------+-------------------------------------------------------+
| 1.1       | Add function definitions for buffer read and write    |
+-----------+-------------------------------------------------------+
| 1.2       | Add function definitions for fill commands            |
+-----------+-------------------------------------------------------+
| 1.3       | Add function definitions for Prefetch and Advise      |
|           | commands                                              |
+-----------+-------------------------------------------------------+
| 1.4       | Add function definitions for kernel command update    |
+-----------+-------------------------------------------------------+
| 1.5       | Add support for updating kernel handles.              |
+-----------+-------------------------------------------------------+
| 1.6       | Command level synchronization with event objects      |
+-----------+-------------------------------------------------------+
| 1.7       | Remove command handle reference counting and querying |
+-----------+-------------------------------------------------------+
| 1.8       | Change Kernel command update API to take a list       |
+-----------+-------------------------------------------------------+

Contributors
--------------------------------------------------------------------------------

* Ben Tracy `ben.tracy@codeplay.com <ben.tracy@codeplay.com>`_
* Ewan Crawford `ewan@codeplay.com <ewan@codeplay.com>`_
* Maxime France-Pillois `maxime.francepillois@codeplay.com <maxime.francepillois@codeplay.com>`_
* Aaron Greig `aaron.greig@codeplay.com <aaron.greig@codeplay.com>`_
* Fábio Mestre `fabio.mestre@codeplay.com <fabio.mestre@codeplay.com>`_
