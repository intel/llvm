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

Command-Buffer Creation
--------------------------------------------------------------------------------

Command-Buffers are tied to a specific ${x}_context_handle_t and
${x}_device_handle_t. ${x}CommandBufferCreateExp optionally takes a descriptor
to provide additional properties for how the command-buffer should be
constructed. There are currently no unique members defined for
${x}_exp_command_buffer_desc_t, however they may be added in the future.

Command-buffers are reference counted and can be retained and released by
calling ${x}CommandBufferRetainExp and ${x}CommandBufferReleaseExp respectively.

Appending Commands
--------------------------------------------------------------------------------

Commands can be appended to a command-buffer by calling any of the
command-buffer append functions. Typically these closely mimic the existing
enqueue functions in the Core API in terms of their command-specific parameters.
However, they differ in that they take a command-buffer handle instead of a
queue handle, and the dependencies and return parameters are sync-points instead
of event handles.

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
actually be appended to the equiavalent adapter native constructs.

Sync-Points
--------------------------------------------------------------------------------

A sync-point is a value which represents a command inside of a command-buffer
which is returned from command-buffer append function calls. These can be
optionally passed to these functions to define execution dependencies on other
commands within the command-buffer.

Sync-points are unique and valid for use only within the command-buffer they
were obtained from.

.. parsed-literal::
    // Append a memcpy with no sync-point dependencies
    ${x}_exp_command_buffer_sync_point_t syncPoint;

    ${x}CommandBufferAppendUSMMemcpyExp(hCommandBuffer, pDst, pSrc, size, 0, 
                                        nullptr, &syncPoint);
    
    // Append a kernel launch with syncPoint as a dependency, ignore returned
    // sync-point
    ${x}CommandBufferAppendKernelLaunchExp(hCommandBuffer, hKernel, workDim, 
                                           pGlobalWorkOffset, pGlobalWorkSize, 
                                           pLocalWorkSize, 1, &syncPoint, 
                                           nullptr);

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

API
--------------------------------------------------------------------------------

Macros
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${X}_COMMAND_BUFFER_EXTENSION_STRING_EXP

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_result_t
    * ${X}_RESULT_ERROR_INVALID_COMMAND_BUFFER_EXP
    * ${X}_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_EXP
    * ${X}_RESULT_ERROR_INVALID_COMMAND_BUFFER_SYNC_POINT_WAIT_LIST_EXP
* ${x}_structure_type_t
    * ${X}_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC
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



Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_exp_command_buffer_desc_t
* ${x}_exp_command_buffer_sync_point_t
* ${x}_exp_command_buffer_handle_t


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

Contributors
--------------------------------------------------------------------------------

* Ben Tracy `ben.tracy@codeplay.com <ben.tracy@codeplay.com>`_
* Ewan Crawford `ewan@codeplay.com <ewan@codeplay.com>`_
* Maxime France-Pillois `maxime.francepillois@codeplay.com <maxime.francepillois@codeplay.com>`_
