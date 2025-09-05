<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-enqueue-native-command:

================================================================================
Enqueue Native Command
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------
Interop is an important use case for many programming APIs. Through
${x}EnqueueNativeCommandExp the user can immediately invoke some native API
calls in a way that the UR is aware of. In doing so, the UR adapter can
integrate its own scheduling of UR commands with native commands.

In order for UR to guarantee correct synchronization of commands enqueued
within the native API through the function passed to
${x}EnqueueNativeCommandExp, the function argument must only use the native
queue accessed through ${x}QueueGetNativeHandle. Use of a native queue that is
not the native queue returned by ${x}QueueGetNativeHandle results in undefined
behavior.

Any args that are needed by the func must be passed through a ``void*`` and unpacked
within the func. If ``${x}_mem_handle_t`` arguments are to be used within
``pfnNativeEnqueue``, they must be accessed using ${x}MemGetNativeHandle.
``${x}_mem_handle_t`` arguments must be packed in the void* argument that will be
used in ``pfnNativeEnqueue``, as well as ${x}EnqueueNativeCommandExp's ``phMemList``
argument.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_ENQUEUE_NATIVE_COMMAND_SUPPORT_EXP
* ${x}_command_t
    * ${X}_COMMAND_ENQUEUE_NATIVE_EXP
* ${x}_exp_enqueue_native_command_flags_t

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

${x}_exp_enqueue_native_command_properties_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}EnqueueNativeCommandExp

Changelog
--------------------------------------------------------------------------------

+-----------+---------------------------+
| Revision  | Changes                   |
+===========+===========================+
| 1.0       | Initial Draft             |
+-----------+---------------------------+
| 1.1       | Make ``phEvent`` optional |
+-----------+---------------------------+


Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return true for the new
``${X}_DEVICE_INFO_ENQUEUE_NATIVE_COMMAND_SUPPORT_EXP`` device info query.


Contributors
--------------------------------------------------------------------------------

* Hugh Delaney `hugh.delaney@codeplay.com <hugh.delaney@codeplay.com>`_
* Kenneth Benzie (Benie) `k.benzie@codeplay.com <k.benzie@codeplay.com>`_
* Ewan Crawford `ewan@codeplay.com <ewan@codeplay.com>`_
