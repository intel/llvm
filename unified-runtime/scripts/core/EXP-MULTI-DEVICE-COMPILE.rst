<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-multi-device-compile:

================================================================================
Multi Device Compile
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.



Motivation
--------------------------------------------------------------------------------

Instead of relying on the list of devices used to create a context, provide
interfaces which instead take a list of devices. This more closely aligns with
PI and OpenCL. Introduced to workaround a regression. May be superseded in
future.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_MULTI_DEVICE_COMPILE_SUPPORT_EXP

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}ProgramBuildExp
* ${x}ProgramCompileExp
* ${x}ProgramLinkExp

Changelog
--------------------------------------------------------------------------------

+-----------+---------------------------------------------+
| Revision  | Changes                                     |
+===========+=============================================+
| 1.0       | Initial Draft                               |
+-----------+---------------------------------------------+
| 1.1       | Switch from extension string macro to       |
|           | device info enum for reporting support.     |
+-----------+---------------------------------------------+

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return ``true`` when
queried for ${X}_DEVICE_INFO_MULTI_DEVICE_COMPILE_SUPPORT_EXP via
${x}DeviceGetInfo. Conversely, before using any of the functionality defined
in this experimental feature the user *must* use the device query to determine
if the adapter supports this feature.

Contributors
--------------------------------------------------------------------------------

* Kenneth Benzie (Benie) `k.benzie@codeplay.com <k.benzie@codeplay.com>`_
* Aaron Greig `aaron.greig@codeplay.com <aaron.greig@codeplay.com>`_
