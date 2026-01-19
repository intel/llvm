<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-dynamic-link:

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

Some adapters support the ability to do dynamic linking between programs,
resolving external symbols through that. This may allow AOT compiled binaries to
be linked, despite already having been built.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_DYNAMIC_LINK_SUPPORT_EXP

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}ProgramDynamicLinkExp

Changelog
--------------------------------------------------------------------------------

+-----------+---------------------------------------------+
| Revision  | Changes                                     |
+===========+=============================================+
| 1.0       | Initial Draft                               |
+-----------+---------------------------------------------+

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return ``true`` when
queried for ${X}_DEVICE_INFO_DYNAMIC_LINK_SUPPORT_EXP via
${x}DeviceGetInfo. Conversely, before using any of the functionality defined
in this experimental feature the user *must* use the device query to determine
if the adapter supports this feature.

Contributors
--------------------------------------------------------------------------------

* Steffen Larsen `steffen.larsen@intel.com <steffen.larsen@intel.com>`_
