<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-usm-p2p:

================================================================================
USM P2P
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Terminology
--------------------------------------------------------------------------------
"Peer to peer" ("P2P") is used to indicate a "command" device either accessing
or copying the memory located on a separate "peer" device.

Motivation
--------------------------------------------------------------------------------
Programming models like SYCL or OpenMP aim to support several important
projects that utilise fine-grained peer-to-peer memory access controls.
This experimental extension to the Unified-Runtime API aims to provide a
portable interface that can call appropriate driver functions to query and
control peer memory access within different adapters such as CUDA, HIP and
Level Zero.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_USM_P2P_SUPPORT_EXP

* ${x}_exp_peer_info_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}UsmP2PEnablePeerAccessExp
* ${x}UsmP2PDisablePeerAccessExp
* ${x}UsmP2PPeerAccessGetInfoExp

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return ``true`` when
queried for ${X}_DEVICE_INFO_USM_P2P_SUPPORT_EXP via ${x}DeviceGetInfo.
Conversely, before using any of the functionality defined in this experimental
feature the user *must* use the device query to determine if the adapter
supports this feature.

Changelog
--------------------------------------------------------------------------------

+-----------+---------------------------------------------+
| Revision  | Changes                                     |
+===========+=============================================+
| 1.0       | Initial Draft                               |
+-----------+---------------------------------------------+
| 1.1       | Added USM_P2P_EXTENSION_STRING_EXP ID Macro |
+-----------+---------------------------------------------+
| 1.2       | Switch Info types from uint32_t to int      |
+-----------+---------------------------------------------+
| 1.3       | Switch from extension string macro to       |
|           | device info enum for reporting support.     |
+-----------+---------------------------------------------+

Contributors
--------------------------------------------------------------------------------

* JackAKirk `jack.kirk@codeplay.com <jack.kirk@codeplay.com>`_
* Aaron Greig `aaron.greig@codeplay.com <aaron.greig@codeplay.com>`_
