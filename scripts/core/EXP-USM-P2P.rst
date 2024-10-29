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

Macros
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${X}_USM_P2P_EXTENSION_STRING_EXP

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_exp_peer_info_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}UsmP2PEnablePeerAccessExp
* ${x}UsmP2PDisablePeerAccessExp
* ${x}UsmP2PPeerAccessGetInfoExp

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return the valid string
defined in ``${X}_USM_P2P_EXTENSION_STRING_EXP`` as one of the options from
${x}DeviceGetInfo when querying for ${X}_DEVICE_INFO_EXTENSIONS.

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

Contributors
--------------------------------------------------------------------------------

* JackAKirk `jack.kirk@codeplay.com <jack.kirk@codeplay.com>`_
