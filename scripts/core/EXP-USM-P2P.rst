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
Several important projects that the SYCL programming model aims to support use
fine-grained peer to peer memory access controls.
Two such examples that SYCL supports are Pytorch and Gromacs.
This experimental extension to UR aims to provide a portable interface that can
call appropriate driver functions to query and control peer memory access
across the CUDA, HIP and L0 adapters.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_exp_peer_info_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}UsmP2PEnablePeerAccessExp
* ${x}UsmP2PDisablePeerAccessExp
* ${x}UsmP2PPeerAccessGetInfoExp

Changelog
--------------------------------------------------------------------------------

+-----------+------------------------+
| Revision  | Changes                |
+===========+========================+
| 1.0       | Initial Draft          |
+-----------+------------------------+

Contributors
--------------------------------------------------------------------------------

* JackAKirk `jack.kirk@codeplay.com <jack.kirk@codeplay.com>`_
