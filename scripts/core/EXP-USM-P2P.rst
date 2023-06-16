<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-usm-p2p:

================================================================================
Usm P2P
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

================================================================================
API
================================================================================

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_exp_peer_info_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}EnablePeerAccessExp
* ${x}DisablePeerAccessExp
* ${x}PeerAccessGetInfoExp

Changelog
--------------------------------------------------------------------------------
.. comment:
    When making a change to an experimental feature, increment the version and 
    provide a brief description of the change in the table below.

+-----------+------------------------+
| Revision  | Changes                |
+===========+========================+
| 1.0       | Initial Draft           |
+-----------+------------------------+

Contributors
--------------------------------------------------------------------------------
.. comment:
    Please list all people who wish to be credited for contribution to this 
    experimental feature.

* JackAKirk `jack.kirk@codeplay.com <jack.kirk@codeplay.com>`_
