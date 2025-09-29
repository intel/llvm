
<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>
.. _experimental-usm-import-release:

==================
USM Import/Release
==================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Data transfer between Host and Device is most efficient when source and
destination are both allocated in USM memory.
In situations where host data will participate in host/device transfers
and the host data allocation is under user control, USM functions
such as malloc_host could be used to allocate USM memory instead of
system memory.
However, this is not always possible if the source code where the allocation
is made is not available, or source code changes are prohibited for portability
reasons.
In these situations a mechanism to temporarily promote system memory to USM
for the duration of the host/device data transfers is useful for maximizing
data transfer rate.


Import Host Memory into USM
===========================

Import a range of host memory into USM.

.. parsed-literal::

    // Import into USM
    ${x}USMImportExp(hContext, hostPtr, size);

Release Host Memory Previously Imported into USM
================================================

Release from USM a range of memory that had been previously imported
into USM.


.. parsed-literal::

    // Release from USM
    ${x}USMReleaseExp(hContext, hostPtr);

