=================================
SYCL_EXT_ONEAPI_NUM_COMPUTE_UNITS
=================================

.. contents::
   :local:


Contact
=======

To report problems with this extension, please open a new issue at:

https://github.com/intel/llvm/issues


Dependencies
============

This extension is written against the SYCL 2020 revision 9 specification.
All references below to the "core SYCL specification" or to section numbers in
the SYCL specification refer to that revision.


Status
======

This is a proposed extension specification, intended to gather community
feedback.
Interfaces defined in this specification may not be implemented yet or may be in
a preliminary state.
The specification itself may also change in incompatible ways before it is
finalized.
**Shipping software products should not rely on APIs defined in this
specification.**


Overview
========

SYCL 2020 allows developers to query the maximum number of compute units in a
device via the ``info::device::max_compute_units`` query.
There are two issues with this existing query: first, that it refers to a
"maximum", despite the number of compute units being a fixed property of a
device; and second, that the definition of "compute units" is vague.
Different implementations and backends do not provide consistent interpretations
of this query, which makes it difficult for developers to use the number of
compute units in a portable way.

This extension provides a new query, ``info::device::num_compute_units``, with
the aim to clarify the meaning of "compute units" in SYCL and drive consistency
across implementations.


Specification
=============

Feature test macro
------------------

This extension provides a feature-test macro as described in the core SYCL
specification.
An implementation supporting this extension must predefine the macro
``SYCL_EXT_ONEAPI_NUM_COMPUTE_UNITS`` to one of the values defined in the table
below.
Applications can test for the existence of this macro to determine if the
implementation supports this feature, or applications can test the macro's value
to determine which of the extension's features the implementation supports.

.. table::
   :align: left

   =====  ===========
   Value  Description
   =====  ===========
   1      Initial version of this extension.
   =====  ===========

Compute units
-------------

A SYCL device is divided into one or more compute units, which are each divided
into one or more processing elements.

All work-items in a given work-group must execute on exactly one compute unit.
The mapping of work-groups to compute units is not guaranteed: work-groups may
be dispatched to compute units in any order, and this order may be different
for every kernel launch.

An implementation may execute multiple work-groups on a single compute unit
simultaneously, subject to the resource constraints described by other device
and kernel queries.

The representation of specific hardware architectures in terms of compute units
is tied to the execution model exposed by an implementation and is thus
implementation-defined.

[*Note:* To improve the portability of SYCL programs, implementations are
encouraged to define compute units such that it is possible to fully utilize the
hardware resources of a device by launching one work-group of size
``max_work_group_size`` on each compute unit.
*--end note*]

Device queries
--------------

This extension adds the following device information descriptor.

----

info::device::num_compute_units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   namespace sycl::ext::oneapi::info::device {
   struct num_compute_units {
     using return_type = size_t;
   };
   }

*Remarks:* Template parameter to ``device::get_info``.

*Returns:* The number of compute units in the device.
The minimum value is 1.

[*Note:* The value is not required to be equal to the value returned by
``max_compute_units``.
*--end note*]

----


Implementation in DPC++
=======================

This section is non-normative and applies only to the DPC++ implementation.

The table below explains how DPC++ calculates the number of compute units for
different combinations of device and backend.

.. table::
   :align: left

   ===========  ==========  =================
   Device Type  Backend(s)  Number of Domains
   ===========  ==========  =================
   CPU          OpenCL      Number of logical cores.
   Intel GPU    Any         Number of Xe cores.
   NVIDIA GPU   Any         Number of streaming multiprocessors (SMs).
   ===========  ==========  =================
