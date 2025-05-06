=====================================
SYCL_EXT_ONEAPI_FREE_FUNCTION_QUERIES
=====================================

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

This extension is implemented and fully supported by DPC++.


Overview
========

The extension allows developers to access ``sycl::nd_item``, ``sycl::group`` and
``sycl::sub_group`` instances globally, without having to explicitly pass them
as arguments to each function used on the device.

[*Note:* Passing such instances as arguments can result in a clearer interface
that is less error-prone to use.
For example, if a function accepts a ``sycl::group``, the caller must assume
that function may call a ``sycl::group_barrier`` and ensure that associated
control flow requirements are satisfied.
It is recommended that this extension is used only when modifying existing
interfaces is not feasible.
*--end note*]


Specification
=============

Feature test macro
------------------

This extension provides a feature-test macro as described in the core SYCL
specification.
An implementation supporting this extension must predefine the macro
``SYCL_EXT_ONEAPI_FREE_FUNCTION_QUERIES`` to one of the values defined in the
table below.
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

Accessing instances of special SYCL classes
-------------------------------------------

The ``sycl::ext::oneapi::this_work_item`` namespace contains functionality
related to the currently executing kernel.

It is the user's responsibility to ensure that these functions are called
in a manner that is compatible with the kernel's launch parameters, as detailed
in the definition of each function.
Calling these functions from an incompatible kernel results in undefined
behavior.

----

this_work_item::get_nd_item
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

     namespace sycl::ext::oneapi::this_work_item {

     template <int Dimensions>
     nd_item<Dimensions> get_nd_item();

     }

*Preconditions:* ``Dimensions`` must match the dimensionality of the currently
executing kernel.
The currently executing kernel must have been launched with a ``sycl::nd_range``
argument.

*Returns:* A ``sycl::nd_item`` instance representing the current work-item in a
``sycl::nd_range``.

----

this_work_item::get_work_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

     namespace sycl::ext::oneapi::this_work_item {

     template <int Dimensions>
     group<Dimensions> get_work_group();

     }

*Preconditions:* ``Dimensions`` must match the dimensionality of the currently
executing kernel.
The currently executing kernel must have been launched with a ``sycl::nd_range``
argument.

*Returns:* A ``sycl::group`` instance representing the work-group to which the
current work-item belongs.

----

this_work_item::get_sub_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

     namespace sycl::ext::oneapi::this_work_item {

     sub_group get_sub_group();

     }

*Preconditions:* The currently executing kernel must have been launched with a
``sycl::nd_range`` argument.

*Returns:* A ``sycl::sub_group`` instance representing the sub-group to which
the current work-item belongs.

----

Deprecated functionality
------------------------

The functionality in this section was previously part of an experimental
version of this extension, but is now deprecated.

----

experimental::this_id
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

     namespace sycl::ext::oneapi::experimental {

     template <int Dimensions>
     id<Dimensions> this_id();

     }

*Preconditions:* ``Dimensions`` must match the dimensionality of the currently
executing kernel.
The currently executing kernel must have been launched with a ``sycl::range`` or
``sycl::nd_range`` argument.

*Returns:* A ``sycl::id`` instance representing the current work-item in the
global iteration space.

----

experimental::this_item
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

     namespace sycl::ext::oneapi::experimental {

     template <int Dimensions>
     id<Dimensions> this_id();

     }

*Preconditions:* ``Dimensions`` must match the dimensionality of the currently
executing kernel.
The currently executing kernel must have been launched with a ``sycl::range`` or
``sycl::nd_range`` argument.

*Returns:* A ``sycl::item`` instance representing the current work-item in the
global iteration space.

[*Note:* The ``offset`` parameter to ``parallel_for`` is deprecated in SYCL
2020, as is the ability of an ``item`` to carry an offset.
This extension returns an ``item`` where the ``WithOffset`` template parameter
is set to ``false`` to prevent usage of the new queries in conjunction with
deprecated functionality.
*--end note*]

----

experimental::this_nd_item
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

     namespace sycl::ext::oneapi::experimental {

     template <int Dimensions>
     nd_item<Dimensions> this_nd_item();

     }

*Effects:* Equivalent to ``return this_work_item::get_nd_item()``.

----

experimental::this_group
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

     namespace sycl::ext::oneapi::experimental {

     template <int Dimensions>
     group<Dimensions> this_group();

     }

*Effects:* Equivalent to ``return this_work_item::get_work_group()``.

----

experimental::this_sub_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

     namespace sycl::ext::oneapi::experimental {

     sub_group this_sub_group();

     }

*Effects:* Equivalent to ``return this_work_item::get_sub_group()``.

----


Issues
======

1. Can undefined behavior be avoided or detected?

**UNRESOLVED**: Good run-time errors would likely require support for
device-side assertions or exceptions, while good compile-time errors would
likely require some additional compiler modifications and/or kernel properties.
