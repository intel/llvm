===========================
SYCL_EXT_ONEAPI_MYEXTENSION
===========================

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

*If your extension depends on other SYCL extensions, add a paragraph and
bulleted list like this:*

This extension also depends on the following other SYCL extensions:

* :doc:`supported/sycl_ext_oneapi_free_function_queries`

*However, if the dependent extension is not yet written in reStructuredText,
use a link like this instead:*

* `SYCL_EXT_ONEAPI_PROPERTIES`_

.. _`SYCL_EXT_ONEAPI_PROPERTIES`: https://github.com/intel/llvm/tree/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_properties.asciidoc


Status
======

*Choose one of the following according to the status of your extension.
For a "proposed" extension:*

This is a proposed extension specification, intended to gather community
feedback.
Interfaces defined in this specification may not be implemented yet or may be in
a preliminary state.
The specification itself may also change in incompatible ways before it is
finalized.
**Shipping software products should not rely on APIs defined in this
specification.**

*Use this if the extension becomes "supported":*

This extension is implemented and fully supported by DPC++.

*Use this if the extension becomes "experimental":*

This is an experimental extension specification, intended to provide early
access to features and gather community feedback.
Interfaces defined in this specification are implemented in DPC++, but they are
not finalized and may change incompatibly in future versions of DPC++ without
prior notice.
**Shipping software products should not rely on APIs defined in this
specification.**

*Use this if the extension becomes "deprecated":*

This extension has been deprecated.
Although it is still supported in DPC++, we expect that the interfaces defined
in this specification will be removed in an upcoming DPC++ release.
**Shipping software products should stop using APIs defined in this
specification and use an alternative instead.**

*Use this if the extension becomes "removed":*

This extension is no longer implemented in DPC++.
This specification is being archived only for historical purposes.
**The APIs defined in this specification no longer exist and cannot be used.**


Backend support status
======================

*Sometimes an extension can be used only on certain backends.
Your extension document should include this section when this is the case.*

*Our preference is to tie the extension to a device aspect when it is
available only on certain backends because it results in a clear error
if the application mistakenly tries to use it on an unsupported backend.
When this is the case, include a paragraph like this:*

The APIs in this extension may be used only on a device that has
``aspect::ext_oneapi_foo``.
The application must check that the device has this aspect before submitting a
kernel using any of the APIs in this extension.
If the application fails to do this, the implementation throws a synchronous
exception with the ``errc::kernel_not_supported`` error code when the kernel is
submitted to the queue.

*Occasionally, an extension is limited to certain backends and there is no
related device aspect. When this is the case, include a paragraph like:*

This extension is currently implemented in DPC++ only for GPU devices and
only when using the Level Zero backend.
Attempting to use this extension in kernels that run on other devices or
backends may result in undefined behavior.
Be aware that the compiler is not able to issue a diagnostic to warn you if this
happens.


Overview
========

*Provide a brief overview of the extension here and explain the motivation if
appropriate.
This is also a good place to show an example usage, but there is no need to
exhaustively show all aspects of your extension.
Those details should be explained in the sections that follow.
This section is just an overview to introduce your readers to your extension.*

*It is also appropriate to give an indication of who the target audience is for
the extension.
For example, if the extension is intended only for ninja programmers, we might
say something like:*

The properties described in this extension are advanced features that most
applications should not need to use.
In most cases, applications get the best performance without using these
properties.

*Occasionally, we might add an extension as a stopgap measure for a limited
audience.
When this happens, it's best to discourage general usage with a statement like:*

This extension exists to solve a specific problem, and a general solution is
still being evaluated.
It is not recommended for general usage.


Specification
=============

Feature test macro
------------------

*All extensions should provide a feature-test macro, so that applications
can use* ``#ifdef`` *to protect code that uses your extension.
Use this text for all extensions:*

This extension provides a feature-test macro as described in the core SYCL
specification.
An implementation supporting this extension must predefine the macro
``SYCL_EXT_ONEAPI_MYEXTENSION`` to one of the values defined in the table below.
Applications can test for the existence of this macro to determine if the
implementation supports this feature, or applications can test the macro's value
to determine which of the extension's features the implementation supports.

*And follow the text with a table like this* **unless the extension is
"experimental"**.
*Note that your table may have more than one row if it has multiple versions.*

.. table::
   :align: left

   =====  ===========
   Value  Description
   =====  ===========
   1      Initial version of this extension.
   =====  ===========

*If your extension is "experimental", use this table instead:*

.. table::
   :align: left

   =====  ===========
   Value  Description
   =====  ===========
   1      The APIs of this experimental extension are not versioned, so the
          feature-test macro always has this value.
   =====  ===========

Guidelines for writing specifications
-------------------------------------

*Your extension specification will certainly have more sections which describe
the APIs of your extension.
Define these sections as you see fit, but observe the guidelines in the
following subsections.*

*After these guidelines there are a number of sections which demonstrate the
recommended format to use for various scenarios that commonly occur with SYCL
extensions.*

Line breaks
^^^^^^^^^^^

Break lines after 80 columns or at the end of a sentence, whichever comes first
(like this template).
We have found that this works well with "git diff" and other tooling.

Ensure good HTML rendering
^^^^^^^^^^^^^^^^^^^^^^^^^^

Specifications should be written in reStructuredText, and we use Sphinx tools to
generate HTML pages from the reStructuredText source code.
However, we also expect many users to read the specifications directly from
GitHub.
If you open a file with the ``.rst`` suffix from GitHub in a browser, the GitHub
server displays an HTML rendering of the file.
Therefore, there are two ways these specifications can be rendered into HTML:
via the Sphinx tools and via the GitHub server.
We expect these specifications to be nicely formatted in both cases.

When writing a specification, make sure your document is rendered nicely in both
cases.
You can generate the HTML locally via Sphinx with the following commands:

.. code-block:: bash

   $ cd build
   $ cmake --build . --target docs-sycl-html

And the HTML will be generated in ``build/tools/sycl/doc/html/extensions``.

To check the HTML rendering via GitHub, push your changes to a branch in
GitHub, and then use a browser to navigate to your specification file(s).

Describe your extension not how to change the SYCL specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Do not describe **how** you will change the core SYCL specification.
For example, there is no need to say something like "Add the following paragraph
to section 4.6.7 of the core SYCL specification ..."
Instead, just describe the semantics and APIs of the extension itself.
We will figure out later how to change the core SYCL specification if the
extension is adopted.

Avoid unnecessary duplication of the SYCL specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Avoid unnecessary duplication of APIs or text from the core SYCL specification.
For example, if your extension adds a new member function to an existing class,
there is no need to duplicate the existing member functions in your
specification.
Duplication like this can be troublesome if something in the core SYCL
specification changes later.

Use ISO C++ "words of power"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When specifying a function, use paragraphs with the following titles:

* *Constraints:* This paragraph can be used whenever the function is templated
  or when the function is a member of a templated class.
  The constraints describe the SFINAE constraints or ``requires`` constraints
  of the template parameters.

* *Preconditions:* This paragraph tell the conditions that the application must
  obey when calling the function.
  If the application violates a precondition, the behavior is undefined.

* *Effects:* Tells what the function does.

* *Returns:* Tells what value the function returns.

* *Throws:* Tells what exceptions the function is required to throw and the
  circumstances under which they are thrown.

* *Remarks:* Tells other information about the function if it is not covered by
  the previous paragraphs.

All these terms are also used in the ISO C++ specification, and we use them in
the SYCL specification with the same meaning they have in C++.

Often the *Constraints*, *Preconditions*, or *Throws* paragraphs will have
several items.
When this happens, use a bulleted list.

Only include a paragraphs if its term is needed to specify the function.
Simply omit paragraphs whose terms are unnecessary.

Normative vs. non-normative text
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A specification is usually a mixture of normative and non-normative text.
Normative text is the formal description of the extension, while non-normative
text is intended to provide general context and clarity.
A good specification should describe all of its features using normative text.
In other words, it should be possible for someone to implement your extension
just by reading the normative text.

The main section titled "Specification" (and its subsections) are the normative
part of the specification document.
All other sections (e.g. "Overview", "Examples", etc.) are non-normative.
In addition, examples are non-normative even if they appears inside the
"Specification" section.
This applies to example code and also to sentences that describe an example.
(Such sentences often start with "For example, ...".)
Note that the code synopses are not examples, and therefore they are normative.

As a result, make sure your specification does not rely on the Overview section
or on the examples to fully specify your extension.
These should add clarity to the specification.
Do not rely on them for the specification itself.

Sometimes, it is useful to make a clarification that is non-normative.
In this case, format the text as a "note" like this:

[*Note:* This is non-normative text.
*--end note*]

A "note" like this can contain several sentences or even several paragraphs if
necessary.

Use of summary synopses
^^^^^^^^^^^^^^^^^^^^^^^

When an extension adds or augments a class or enumeration, the specification
shows a summary synopsis of the class or enumeration that just shows the
declarations of the new members or new enumerators.
After that, each member or enumerator is shown again along with its
specification.
We recognize that this introduces some duplication in the specification, but
we feel that the summary synopsis adds enough value that it is worth the
duplication.

However, we do not add a summary synopsis for non-member functions that the
extension adds.
Instead, we just have a single synopsis with the function's declaration, which
is followed by the function's specification.

The example sections below illustrate these styles.

Group related functions together
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When two or more functions are closely related, they can be grouped together
in a single synopsis which specifies all functions in the group.
When doing this, paragraphs like *Effects* and *Returns* apply to all functions
in the group.
Sometimes it is helpful to call out a specific function in the grup.
To do this, add numeric callouts to the synopsis, and then use these numbers in
the specification text.

See the example sections below for the exact style to use in this case.

Use of horizontal bars
^^^^^^^^^^^^^^^^^^^^^^

When more than one function or enumerator is specified in the same section, we
use horizontal bars to separate them.
The style guidelines are:

* If the section starts with some introductory text before the first function
  or enumerator specification, put a horizontal bar after that introductory text
  and before the first specification.

* Put a horizontal bar between each pair of function / enumerator
  specifications.

* Put a horizontal bar after the last function / enumerator specification.
  However, do not put a horizontal bar after a specification if there is no
  horizontal bar before the specification (which can happen if the document
  section has only one such specification and there is no introductory text).

Add new enumerators to an enumeration
-------------------------------------

*Recommended format for an extension that adds new enumerators to an existing
SYCL enumeration.*

This extension adds the following new enumerators to the ``aspect`` enumeration:

.. code-block:: c++

   namespace sycl {

   enum class aspect {
     ext_oneapi_foo
     ext_oneapi_bar

     // ...
   };

   }

----

ext_oneapi_foo
^^^^^^^^^^^^^^

Indicates that the device allows foo operations.

----

ext_oneapi_bar
^^^^^^^^^^^^^^

Indicates that the device allows bar operations.

----

Add new member functions to a class
-----------------------------------

*Recommended format for an extension that adds new member functions to an
existing SYCL class.*

This extension adds the following new member functions to the ``queue`` class:

.. code-block:: c++

   namespace sycl {

   class queue {
     int ext_oneapi_foo();

     template<typename T>
     T ext_oneapi_bar(T val);

     // ...
   };

   }

----

ext_oneapi_foo
^^^^^^^^^^^^^^

.. code-block:: c++

   int ext_oneapi_foo();

*Effects:* Does the foo thing to the queue.

*Returns:* The number of foo things that have been done to the queue so far.

----

ext_oneapi_bar
^^^^^^^^^^^^^^

.. code-block:: c++

   template<typename T>
   T ext_oneapi_bar(T val);

*Constraints:* ``T`` is an integral type.

*Preconditions:* The value ``val`` is not zero.

*Effects:* Adds ``val`` "bar" counters to the queue.

*Returns:* The previous number of "bar" counters in the queue.

*Throws:*

* A synchronous ``exception`` with the ``errc::invalid`` error code if the queue
  is in order.
* A synchronous ``exception`` with the ``errc::feature_not_supported`` error
  code if the queue's device does not have ``aspect::ext_oneapi_bar``.

----

Add a new class
---------------

*Recommended format for an extension that adds a new class.*

This extension adds the following class which can zap things.

.. code-block:: c++

   namespace sycl::ext::oneapi {

   class zapper {
    public:
     zapper(const device& dev);
     zapper(const std::vector<device>& devs);

     void send();
     void wait();
   }

   }

Constructors and copy assignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   zapper(const device& dev);                (1)
   zapper(const std::vector<device>& devs);  (2)

*Preconditions (2):* ``devs`` is not empty.

*Effects:* Constructs a zapper object that can send zap signals to a single
device or a set of devices.

Member functions and operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

send
""""

.. code-block:: c++

   void send();

*Effects:* Sends a zap signal to the device(s) in the zapper object.

----

wait
""""

.. code-block:: c++

   void wait();

*Effects:* Wait for all devices to be zapped.

----


Add new free functions
----------------------

*Recommended format for an extension that adds new non-member functions.*

----

throb
^^^^^

.. code-block:: c++

   namespace sycl::ext::oneapi {

   void throb(const device &dev);

   }

*Effects:* Tells the device(s) to throb.

----

thrum
^^^^^

.. code-block:: c++

   namespace sycl::ext::oneapi {

   void thrum(const device &dev);

   }

*Effects:* Tells the device to thrum.

----

Add new information descriptors
-------------------------------

*Recommended format for an extension that adds new information descriptors.*

This extension adds the following new device information descriptors.

----

info::device::num_foos
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   namespace sycl::ext::oneapi::info::device {
   struct num_foos {
     using return_type = int;
   };
   }

*Remarks:* Template parameter to ``device::get_info``.

*Returns:* The number of foo things in the device.

----

info::device::num_bars
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

   namespace sycl::ext::oneapi::info::device {
   struct num_bars {
     using return_type = size_t;
   };
   }

*Remarks:* Template parameter to ``device::get_info``.

*Returns:* The number of bar things in the device.

----


Examples
========

*It is often useful to include an Examples section in your document that shows
how the extension is typically used.
Some guidelines to follow for good examples:*

* Prefer examples that are complete programs if at all possible.
  This includes the definition of the ``main`` function and the
  ``#include <sycl/sycl.hpp>``.

* Do not use ``using namespace``.
  Instead, use fully qualified names, so it is clear which namespace they are
  contained within.
  In order to avoid verbosity, define a namespace alias at the top of the
  example, such as ``namespace syclex = sycl::ext::oneapi``, and then use that
  namespace alias in the rest of the example.


Implementation notes
====================

This non-normative section provides information about one possible
implementation of this extension.
It is not part of the specification of the extension's API.

*This section is not normally needed, but occasionally a "proposed" extension
will contain some notes about the intended implementation.
If so, add this section, and include the text in the first paragraph above
indicating that the section is non-normative.
Follow that paragraph with whatever implementation notes you think are
necessary.
Usually, this section will be removed by the time the extension is implemented,
and a more detailed DPC++ design document will be written instead.*


Issues
======

*Sometimes there will be unresolved issues in a "proposed" extension.
If this is the case, add an "Issues" section towards the end of the document,
and list each issue.*

1. This is the first open issue.

2. This is the second open issue.
