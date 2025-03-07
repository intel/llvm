<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>
====================
 Contribution Guide
====================

This contribution guide explains how to contribute to the Unified Runtime
project and what processes you *must* follow in order to have your changes
accepted into the project.

.. important::

    Before making a contribution to the specification you *should* determine if
    the change should be made directly to the core specification or introduced
    as an experimental feature. The criteria we use to make this distinction
    are as follows:

    *   The feature exists to enable an experimental feature in a parallel
        language runtime being built on top of Unified Runtime.

    *   The design phase of the feature is expected to span multiple oneAPI
        releases.

    *   A proof of concept implementation exists for a single adapter but
        multiple adapters are intended to be supported. It is important to
        consider as early as possible whether the feature is appropriate for
        other adapters to evaluate its portability.

    If the feature in question matches any of these criteria, please refer to
    the `Experimental Features`_ section, otherwise refer to the `Core
    Features`_ section. If you are unsure how to proceed please `create an
    issue <https://github.com/intel/llvm/issues/new?template=Blank+issue>`_
    asking or clarification.

    If you are unsure whether a feature can be supported by certain adapters
    please seek the advice of an appropriate stakeholder or ask the Unified
    Runtime team via the `GitHub issue tracker
    <https://github.com/intel/llvm/issues/new?template=Blank+issue>`_.

    When creating issues pertaining the to unified runtime, please include the
    [UR] tag in the issue title.

Build Environment
=================

To be able to generate the source from the YAML files, the build environment
must be configured correctly and all dependencies must be installed. The
instructions for a basic setup are available in the `README
<https://github.com/intel/llvm/blob/sycl/unified-runtime/README.md#building>`_.

The following additional dependencies are required to support the ``generate``
target:

*    Doxygen (>= 1.8)

*    The Python script requirements listed in `thirdparty/requirements.txt`_

Doxygen can be installed via your system's package manager, e.g. on Ubuntu
``sudo apt install doxygen``, or by downloading it from the Doxygen website. It
must be available on the current ``PATH`` when the script is run.

One way to install the requirements for the script is using a Python virtual
environment. This can be set up by running the following commands from the
project root:

.. code-block:: console

    $ python3 -m venv .local
    $ source .local/bin/activate
    $ pip install -r third_party/requirements.txt

The virtual environment can be subsequently reactivated before any builds
without needing to reinstall the requirements:

.. code-block:: console

    $ source .local/bin/activate

Alternatively, a Docker container can be used instead of a virtual environment.
Instructions on building and using a Docker image can be found in
`.github/docker`_

You *must* also enable the ``UR_FORMAT_CPP_STYLE`` CMake option to allow
formatting of the generated code, or the ``generate`` target will not be
available.

.. code-block:: console

    $ cmake -B build/ -DUR_FORMAT_CPP_STYLE=ON

You can then follow the instructions below to use the ``generate`` target to
regenerate the source.

.. _thirdparty/requirements.txt:
   https://github.com/intel/llvm/blob/sycl/unified-runtime/third_party/requirements.txt
.. _.github/docker:
   https://github.com/intel/llvm/blob/sycl/unified-runtime/.github/docker

Generating Source
=================

The specification and many other components in the Unified Runtime project
are generated from a set of YAML_ files which are used as inputs to a Mako_
based templating system. The YAML file syntax is defined in `YAML syntax`_. To
generate the outputs of the Mako templates a build directory must be
configured as detailed above. Upon successfully configuring a build directory,
generate the outputs with the following command (or suitable build system
equivalent):

.. code-block:: console

    $ cmake --build build --target generate

.. _YAML: https://yaml.org/
.. _Mako: https://www.makotemplates.org/
.. _YAML syntax:
   https://github.com/intel/llvm/blob/sycl/unified-runtime/scripts/YaML.md

.. note::

    The generated source and header files are placed into ``/source`` and 
    ``/include`` directories respectively. You *should* make no attempt to 
    modify them directly. When the generator is run all your changes will be 
    overwritten.

Writing YAML
============

Please read the :ref:`core/CONTRIB:Naming Convention` section prior to making a
contribution and refer to the `YAML syntax`_ for specifics of how to define the
required constructs.

When writing ``*.yml`` files and ``ur`` or ``UR`` should exist in the output
use ``$${'x'}`` or ``$${'X'}`` respectively. These will be replaced while
`Generating Source`_.

Additionally, the following conventions *must* be followed for function
arguments:

*   Argument names are ``camelCase``.
*   Arguments with pointer types are prefixed with ``p`` for each pointer in
    the type i.e. ``char *pMessage``, ``char **ppMessage``, etc.
*   Handle arguments are prefixed with ``h`` i.e. ``hQueue``.
*   Pointer to handle arguments, such as out parameters, are prefixed with
    ``ph`` i.e. ``phQueue``.

Limitations
-----------

There are some limitations on the patterns our spec generator can handle. These
limitations are due to convenience of implementation rather than design: if
they are preventing you from implementing a feature please open an issue and we
will be happy to try and accommodate your use case. Otherwise beware of the
following:

* A function parameter or struct member which is a struct type that has any of
  the following members in its type definition must not have the ``[range]``
  tag:

  * An object handle with the ``[range]`` tag

  * A struct type with the ``[range]`` tag that has an object handle member

* A struct member which is a pointer to a struct type must not have the
  ``[optional]`` tag if that struct (or any of its members, recursively) has
  an object handle member in its definition.

* A struct member which is an object handle must not have the ``[out]`` tag.

Forks and Pull Requests
=======================

To submit a pull request to Unified Runtime, you must first create your own
personal fork of the project and submit your changes to a branch. By convention
we name our branches ``<your_name>/<short_description>``, where the description
indicates the intent of your change. You can then raise a pull request
targeting ``intel/llvm:sycl``.

Please ensure you include the ``[UR]`` tag in the title of your pull request.

When making changes to the specification you *must* commit all changes to files
in the project as a result of `Generating Source`_.

Before your pull request is merged it *must* pass all jobs in the GitHub
Actions workflow and *must* be reviewed by all reviewer teams tagged as
code-owners.

.. hint::

    When rebasing a branch on top of ``main`` results in merged conflicts it is
    recommended to resolve conflicts in the ``*.yml`` files then `Generating
    Source`_. This will automatically resolve conflicts in the generated source
    files, leaving only conflicts in non-generated source files to be resolved,
    if any.

By default, any new fork has all GitHub Actions workflows disabled. If you would
like to, e.g., test your branch using our CI workflows *before* creating
a pull request, you have to enter the *Actions* tab on your fork and enable
workflows for this repository. When they are not needed anymore, you can disable
them again, but it has to be done one by one. The CI on the upstream repository
gets busy from time to time. That's why you may want to enable workflows on your
fork to get the testing results quicker. The disadvantage of the CI on your fork
is that it may report some failing jobs you may not expect, and it does not run
some of the jobs (due to a lack of specific hardware from self-hosted runners).

Core Features
=============

A core feature *must* have a stable API/ABI and *should* strive to be supported
across all adapters. However, core features *may* be optional and thus only
supported in one or more adapters. A core feature *should* also strive to
enable similar functionality in parallel language runtimes (such as SYCL,
OpenMP, ...) where possible although this is a secondary concern.

.. hint::

    Optional features should be avoided as much as possible to maximize
    portability across adapters and reduce the overhead required to make use of
    features in parallel language runtimes.

Core features are defined in the ``*.yml`` files in the `scripts/core
<https://github.com/intel/llvm/tree/sycl/unified-runtime/scripts/core>`_
directory. Most of the files are named after the API object who's interface is
defined within them, with the following exceptions:

*   `scripts/core/common.yml`_ defines symbols which are used by multiple
    interfaces through the specification, e.g. macros, object handles, result
    enumerations, and structure type enumerations.
*   `scripts/core/enqueue.yml`_ defines commands which can be enqueued on a
    queue object.
*   `scripts/core/loader.yml`_ defines global symbols pertaining to
    initialization and tear down of the loader.
*   `scripts/core/registry.yml`_ contains an enumeration of all entry-points,
    past and present, for use in the XPTI tracing framework. It is
    automatically updated so shouldn't require manual editing.
*   ``scripts/core/exp-<feature>.yml`` see `Experimental Features`_.

.. _scripts/core/common.yml:
   https://github.com/intel/llvm/blob/sycl/unified-runtime/scripts/core/common.yml
.. _scripts/core/enqueue.yml:
   https://github.com/intel/llvm/blob/sycl/unified-runtime/scripts/core/enqueue.yml
.. _scripts/core/loader.yml:
   https://github.com/intel/llvm/blob/sycl/unified-runtime/scripts/core/loader.yml
.. _scripts/core/registry.yml:
   https://github.com/intel/llvm/blob/sycl/unified-runtime/scripts/core/registry.yml

Core Optional Features
----------------------

Optional core features *must* be supported in at least one adapter. Support for
an optional core feature *must* be programmatically exposed to the user via
boolean query call to ${x}DeviceGetInfo and a new enumerator of the form
``UR_DEVICE_INFO_<FEATURE_NAME>_SUPPORT`` in ${x}_device_info_t.

Conformance Testing
-------------------

For contributions to the core specification conformance tests *should* be
included as part of your change. The conformance tests can be found
under ``test/conformance/<component>``, where component refers to the API
object an entry-point belongs to i.e. platform, enqueue, device.

The conformance tests *should* ideally include end-to-end testing of all the
changes to the specification if possible. At minimum, they *must* cover at
least one test for each of the possible error codes returned, excluding any
disaster cases like ${X}_RESULT_ERROR_OUT_OF_HOST_MEMORY or similar.

Conformance tests *must* not make assumptions about the adapter under test.
Tests fixtures or cases *must* query for support of optional features and skip
testing if unsupported by the adapter.

All tests in the Unified Runtime project are configured to use CTest to run. 
All conformance tests have the ``conformance`` label attached to them which 
allows them to be run independently. To run all the conformance tests, execute 
the following command from the build directory.

.. code-block:: console
     
    ctest -L "conformance"

Experimental Features
=====================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions
        over time.
    *   Do not require conformance testing of their own additions.

Experimental features *must* be defined in two new files, where
``<FEATURE>``/``<feature>`` are replaced with an appropriate name:

*   ``scripts/core/EXP-<FEATURE>.rst`` document specifying the experimental
    feature in natural language.
*   ``scripts/core/exp-<feature>.yml`` defines the interface as an input to
    `Generating Source`_.

To simplify this process please use the provided python script which will create
these template files for you. You can then freely modify these files to 
implement your experimental feature. 

.. code-block:: console

    $ python scripts/add_experimental_feature.py <name-of-your-experimental-feature>


Experimental features *must* not make any changes to the core YaML files and 
*must* be described entirely in their own YaML file. Sometimes, however 
experimental feature require extending enumerations of the core specification. 
If this is necessary, create a new enum with the ``extend`` field set to true 
and list the required enumerations to support the experimental feature. These 
additional enumerations will updated the specification with the appropriate 
values.


Naming Convention
=================

The following naming conventions must be followed:

## --validate=off
*   All functions must be prefixed with ``${x}``
*   All functions must use camel case ``${x}ObjectAction`` convention
*   All macros must use all caps ``${X}_NAME`` convention
*   All structures, enumerations and other types must follow ``${x}_name_t`` 
    snake case convention
*   All structure members and function parameters must use camel case 
    convention
*   All enumerator values must use all caps ``${X}_ENUM_ETOR_NAME`` 
    convention
*   All handle types must end with ``handle_t``
*   All descriptor structures must end with ``desc_t``
*   All property structures must end with ``properties_t``
*   All flag enumerations must end with ``flags_t``
## --validate=on

The following coding conventions must be followed:

*   All descriptor structures must be derived from ${x}_base_desc_t
*   All property structures must be derived from ${x}_base_properties_t
*   All function input parameters must precede output parameters
*   All functions must return ${x}_result_t

In addition to the requirements referred to in the `Writing YAML`_ section, and
to easily differentiate experimental feature symbols, the following conventions
*must* be adhered to when defining experimental features:

## --validate=off
*   All functions must use camel case ``${x}ObjectActionExp`` convention.
*   All macros must use all caps ``${X}_NAME_EXP`` convention.
*   All structures, enumerations, and other types must follow
    ``${x}_exp_name_t`` name case convention.
## --validate=on
