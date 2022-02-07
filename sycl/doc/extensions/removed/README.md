This directory contains an archive of old DPC++ extensions which are no longer
implemented.

Normally, a supported extension is first marked "deprecated", and the compiler
raises a warning for applications that use it.  After the deprecation period,
support for the extension may be removed, and the specification for the
extension is moved to this directory for reference.

Experimental extensions may change or be removed without any deprecation
period.  Since we do not expect production code to use experimental extensions,
we do not archive their specifications when they are changed or removed.
Likewise, we do not archive "proposed" extension specifications if we later
decide not to implement them.

Note that the following extension specifications have been removed because
their features have been incorporated into the core
[SYCL 2020 specification][1].  Please see that document for the most accurate
description of these features.

[1]: <https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html>

| Extension                                     | Description                                                 |
|-----------------------------------------------|-------------------------------------------------------------|
|SYCL\_INTEL\_bitcast                           | Adds `sycl::bit_cast`                                       |
|SYCL\_INTEL\_device\_specific\_kernel\_queries | Adds `info::kernel_device_specific` queries                 |
|SYCL\_INTEL\_attribute\_style                  | Changes position of C++ attributes                          |
|Queue Order Properties                         | Adds `property::queue::in_order`                            |
|SYCL\_INTEL\_parallel\_for\_simplification     | Makes calls to `parallel_for` less verbose                  |
|Queue Shortcuts                                | Adds shortcut functions to `queue`                          |
|SYCL\_INTEL\_relax\_standard\_layout           | Drops standard layout requirement for data in buffers, etc. |
|Unified Shared Memory                          | Adds new unified shared memory APIs                         |
|SYCL\_INTEL\_unnamed\_kernel\_lambda           | Makes kernel type-names optional                            |
|SYCL\_INTEL\_deduction\_guides                 | Simplifies SYCL object construction by using C++ CTAD       |
|SYCL\_INTEL\_math\_array                       | Adds `sycl::marray`                                         |
