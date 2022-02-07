# Supported Extensions

This directory contains the specifications for SYCL extensions that are fully
supported by the DPC++ implementation.  The APIs in these extensions are
generally stable in future releases of DPC++, retaining backward compatibility
with application code.

If support is dropped for one of these extensions, it goes through a
deprecation process.  The APIs in the extension are first marked "deprecated",
so that the compiler issues a warning when they are used, but the extension
remains supported during this time.  Once the deprecation period elapses, the
support for the extension may be dropped, and the extension specification
document is moved to the "../removed" directory.
