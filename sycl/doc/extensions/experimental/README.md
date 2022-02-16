This directory contains the specifications for SYCL extensions that are
considered experimental in the DPC++ implementation.  The APIs in these
extensions are not stable.  They may be changed or even removed in subsequent
releases of DPC++ without prior notice.  As a result, they are not recommended
for use in production code.

Experimental extensions may eventually be promoted to "supported".  When this
happens, a new specification is added to the "supported" directory, which may
not exactly match the experimental version.  (In particular, the namespace
containing the APIs is often changed from `sycl::ext::oneapi::experimental` to
`sycl::ext::oneapi`.)  The original experimental specification may be retained
for a time, or it may be removed.
