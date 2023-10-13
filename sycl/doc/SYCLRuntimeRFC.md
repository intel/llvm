scourse topic details

- Category: [Runtimes](https://discourse.llvm.org/c/runtimes/64)
- Title: "RFC: SYCL runtime upstreaming"
- Tags: sycl

## Overview

Our SYCL runtime provides an implementation of the SYCL 2020 API specification.
It's responsible for managing resources for enqueuing tasks to the offload
device, tracking dependencies between them, and data movement between the host
and devices. The SYCL runtime is device-agnostic and uses Unified Runtime
(https://github.com/oneapi-src/unified-runtime) as an external dependency that
serves as an interface layer between the SYCL runtime and device-specific
backends. Unified Runtime has several adapters that bind to various backends in
a similar fashion to libomptarget.

This RFC contains a brief overview of the core functionality of the SYCL runtime
and its major components. The main RFC for SYCL implementation upstreaming can
be found here: (TODO add link to the main RFC).

## Host and device code integration

Consider this SYCL code:

```
#include <sycl.hpp>
int main() {
  sycl::queue q;
  sycl::buffer<char> b{sycl::range{1024}};
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor acc{b, cgh};
    int i = 13;
    struct S {
      char c;
      int i;
    } test_s;
    test_s.c = 14;
    cgh.single_task([=] {
      if (i == 13 && test_s.c == 14) {
        acc[0] = 'a';
      }
    });
  });
}
```

The lambda passed to the `sycl::handler::single_task` function
represents a device kernel. When a SYCL application is started or a
dynamic SYCL library is loaded, all device images contained in the
multi-targeted binary are registered in the program manager, one of the
larger components of the SYCL runtime, using SYCL's `__tgt_register_lib`
counterpart, `__sycl_register_lib`. The integration header, which is
generated during device compilation and pre-included during host
compilation, helps the runtime map SYCL kernel invocations to their
corresponding symbols in the device code. The program manager uses the
information contained in device image wrappers to map the kernel name to
a set of device images, chooses an image compatible with the requested
device, passes it to the backend to be just-in-time compiled if needed,
and caches the result for reuse.

The integration header also provides information about kernel arguments.
It assumes that the lambda layout is the same for both device and host
compilation, and the runtime uses the information about argument size
and offsets to extract their values from the lambda object passed to
`sycl::handler::single_task` and set them when enqueuing the kernel. The
integration header also includes information about argument types, which
is needed to handle special cases like `sycl::accessor` objects. For
more details about the integration header, see (TODO add link to the
header/footer RFC).

## Scheduler

Another large runtime component is the scheduler. The scheduler
maintains a directed acyclic graph representation of the submitted
command groups, managing dependencies and data movement required by the
application code explicitly or with the use of accessors. Most of the
actual synchronization is handled by the underlying device backend, with
the scheduler simply enqueuing tasks as they are submitted and passing a
list of deduced dependencies. There are some exceptions to this: a SYCL
application may require access to some data on the host or submit an
asynchronous host task. Such cases are handled by the scheduler directly
by delaying the submission of device workloads until their host
dependencies are satisfied.

## Device headers and libraries

SYCL runtime headers provide various algorithms and built-in functions
to be used in device code. A large portion of these are simple enough to
be implemented in the headers (for example, those directly lowered into
SPIR-V built-ins), but some have more complex implementations (for
example, fallback implementations for functionality that might not be
supported natively). Those are available in the form of several
pre-built device libraries that are linked with the user device code
either during ahead-of-time compilation or at runtime by the program
manager, using the information about device library dependencies
embedded in device image wrappers by SYCL compilation tools.
