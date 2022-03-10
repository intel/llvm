# Implementation design for "Host Pipes"

This document describes the implementation design for the host pipes section
of the DPC++ extension [SYCL_INTEL_data_flow_pipes][1]. Pipes are a FIFO construct 
that provide links between elements of a design that are accessed through read 
and write application programming interfaces (APIs), without the notion of a
memory address/pointer to elements within the FIFO. A host pipe is a pipe that 
links a device kernel with a host program.

[1]: https://github.com/intel-sandbox/ip-authoring-specs/blob/main/Pipe/Spec/data_flow_pipes.asciidoc

## Requirements

The extension specification document referenced above contains the full set of
requirements for this feature, but a requirement that is particularly
relevant to the design, and similar in nature to one raised in the [device_global][2]
design is called out here.

This issue relates to the mechanism for integrating host and device code.
Like device global variables, host pipes are referenced in both
host and device code, so they require some mechanism to correlate the pipe
instance in device code with the pipe instance in host code. We will use
a similar mechanism as the device global implementation that creates a map
database in the integration headers and footers.

[2]: <DeviceGlobal.md>

## Design

### Changes to DPC++ headers

#### Attributes attached to the class

The `pipe` class declaration borrows the C++ attribute `sycl-host-access` from 
`device_global` to convey name information to the FPGA backend. Since this
is only needed for naming, we will set value of the property to `readwrite`.
As this attribute is also only needed for the device compiler, the `#ifdef __SYCL_DEVICE_ONLY__` 
allows the customer to ue another host compiler, even if it does not recognize these attributes.
Also note that these attributes are all in the `__sycl_detail__` namespace, so
they are considered implementation details of DPC++.  We do not intend to
support them as general attributes that customer code can use.

```
template <typename name,
          typename dataT,
          typename property_listT = property_list<>>
class pipe
{ 
  struct
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::add_ir_global_variable_attributes(
    "sycl-host-access",
    "readwrite"
    )]]
#endif
  __pipeType { const char __p; };
  
  static const __pipeType __pipe;
  ...
};
```
The `[[__sycl_detail__::add_ir_attributes_global_variable()]]` attribute is 
described more fully by the [compile-time properties][3] design 
document. This attribute is also used for other classes that have properties,
so it is not specific to the `pipe` class. 

The address of `static const char` member `__pipe` will be used to identify the pipe
in host code, and provide one half of the host-to-device mapping of the pipe 
(see the section on __New content in the integration header and footer__ below).

[3]: <CompileTimeProperties.md>

### Changes to the DPC++ front-end

There are several changes to the device compiler front-end:

* The front-end adds a new LLVM IR attribute `sycl-unique-id` to the definition
  of each `pipe` variable, which provides a unique string identifier
  for each.

* The front-end generates new content in both the integration header and the
  integration footer, which is described in more detail below.

#### New content in the integration header and footer

New content in the integration header and footer provides a mapping from the
host address of each pipe variable to the unique string for that
variable. To illustrate, consider a translation unit that defines two
`pipe` classes:

```
#include <sycl/sycl.hpp>

class some_pipe;
namespace inner {
  class some_other_pipe;
} // namespace inner
...
pipe<class some_pipe, ...>::write(...); // a usage of pipe<class some_pipe, ...>
...
pipe<class some_other_pipe, ...>::read(...); // a usage of pipe<class some_other_pipe, ...> 
...

```

The corresponding integration header defines a namespace scope variable of type
`__sycl_host_pipe_registration` (referred to below as the __host pipe registrar__)
whose sole purpose is to run its constructor before the application's main() function:

```
namespace sycl::detail {
namespace {

class __sycl_host_pipe_registration {
 public:
  __sycl_host_pipe_registration() noexcept;
};
__sycl_host_pipe_registration __sycl_host_pipe_registrar;

} // namespace (unnamed)
} // namespace sycl::detail
```

The integration footer contains the definition of the constructor, which calls
a function in the DPC++ runtime with the following information for each host
pipe that is used in the translation unit:

* The (host) address of the static member variable `__pipe`.
* The variable's string from the `sycl-unique-id` attribute.

```
namespace sycl::detail {
namespace {

__sycl_host_pipe_registration::__sycl_host_pipe_registration() noexcept {
  host_pipe_map::add(&pipe<some_pipe, ...>::__pipe,
    /* same string returned from __builtin_sycl_unique_pipe_id(pipe<some_pipe, ...>::__pipe) */);
  host_pipe_map::add(&inner::pipe<some_other_pipe>::__pipe,
    /* same string returned from __builtin_sycl_unique_pipe_id(pipe<some_other_pipe, ...>::__pipe) */);
}

} // namespace (unnamed)
} // namespace sycl::detail
```

Further details on adherence to C++ rules for unconstructed objects can be found
in the [device_global][3] design.

[3]: <DeviceGlobal.md>

Generating a unique pipe id is addressed in Open Questions below.

### Changes to the DPC++ runtime

Several changes are needed to the DPC++ runtime

* As we noted above, the front-end generates new content in the integration
  footer which calls the function `sycl::detail::host_pipe_map::add()`.
  The runtime defines this function and maintains information about all the
  device global variables in the application.  This information includes:

  - The host address of the variable.
  - The string which uniquely identifies the variable.

* The runtime implements the `read` and `write` functions of the pipe 
  class. These will use this [host pipe API][4]. These functions will
  need to retrieve the mapping added to the __host pipe registrar__
  for the pipe being read or written to, and pass it to the corresponding
  underlying OpenCL API call
  
[4]: https://github.com/intel-sandbox/ip-authoring-specs/blob/MJ_ChangeDocs4/Pipe/Spec/cl_intel_host_pipe_symbol.asciidoc

### Open Questions

* The 'unique pipe id' must be globally unique. Since all global variables in 
  the LLVM IR must have such a unique names, it is our intention to use this
  naming. Is this possible? We would also need to define a builtin to return
  this string (see 'builtin_sycl_unique_id' in the headers and footers section).

* Is the `[[__sycl_detail__::add_ir_attributes_global_variable()]]` usable on the 
  static class member `__pipe` as shown?
