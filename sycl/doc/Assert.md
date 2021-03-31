# Assert feature

**IMPORTANT**: This document is a draft.

During debugging of kernel code user may put assertions here and there.
The expected behaviour of assertion failure at host is application abort.
Our choice for device-side assertions is asynchronous exception in order to
allow for extensibility.

The user is free to disable assertions by defining `NDEBUG` macro at
compile-time.


## Use-case example

```
using namespace cl::sycl;
auto ErrorHandler = [] (exception_list Exs) {
  for (exception_ptr const& E : Exs) {
    try {
      std::rethrow_exception(E);
    }
    catch (event_error const& Ex) {
      std::cout << “Exception - ” << Ex.what(); // assertion failed
      std::abort();
    }
  }
};

void user_func(item<2> Item) {
  assert((Item[0] % 2) && “Nil”);
}

int main() {
  queue Q(ErrorHandler);
  q.submit([&] (handler& CGH) {
    CGH.parallel_for<class TheKernel>(range<2>{N, M}, [=](item<2> It) {
      do_smth();
      user_func(It);
      do_smth_else();
    });
  });
  Q.wait_and_throw();
  std::cout << “One shouldn’t see this message.“;
  return 0;
}
```

In this use-case every work-item with even X dimension will trigger assertion
failure. Assertion failure should be reported via asynchronous exceptions. If
asynchronous exception handler is set the failure is reported with
`cl::sycl::event_error` exception. Otherwise, SYCL Runtime should trigger abort.
At least one failed assertion should be reported.

When multiple kernels are enqueued and both fail at assertion at least single
assertion should be reported.

## User requirements

From user's point of view there are the following requirements:

| # | Title | Description | Importance |
| - | ----- | ----------- | ---------- |
| 1 | Handle assertion failure | Signal about assertion failure via SYCL asynchronous exception | Must have |
| 2 | Print assert message | Assert function should print message to stderr at host | Must have |
| 3 | Stop under debugger | When debugger is attached, break at assertion point | Highly desired |
| 4 | Reliability | Assert failure should be reported regardless of kernel deadlock | Highly desired |

## Contents of `cl::sycl::event_error`

`cl::sycl::event_error::what()` should return the same assertion failure message
as is printed at the time being.

Other than that, interface of `cl::sycl::event_error` should look like:
```
class event_error : public runtime_error {
public:
  event_error() = default;

  event_error(const char *Msg, cl_int Err)
      : event_error(string_class(Msg), Err) {}

  event_error(const string_class &Msg, cl_int Err) : runtime_error(Msg, Err) {}

  /// Returns global ID with the dimension provided
  int globalId(int Dim) const;

  /// Returns local ID with the dimension provided
  int localId(int Dim) const;
};
```

Regardless of whether asynchronous exception handler is set or not, there's an
action to be performed by SYCL Runtime. To achieve this, information about
assert failure should be propagated from device-side to SYCL Runtime. This
should be performed via calls to `clGetEventInfo` for OpenCL backend and
`zeEventQueryStatus` for Level-Zero backend.

## Terms

 - Device-side Runtime - part of device-code, which is supplied by Device-side
   Compiler.
 - Low-level Runtime - the backend/runtime, behind DPCPP Runtime.
 - Device-side Compiler - compiler which generates device-native bitcode based
   on input SPIR-V image.
 - Accessor metadata - parts of accessor representation at device-side: pointer,
   ranges, offset.

## How it works?

For the time being, `assert(expr)` macro ends up in call to
`__devicelib_assert_fail` function. This function is part of [Device library extension](doc/extensions/C-CXX-StandardLibrary/DeviceLibExtensions.rst#cl_intel_devicelib_cassert).
Device code already contains call to the function. Currently, a device-binary
is always linked against fallback implementation.
Device-side compiler/linker provides their implementation of `__devicelib_assert_fail`
and prefer this implementation over fallback one.

If Device-side Runtime supports `__devicelib_assert_fail` then Low-Level Runtime
is responsible for:
 - detecting if assert failure took place;
 - flushing assert message to `stderr` on host.
When detected, Low-level Runtime reports assert failure to DPCPP Runtime
at synchronization points.

Refer to [OpenCL](doc/extensions/Assert/opencl.md) and [Level-Zero](doc/extensions/Assert/level-zero.md)
extensions.

If Device-side Runtime doesn't support `__devicelib_assert_fail` then a buffer
based approach comes in place. The approach doesn't require any support from
Device-side Runtime. Neither it does from Low-level Runtime.

Within this approach, a dedicated assert buffer is allocated and implicit kernel
argument is introduced. The argument is an accessor with `discard_read_write`
or `discard_write` access mode. Accessor metadata is stored to program scope
variable. This allows to refer to the accessor without modifying each and every
user's function. Fallback implementation of `__devicelib_assert_fail` restores
accessor metadata from program scope variable and writes assert information to
the assert buffer. Atomic operations are used in order to not overwrite existing
information.

Storing and restoring of accessor metadata to/from program scope variable is
performed with help of builtins. Implementations of these builtins are
substituted by frontend.

