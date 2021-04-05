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
At least one failed assertion should be reported. The assertion failure message
is printed to `stderr` by SYCL Runtime.

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

Interface of `cl::sycl::event_error` should look like:
```
class event_error : public runtime_error {
public:
  event_error() = default;

  event_error(const char *Msg, cl_int Err)
      : event_error(string_class(Msg), Err) {}

  event_error(const string_class &Msg, cl_int Err) : runtime_error(Msg, Err) {}
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
 - Device-side Compiler - compiler which generates device-native binary image
   based on input SPIR-V image.
 - Low-level Runtime - the backend/runtime behind DPCPP Runtime.
 - Accessor metadata - parts of accessor representation at device-side: pointer,
   ranges, offset.


## How it works?

For the time being, `assert(expr)` macro ends up in call to
`__devicelib_assert_fail` function. This function is part of [Device library extension](extensions/C-CXX-StandardLibrary/DeviceLibExtensions.rst#cl_intel_devicelib_cassert).
Device code already contains call to the function. Currently, a device-binary
is always linked against fallback implementation.


### Device-specific approach

Device-side compiler/linker provides their implementation of `__devicelib_assert_fail`
and prefer this implementation over fallback one.

If Device-side Runtime supports `__devicelib_assert_fail` then Low-Level Runtime
is responsible for:
 - detecting if assert failure took place;
 - flushing assert message to `stderr` on host.
When detected, Low-level Runtime reports assert failure to DPCPP Runtime
at synchronization points.

Refer to [OpenCL](extensions/Assert/opencl.md) and [Level-Zero](extensions/Assert/level-zero.md)
extensions.


### Device-agnostic approach

If Device-side Runtime doesn't support `__devicelib_assert_fail` then a buffer
based approach comes in place. The approach doesn't require any support from
Device-side Runtime and Compiler. Neither it does from Low-level Runtime.

Within this approach, a dedicated assert buffer is allocated and implicit kernel
argument is introduced. The argument is an accessor with `discard_read_write`
or `discard_write` access mode. Accessor metadata is stored to program scope
variable. This allows to refer to the accessor without modifying each and every
user's function. Fallback implementation of `__devicelib_assert_fail` restores
accessor metadata from program scope variable and writes assert information to
the assert buffer. Atomic operations are used in order to not overwrite existing
information.

Both storing of accessor metadata and writing assert failure is performed with
help of built-ins. Implementations of these builtins are substituted by
frontend.

#### Built-ins operation

Accessor is a pointer augmented with offset and two ranges (access range and
memory range).

There are two built-ins provided by frontend:
 * `__store_acc()` - to store accessor metadata into program-scope variable.
 * `__store_assert_failure()` - to store flag about assert failure in a buffer
   using the metadata stored in program-scope variable.

The accessor should be stored to program scope variable in global address space
using atomic operations. Motivation for using atomic operations: the program may
contain several kernels and some of them could be running simultaneously on a
single device.

The `__store_assert_failure()` built-in atomically sets a flag in a buffer. The
buffer is accessed using accessor metadata from program-scope variable. This
built-in return a boolean value which is `true` if the flag is set by this call
to `__store_assert_failure()` and `false` if the flag was already set.
Motivation for using atomic operation is the same as with `__store_acc()`
builtin.

The following pseudo-code snippets shows how these built-ins are used.
First of all, assume the following code as user's one:
```
void user_func(int X) {
  assert(X && “X is nil”);
}

int main() {
  queue Q(...);
  Q.submit([&] (handler& CGH) {
    CGH.single_task([=] () {
      do_smth();
      user_func(0);
      do_smth_else();
    });
  });
  ...
}
```

The following LLVM IR pseudo code will be generated for the user's code:
```
@AssertBufferPtr = global void* null
@AssertBufferAccessRange = ...
@AssertBufferMemoryRange = ...
@AssertBufferOffset = ...

/// user's code
void user_func(int X) {
if (!(X && “X is nil")) {
    __assert_fail(...);
  }
}

users_kernel(...) {
  do_smth()
  user_func(0);
  do_smth_else();
}

/// a wrapped user's kernel
kernel(AssertBufferAccessor, OtherArguments...) {
  __store_acc(AssertBufferAccessor);
  users_kernel(OtherArguments...);
}

/// __assert_fail belongs to Linux version of devicelib
void __assert_fail(...) {
  ...
  __devicelib_assert_fail(...);
}

void __devicelib_assert_fail(Expr, File, Line, GlobalID, LocalID) {
  ...
  if (__store_assert_info())
    printf("Assertion `%s' failed in %s at line %i. GlobalID: %i, LocalID: %i",
           Expr, File, Line, GlobalID, LocalID);
}

/// The following are built-ins provided by frontend
void __store_acc(accessor) {
  %1 = accessor.getPtr();
  store void * %1, void * @AssertBufferPtr
}

bool __store_assert_info(...) {
  AssertBAcc = __fetch_acc();
  // fill in data in AsBAcc
  volatile int *Ptr = (volatile int *)AssertBAcc.getPtr();
  bool Expected = false;
  bool Desired = true;

  return atomic_cas(Ptr, Expected, Desired, SequentialConsistentMemoryOrder);
  // or it could be:
  // return !atomic_exchange(Ptr, Desired, SequentialConsistentMemoryOrder);
}
```

