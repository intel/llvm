# Assert feature

**IMPORTANT**: This document is a draft.

Using the standard C++ `assert` API ("assertions") is an important debugging
technique widely used by developers. This document describes the design of
supporting assertions within SYCL device code.
The basic approach we chose is delivering device-side assertions as host-side
asynchronous exceptions, which allows further extensibility, such as better
error handling or potential recovery.

As usual, device-side assertions can be disabled by defining `NDEBUG` macro at
compile time.

## Use-case example

```c++
#include <cassert>
#include <sycl/sycl.hpp>

using namespace sycl;
auto ErrorHandler = [] (exception_list Exs) {
  for (std::exception_ptr const& E : Exs) {
    try {
      std::rethrow_exception(E);
    }
    catch (const exception& Ex) {
      if (Ex.code() == errc::ext_oneapi_assert) {
        std::cout << “Exception - ” << Ex.what(); // assertion failed
        std::abort();
      }
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
failure. Assertion failure should be reported via asynchronous exceptions with
[`assert` error code](extensions/Assert/SYCL_INTEL_assert_exception.asciidoc).
Even though multiple failures of the same or different assertions can happen in
multiple workitems, implementation is required to deliver only one. The
assertion failure message is printed to `stderr` by DPCPP Runtime.

When multiple kernels are enqueued and more than one fail at assertion, at least
single assertion should be reported.


## User requirements

From user's point of view there are the following requirements:

| # | Title | Description | Importance |
| - | ----- | ----------- | ---------- |
| 1 | Handle assertion failure | Signal about assertion failure via SYCL asynchronous exception | Must have |
| 2 | Print assert message | Assert function should print message to stderr at host | Must have |
| 3 | Stop under debugger | When debugger is attached, break at assertion point | Highly desired |
| 4 | Reliability | Assert failure should be reported regardless of kernel deadlock | Highly desired |

Implementations without enough capabilities to implement fourth requirement are
allowed to realize the fallback approach described below, which does not
guarantee assertion failure delivery to host, but is still useful in many
practical cases.


## Terms

 - Device-side Runtime - runtime library supplied by the Native Device Compiler
   and running on the device.
 - Native Device Compiler - compiler which generates device-native binary image
   based on input SPIR-V image.
 - Low-level Runtime - the backend/runtime behind DPCPP Runtime attached via the
   Plugin Interface.
 - Accessor metadata - parts of accessor representation at device-side: pointer,
   ranges, offset.


## How it works?

`assert(expr)` macro ends up in call to `__devicelib_assert_fail`. This function
is part of [Device library extension](extensions/C-CXX-StandardLibrary/DeviceLibExtensions.rst#cl_intel_devicelib_cassert).

Format of assert failure message, printed to `stderr` is the following:
```
<file>:<line>: <function>: global id: [<gid0>,<gid1>,<gid2>], local id: [<lid0>,<lid1>,<lid2>] Assertion `<expr>` failed.
```

Implementation of this function is supplied by Native Device Compiler for
safe approach or by DPCPP Compiler for fallback one.

NB: Due to lack of support of online linking in Level-Zero, the application is
linked against fallback implementation of `__devicelib_assert_fail`. Hence,
Native Device Compilers should prefer their implementation instead of the one
provided in incoming SPIR-V/LLVM IR binary.

Limitations for user after catching the "assert" asynchronous exception:
 - When using GPU device and the kernel hangs/crashes the subsequent enqueues
   will fail;

When using CPU devices the user can proceed with enqueues to the same
device/queue/context.
DPCPP Runtime remains in valid state after "assert" exception been thrown.


### Current violation

While throwing an asynchronous exception is quite an extensible way, for the
time being DPCPP Runtime merely calls `abort()`.


## Safe approach

This is the preferred approach and implementations should use it when possible.
It guarantees assertion failure notification delivery to the host regardless of
kernel behavior which hit the assertion.

The Native Device Compiler is responsible for providing implementation of
`__devicelib_assert_fail` which completely hides details of communication
between the device code and the Low-Level Runtime from the SYCL device compiler
and runtime. The Low-Level Runtime is responsible for:
 - detecting if assert failure took place;
 - flushing assert message to `stderr` on host.

When detected, Low-level Runtime reports assert failure to DPCPP Runtime
via events objects. To achieve this, information about assert failure should be
propagated from device-side to SYCL Runtime. This should be performed via calls
to `piEventGetInfo`. This Plugin Interface call "lowers" to `clGetEventInfo` for
OpenCL backend and `zeEventQueryStatus` for Level-Zero backend.

Refer to [OpenCL](extensions/Assert/opencl.md) and [Level-Zero](extensions/Assert/level-zero.md)
extensions.

The following sequence of events describes how user code gets notified:
 - Device side:
   1. Assert fails in device-code in kernel
      // It's not defined if GPU thread stops execution
      // Other GPU threads are left untouched
   2. Specialized version of `__devicelib_assert_fail` is called
   3. Device immediately signals to host (Low-Level Runtime)
 - Host side:
   1. The assert failure gets detected by Low-Level Runtime
   2. Low-Level Runtime sets event status
   3. Upon call to `sycl::queue::wait_and_throw()` or
      `sycl::event::wait_and_throw()` DPCPP Runtime checks event status and
      throws "assert" exception


### Fallback approach

If Device-side Runtime doesn't support `__devicelib_assert_fail` then a buffer
based approach comes in place. The approach doesn't require any support from
Device-side Runtime and Native Device Compiler. Neither it does from Low-level
Runtime.

Within this approach, a dedicated assert buffer is allocated and implicit kernel
argument is introduced. The argument is an accessor with `discard_read_write`
or `discard_write` access mode. Accessor metadata is stored to program scope
variable. This allows to refer to the accessor without modifying each and every
user's function. Fallback implementation of `__devicelib_assert_fail` restores
accessor metadata from program scope variable and writes assert information to
the assert buffer. Atomic operations are used in order to not overwrite existing
information.

DPCPP Runtime checks contents of the assert buffer for assert failure flag after
kernel finishes.

The following sequence of events describes how user code gets notified:
 - Device side:
   1. Assert fails in device-code in kernel
   2. Fallback version of `__devicelib_assert_fail` is called
   3. Assert information is stored into assert buffer
   4. Kernel continues running
 - Host side:
   1. Upon call to `sycl::queue::wait_and_throw()` or
      `sycl::event::wait_and_throw()` DPCPP Runtime waits until kernel finishes
      and checks assert buffer for assert information throws exception


#### Online-linking fallback `__devicelib_assert_fail`

Online linking against fallback implementation of `__devicelib_assert_fail` is
performed only when assertion is enabled.

In DPCPP headers one can see if assert is enabled with status of `NDEBUG` macro
with `#ifdef`'s. This allows to add implicit buffer argument to kernel
invocation. Here "implicit" means "implicit to the user".

When in DPCPP Runtime Library this knowledge is obtained from device binary
image descriptor's property sets.

Each device image is supplied with an array of property sets:
```c++
struct pi_device_binary_struct {
  //...
  // Array of property sets
  pi_device_binary_property_set PropertySetsBegin;
  pi_device_binary_property_set PropertySetsEnd;
};
```
Each property set is represented by the following struct:
```c++
// Named array of properties.
struct _pi_device_binary_property_set_struct {
  char *Name;                                // the name
  pi_device_binary_property PropertiesBegin; // array start
  pi_device_binary_property PropertiesEnd;   // array end
};
```
It contains name of property set and array of properties. Each property is
represented by the following struct:
```c++
struct _pi_device_binary_property_struct {
  char *Name;       // null-terminated property name
  void *ValAddr;    // address of property value
  uint32_t Type;    // _pi_property_type
  uint64_t ValSize; // size of property value in bytes
};
```

Whenever `isAssertEnabled` property set is present, this specific device image
was built with `NDEBUG` macro undefined and it requires fallback implementation
of `__devicelib_assert_fail` (i.e. if Device-side Runtime doesn't support it).

Any properties in `isAssertEnabled` property set are ignored.

The property set is added to device binary descriptor whenever at least single
translation unit was compiled with assertions enabled i.e. `NDEBUG` undefined.


##### Compiling with assert enabled/disabled

Consider the following two use-case:
```c++
// impl.cpp
using namespace sycl;
int calculus(int X) {
  assert(X && "Invalid value");
  return X * 2;
}

void enqueueKernel(queue &Q, buffer &B) {
  Q.submit([](handler &H) {
    auto Acc = B.get_access<read_write>(H);
    H.parallel_for(/* range */, [](item It) {
      assert(Acc[It]);
      // ...
    });
  });
}

// main.cpp
// ...
using namespace sycl;

SYCL_EXTERNAL int calculus(int);
void enqueueKernel(queue&, buffer&);

void workload() {
  queue Q;
  buffer B;

  Q.submit([](handler &H) {
    auto Acc = B.get_access<read_write>(H);
    H.parallel_for(/* range */, [](item It) {
      int X = calculus(0); // should fail assertion
      assert(X && "Nil in result");
      Acc[It] = X;
    });
  });

  enqueueKernel(Q, B);
  ...
}
```

These two files are compiled into a single binary application. There are four
states of definedness of `NDEBUG` macro available:

| # | `impl.cpp` | `main.cpp` |
| - | ---------- | ---------- |
| 1 | defined    | defined    |
| 2 | defined    | undefined  |
| 3 | undefined  | defined    |
| 4 | undefined  | undefined  |

States of definedness of `NDEBUG` macro defines the set of assertions which can
fail. Having assertions enabled in at least one translation unit with device
code requires for `isAssertEnabled` property set being present in device image
descriptor structure.


### Storing accessor metadata and writing assert failure to buffer

Both storing of accessor metadata and writing assert failure is performed with
help of built-ins. Implementations of these builtins are substituted by
frontend.

User's kernel is executed through a wrapper. Wrapping takes place in DPCPP
Runtime headers in a following manner:

```
class handler {

template <typename KernelName> parallel_for(KernelFunc, Range) {
#ifndef NDEBUG
  // Assert required
  if (!MQueue->get_device()->assert_fail_supported()) {
    using KName2 = class ASSERT_WRAPPER_NAME(KernelName);
    
    auto AssertBufferAcc = MQueue->get_context()->getAssertBufferAccessor(this);

    parallel_for_impl<KName2>(
      Range,
      [=](Item) {
        __store_acc(AssertBuffAcc);
        KernelFunc(Item);
      });
  } else {
#endif

     // (No assert required) OR (Assert supported by device)
     // ordinary enqueue process

#ifndef NDEBUG
  }
#endif
}

}
```


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

