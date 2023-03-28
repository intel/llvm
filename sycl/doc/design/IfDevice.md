# Implementation design for "if\_device"

This document describes the design for the DPC++ implementation of the
[sycl\_ext\_oneapi\_if\_device][1] extension.

[1]: <../extensions/proposed/sycl_ext_oneapi_if_device.asciidoc>


## Phased implementation

Although the main motivation for the "if\_device" extension is to enable a
1-pass compiler, it can still be implemented in our existing multi-pass
compiler.  This is useful because it allows us to gain experience using this
extension even before we implement the 1-pass compiler.

This document, therefore, describes two implementations.  The first is a
trivial implementation that works in the current multi-pass compiler.  The
other is the design that we will ultimately use in the 1-pass compiler.


## Multi-pass compiler implementation

This implementation requires changes only to the device headers.  The
implementation is very trivial, leveraging the existing `__SYCL_DEVICE_ONLY__`
macro which is defined differently in the host compiler pass vs. the device
compiler passes.

```
namespace sycl::ext::oneapi::experimental {
namespace detail {

// Helper object used to implement "otherwise".  The "MakeCall" template
// parameter tells whether the previous call to "if_device" or "if_host" called
// its "fn".  When "MakeCall" is true, the previous call to "fn" did not
// happen, so the "otherwise" should call "fn".
template<bool MakeCall>
class if_device_or_host_helper {
 public:
  template<typename T>
  void otherwise(T fn) {
    if constexpr (MakeCall) {
      fn();
    }
  }
};

} // namespace detail

template<typename T>
static auto if_device(T fn) {
#ifdef __SYCL_DEVICE_ONLY__
  fn();
  return detail::if_device_or_host_helper<false>{};
#else
  return detail::if_device_or_host_helper<true>{};
#endif
}

template<typename T>
static auto if_host(T fn) {
#ifdef __SYCL_DEVICE_ONLY__
  return detail::if_device_or_host_helper<true>{};
#else
  fn();
  return detail::if_device_or_host_helper<false>{};
#endif
}

} // namespace sycl::ext::oneapi::experimental
```


## Single-pass compiler implementation

This implementation requires changes to the device headers, some changes to
the error diagnostics in the front-end (CFE), and a new IR pass.

### Device headers

The device headers translate the API into calls to two functions that are
decorated with attributes named "sycl-call-if-on-device" and
"sycl-call-if-on-host".

```
namespace sycl::ext::oneapi::experimental {
namespace detail {

// Call the callable object "fn" only when this code runs on a device.
//
// IR passes recognize this function from the "sycl-call-if-on-device"
// attribute.
template<typename T>
[[clang::noinline]]
[[__sycl_detail__::add_ir_attributes_function("sycl-call-if-on-device", true)]]
void call_if_on_device(T fn) {
  fn();
}

// Call the callable object "fn" only when this code runs on the host.
//
// IR passes recognize this function from the "sycl-call-if-on-host" attribute.
template<typename T>
[[clang::noinline]]
[[__sycl_detail__::add_ir_attributes_function("sycl-call-if-on-host", true)]]
void call_if_on_host(T fn) {
  fn();
}

class call_if_on_device_helper {
 public:
  template<typename T>
  void otherwise(T fn) {
    call_if_on_device(fn);
  }
};

class call_if_on_host_helper {
 public:
  template<typename T>
  void otherwise(T fn) {
    call_if_on_host(fn);
  }
};

} // namespace detail

template<typename T>
static auto if_device(T fn) {
  detail::call_if_on_device(fn);
  return detail::call_if_on_host_helper{};
}

template<typename T>
static auto if_host(T fn) {
  detail::call_if_on_host(fn);
  return detail::call_if_on_device_helper{};
}

} // namespace sycl::ext::oneapi::experimental
```

Note the use of `[[clang::noinline]]`.  It is important that the bodies of
these functions are not inlined until after the IR pass described below.

### Changes to the front-end (CFE)

The CFE currently diagnoses some errors that are specific to device code. To do
this, the CFE must first traverse the static call tree to determine which
functions are called from kernels.  This pass of the CFE must recognize the
functions marked with the attribute "sycl-call-if-on-host" and skip the bodies
of these functions when building the static call tree of the kernels.  As a
result, the CFE will not emit any diagnostics that are specific to device code
for the callable object that is passed to these functions.

In a 1-pass compiler, we expect that the CFE will emit a single stream of
LLVM IR for both host and device.  This IR retains any calls to the functions
marked with "sycl-call-if-on-host" or "sycl-call-if-on-device" and retains the
full bodies of those functions.  The filtering described above is used only to
determine the functions that are checked for device-specific errors.

### New IR pass

The 1-pass compiler will eventually split the LLVM IR into two parts: one that
contains the device code and one that contains the host code.  We expect that
this pass will traverse the static call tree of the kernels to identify device
code.  This pass also recognizes the functions marked with
"sycl-call-if-on-host" and "sycl-call-if-on-device".  When generating the IR
for the device code, the bodies of functions marked "sycl-call-if-on-host" are
deleted, leaving empty functions.  When generating the IR for the host code,
the bodies of functions marked "sycl-call-if-on-device" are deleted.

Alternatively, the IR pass could use metadata from the CFE to identify host vs.
device code, rather than repeating the static call tree traversal here.  These
details will be resolved later as part of the 1-pass compiler design.

Up until this point, it was important to prevent inlining of the functions
marked "sycl-call-if-on-host" and "sycl-call-if-on-device".  Once the IR is
split, inlining is permitted, so this IR pass also removes the LLVM IR
`noinline` attributes from these functions.
