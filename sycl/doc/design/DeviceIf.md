# Implementation design for "device\_if"

This document describes the design for the DPC++ implementation of the
[sycl\_ext\_oneapi\_device\_if][1] extension, which allows device code to
conditionally execute code based on the current device.

[1]: <../extensions/proposed/sycl_ext_oneapi_device_if.asciidoc>


## Phased implementation

The implementation is divided into two phases.  In the first phase, we support
only AOT mode.  The second phase adds support also for JIT mode.


## Changes to compiler driver

Both phases require changes to the `-fsycl-targets` option that is recognized
by the compiler driver.  The problem is that the current form of that option
does not identify a specific device name.  As a reminder, the current command
line for AOT compilation on GPU looks like this:

```
$ clang++ -fsycl -fsycl-targets=spir64_gen -Xs "-device skl" ...
```

Notice that `-fsycl-targets` option specifies only the generic name
`spir64_gen` whereas the device name is passed directly to `ocloc` (the Intel
GPU AOT compiler) via `-Xs "-device skl"`.  Since the compiler driver merely
passes the `-Xs` options directly to the underlying `ocloc` without
understanding them, the compiler driver does not currently know the target
device(s) of the AOT compilation.

To fix this, the `-fsycl-targets` option should be changed to accept the
following GPU device names in addition to the target names it currently
recognizes:

* `intel_gpu_bdw`
* `intel_gpu_skl`
* `intel_gpu_kbl`
* `intel_gpu_cfl`
* `intel_gpu_apl`
* `intel_gpu_glk`
* `intel_gpu_whl`
* `intel_gpu_aml`
* `intel_gpu_cml`
* `intel_gpu_icllp`
* `intel_gpu_ehl`
* `intel_gpu_tgllp`
* `intel_gpu_rkl`
* `intel_gpu_adl_s`
* `intel_gpu_rpl_s`
* `intel_gpu_adl_p`
* `intel_gpu_adl_n`
* `intel_gpu_dg1`
* `intel_gpu_acm_g10`
* `intel_gpu_acm_g11`
* `intel_gpu_acm_g12`
* `intel_gpu_pvc`
* `intel_gpu_8_0_0` (alias for `intel_gpu_bdw`)
* `intel_gpu_9_0_9` (alias for `intel_gpu_skl`)
* `intel_gpu_9_1_9` (alias for `intel_gpu_kbl`)
* `intel_gpu_9_2_9` (alias for `intel_gpu_cfl`)
* `intel_gpu_9_3_0` (alias for `intel_gpu_apl`)
* `intel_gpu_9_4_0` (alias for `intel_gpu_glk`)
* `intel_gpu_9_5_0` (alias for `intel_gpu_whl`)
* `intel_gpu_9_6_0` (alias for `intel_gpu_aml`)
* `intel_gpu_9_7_0` (alias for `intel_gpu_cml`)
* `intel_gpu_11_0_0` (alias for `intel_gpu_icllp`)
* `intel_gpu_11_2_0` (alias for `intel_gpu_ehl`)
* `intel_gpu_12_0_0` (alias for `intel_gpu_tgllp`)
* `intel_gpu_12_10_0` (alias for `intel_gpu_dg1`)

The above listed device names may not be mixed with the existing target name
`spir64_gen` on the same command line.  In addition, the user must not pass the
`-device` option to `ocloc` via `-Xs` or related command line options because
the compiler driver will pass this option to `ocloc` automatically.

Note that in the first phase of implementation, only one of the above listed
GPU device names may appear on the command line.  As a result, the first phase
of implementation supports AOT compilation in this new mode only for a single
GPU device.


## Phase 1: AOT mode only

The first phase requires changes only to the compiler driver and to the
device headers.

### Compiler driver macro predefines

Most of the changes to the compiler driver are described above, but there are
a few small additional changes that are specific to phase 1.  If the user
invokes the compiler driver with `-fsycl-targets=<device>` where `<device>` is
one of the GPU device names listed above, the compiler driver must predefine
one of the following corresponding C++ macro names:

* `__SYCL_TARGET_INTEL_GPU_BDW__`
* `__SYCL_TARGET_INTEL_GPU_SKL__`
* `__SYCL_TARGET_INTEL_GPU_KBL__`
* `__SYCL_TARGET_INTEL_GPU_CFL__`
* `__SYCL_TARGET_INTEL_GPU_APL__`
* `__SYCL_TARGET_INTEL_GPU_GLK__`
* `__SYCL_TARGET_INTEL_GPU_WHL__`
* `__SYCL_TARGET_INTEL_GPU_AML__`
* `__SYCL_TARGET_INTEL_GPU_CML__`
* `__SYCL_TARGET_INTEL_GPU_ICLLP__`
* `__SYCL_TARGET_INTEL_GPU_EHL__`
* `__SYCL_TARGET_INTEL_GPU_TGLLP__`
* `__SYCL_TARGET_INTEL_GPU_RKL__`
* `__SYCL_TARGET_INTEL_GPU_ADL_S__`
* `__SYCL_TARGET_INTEL_GPU_RPL_S__`
* `__SYCL_TARGET_INTEL_GPU_ADL_P__`
* `__SYCL_TARGET_INTEL_GPU_ADL_N__`
* `__SYCL_TARGET_INTEL_GPU_DG1__`
* `__SYCL_TARGET_INTEL_GPU_ACM_G10__`
* `__SYCL_TARGET_INTEL_GPU_ACM_G11__`
* `__SYCL_TARGET_INTEL_GPU_ACM_G12__`
* `__SYCL_TARGET_INTEL_GPU_PVC__`

If the user invokes the compiler driver with `-fsycl-targets=spir64_x86_64`,
the compiler driver must predefine the following C++ macro name:

* `__SYCL_TARGET_INTEL_X86_64__`

These macros are an internal implementation detail, so they should not be
documented to users, and user code should not make use of them.

### Changes to the device headers

The device headers implement the [sycl\_ext\_oneapi\_device\_if][1] extension
using these predefined macros and leverage `if constexpr` to discard statements
in the "if" or "else" body when the device does not have the associated aspect.
The following code snippet illustrates the technique:

```
namespace sycl {
namespace detail {

#ifndef __SYCL_TARGET_INTEL_X86_64__
#define __SYCL_TARGET_INTEL_X86_64__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_BDW__
#define __SYCL_TARGET_INTEL_GPU_BDW__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_SKL__
#define __SYCL_TARGET_INTEL_GPU_SKL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_KBL__
#define __SYCL_TARGET_INTEL_GPU_KBL__ 0
#endif
// ...

// This is true when the translation unit is compiled in AOT mode with device
// names that supports the "if_device_has" features.  If an unsupported device
// name is specified via "-fsycl-targets", the associated invocation of the
// device compiler will set this variable to false, and that will trigger an
// error for code that uses "if_device_has".
static constexpr bool is_allowable_aot_mode =
  (__SYCL_TARGET_INTEL_X86_64__ == 1) ||
  (__SYCL_TARGET_INTEL_GPU_BDW__ == 1) ||
  (__SYCL_TARGET_INTEL_GPU_SKL__ == 1) ||
  (__SYCL_TARGET_INTEL_GPU_KBL__ == 1)
  // ...
  ;

// One entry for each aspect telling whether it is allowed to be used with
// "if_device_has".
static constexpr bool valid_aspect[] = {
  false,
  // ...
  true, // ext_intel_x86_64
  true, // ext_intel_gpu_bdw
  true, // ext_intel_gpu_skl
  true  // ext_intel_gpu_kbl
  // ...
};

// One entry for each aspect telling whether the AOT device has the
// corresponding aspect.
static constexpr bool device_capabilities[] = {
  false,
  // ...
  (__SYCL_TARGET_INTEL_X86_64__ == 1),
  (__SYCL_TARGET_INTEL_GPU_BDW__ == 1),
  (__SYCL_TARGET_INTEL_GPU_SKL__ == 1),
  (__SYCL_TARGET_INTEL_GPU_KBL__ == 1)
  // ...
};

// Read the value of "is_allowable_aot_mode" via a template to defer triggering
// static_assert() until template instantiation time.
template<aspect... Aspects>
constexpr static bool allowable_aot_mode() {
  return is_allowable_aot_mode;
}

// Tells if all of the aspects in the parameter pack are valid for use in
// "if_device_has".
template<aspect... Aspects>
constexpr static bool aspects_valid() {
  return (valid_aspect[static_cast<int>(Aspects)] && ...);
}

// Tells if the current device has every aspect in the parameter pack.
template<aspect... Aspects>
constexpr static bool device_has() {
  return (device_capabilities[static_cast<int>(Aspects)] && ...);
}

// Helper object used to implement "else_if_device_has" and "else_device".
// The "MakeCall" template parameter tells whether a previous clause in the
// "if-elseif-elseif ..." chain was true.  When "MakeCall" is false, some
// previous clause was true, so none of the subsequent "else_if_device_has" or
// "else_device" member functions should call the user's function.
template<bool MakeCall>
class if_device_has_helper {
 public:
  template<aspect ...Aspects, typename T, typename ...Args>
  constexpr auto else_if_device_has(T fnTrue, Args ...args) {
    static_assert(detail::aspects_valid<Aspects...>(),
      "Invalid aspect used in else_if_device_has.");
    if constexpr (MakeCall && device_has<Aspects...>()) {
      fnTrue(args...);
      return if_device_has_helper<false>{};
    } else {
      return if_device_has_helper<MakeCall>{};
    }
  }

  template<typename T, typename ...Args>
  constexpr void else_device(T fn, Args ...args) {
    if constexpr (MakeCall) {
      fn(args...);
    }
  }
};

} // namespace detail

namespace ext::oneapi::experimental {

template<aspect ...Aspects, typename T, typename ...Args>
constexpr static auto if_device_has(T fnTrue, Args ...args) {
  static_assert(detail::allowable_aot_mode<Aspects...>(),
    "The if-device-has function may only be used when AOT compiling with "
    "'-fsycl-targets=spir64_x86_64' or '-fsycl-targets=intel_gpu_*'");
  static_assert(detail::aspects_valid<Aspects...>(),
    "Invalid aspect used in if_device_has.");
  if constexpr (detail::device_has<Aspects...>()) {
    fnTrue(args...);
    return detail::if_device_has_helper<false>{};
  } else {
    return detail::if_device_has_helper<true>{};
  }
}

} // namespace ext::oneapi::experimental
} // namespace sycl
```

### Analysis of error checking for unsupported AOT modes

The header file code presented above triggers a `static_assert` if the
`if_device_has` function is used in a translation unit that is compiled for an
unsupported target.  The only supported targets are `spir64_x86_64` and the new
`intel_gpu_*` GPU device names.

The error checking relies on the fact that the device compiler is invoked
separately for each target listed in `-fsycl-target`.  If any target is
unsupported, the associated device compilation will compute
`is_allowable_aot_mode` as `false`, and this will trigger the `static_assert`
in that compilation phase.


## Phase 2: Both JIT and AOT modes

TBD.
