# Implementation design for "device\_if" and "device\_architecture"

This document describes the design for the DPC++ implementation of the
[sycl\_ext\_oneapi\_device\_if][1] and
[sycl\_ext\_oneapi\_device\_architecture][2] extensions.

[1]: <../extensions/proposed/sycl_ext_oneapi_device_if.asciidoc>
[2]: <../extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc>


## Phased implementation

The implementation is divided into two phases.  In the first phase, we support
only [sycl\_ext\_oneapi\_device\_architecture][2] and it is supported only in
AOT mode.  The second phase adds support for both extensions in both AOT and
JIT modes.


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
* `intel_gpu_12_0_0` (alias for `intel_gpu_tgllp`)
* `intel_gpu_12_10_0` (alias for `intel_gpu_dg1`)
* `nvidia_gpu_sm_50`
* `nvidia_gpu_sm_52`
* `nvidia_gpu_sm_53`
* `nvidia_gpu_sm_60`
* `nvidia_gpu_sm_61`
* `nvidia_gpu_sm_62`
* `nvidia_gpu_sm_70`
* `nvidia_gpu_sm_72`
* `nvidia_gpu_sm_75`
* `nvidia_gpu_sm_80`
* `nvidia_gpu_sm_86`
* `nvidia_gpu_sm_87`
* `nvidia_gpu_sm_89`
* `nvidia_gpu_sm_90`
* `amd_gpu_gfx700`
* `amd_gpu_gfx701`
* `amd_gpu_gfx702`
* `amd_gpu_gfx801`
* `amd_gpu_gfx802`
* `amd_gpu_gfx803`
* `amd_gpu_gfx805`
* `amd_gpu_gfx810`
* `amd_gpu_gfx900`
* `amd_gpu_gfx902`
* `amd_gpu_gfx904`
* `amd_gpu_gfx906`
* `amd_gpu_gfx908`
* `amd_gpu_gfx909`
* `amd_gpu_gfx90a`
* `amd_gpu_gfx90c`
* `amd_gpu_gfx940`
* `amd_gpu_gfx941`
* `amd_gpu_gfx942`
* `amd_gpu_gfx1010`
* `amd_gpu_gfx1011`
* `amd_gpu_gfx1012`
* `amd_gpu_gfx1013`
* `amd_gpu_gfx1030`
* `amd_gpu_gfx1031`
* `amd_gpu_gfx1032`
* `amd_gpu_gfx1033`
* `amd_gpu_gfx1034`
* `amd_gpu_gfx1035`
* `amd_gpu_gfx1036`
* `amd_gpu_gfx1100`
* `amd_gpu_gfx1101`
* `amd_gpu_gfx1102`
* `amd_gpu_gfx1103`
* `amd_gpu_gfx1150`
* `amd_gpu_gfx1151`
* `amd_gpu_gfx1200`
* `amd_gpu_gfx1201`

The above listed device names may not be mixed with the existing target name
`spir64_gen` on the same command line.  In addition, the user must not pass the
`-device` option to `ocloc` via `-Xs` or related command line options because
the compiler driver will pass this option to `ocloc` automatically.

Note that in the first phase of implementation, only one of the above listed
GPU device names may appear on the command line.  As a result, the first phase
of implementation supports AOT compilation in this new mode only for a single
GPU device.


## Phase 1

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
* `__SYCL_TARGET_NVIDIA_GPU_SM50__`
* `__SYCL_TARGET_NVIDIA_GPU_SM52__`
* `__SYCL_TARGET_NVIDIA_GPU_SM53__`
* `__SYCL_TARGET_NVIDIA_GPU_SM60__`
* `__SYCL_TARGET_NVIDIA_GPU_SM61__`
* `__SYCL_TARGET_NVIDIA_GPU_SM62__`
* `__SYCL_TARGET_NVIDIA_GPU_SM70__`
* `__SYCL_TARGET_NVIDIA_GPU_SM72__`
* `__SYCL_TARGET_NVIDIA_GPU_SM75__`
* `__SYCL_TARGET_NVIDIA_GPU_SM80__`
* `__SYCL_TARGET_NVIDIA_GPU_SM86__`
* `__SYCL_TARGET_NVIDIA_GPU_SM87__`
* `__SYCL_TARGET_NVIDIA_GPU_SM89__`
* `__SYCL_TARGET_NVIDIA_GPU_SM90__`
* `__SYCL_TARGET_NVIDIA_GPU_SM90A__`
* `__SYCL_TARGET_AMD_GPU_GFX700__`
* `__SYCL_TARGET_AMD_GPU_GFX701__`
* `__SYCL_TARGET_AMD_GPU_GFX702__`
* `__SYCL_TARGET_AMD_GPU_GFX801__`
* `__SYCL_TARGET_AMD_GPU_GFX802__`
* `__SYCL_TARGET_AMD_GPU_GFX803__`
* `__SYCL_TARGET_AMD_GPU_GFX805__`
* `__SYCL_TARGET_AMD_GPU_GFX810__`
* `__SYCL_TARGET_AMD_GPU_GFX900__`
* `__SYCL_TARGET_AMD_GPU_GFX902__`
* `__SYCL_TARGET_AMD_GPU_GFX904__`
* `__SYCL_TARGET_AMD_GPU_GFX906__`
* `__SYCL_TARGET_AMD_GPU_GFX908__`
* `__SYCL_TARGET_AMD_GPU_GFX909__`
* `__SYCL_TARGET_AMD_GPU_GFX90A__`
* `__SYCL_TARGET_AMD_GPU_GFX90C__`
* `__SYCL_TARGET_AMD_GPU_GFX940__`
* `__SYCL_TARGET_AMD_GPU_GFX941__`
* `__SYCL_TARGET_AMD_GPU_GFX942__`
* `__SYCL_TARGET_AMD_GPU_GFX1010__`
* `__SYCL_TARGET_AMD_GPU_GFX1011__`
* `__SYCL_TARGET_AMD_GPU_GFX1012__`
* `__SYCL_TARGET_AMD_GPU_GFX1013__`
* `__SYCL_TARGET_AMD_GPU_GFX1030__`
* `__SYCL_TARGET_AMD_GPU_GFX1031__`
* `__SYCL_TARGET_AMD_GPU_GFX1032__`
* `__SYCL_TARGET_AMD_GPU_GFX1033__`
* `__SYCL_TARGET_AMD_GPU_GFX1034__`
* `__SYCL_TARGET_AMD_GPU_GFX1035__`
* `__SYCL_TARGET_AMD_GPU_GFX1036__`
* `__SYCL_TARGET_AMD_GPU_GFX1100__`
* `__SYCL_TARGET_AMD_GPU_GFX1101__`
* `__SYCL_TARGET_AMD_GPU_GFX1102__`
* `__SYCL_TARGET_AMD_GPU_GFX1103__`
* `__SYCL_TARGET_AMD_GPU_GFX1150__`
* `__SYCL_TARGET_AMD_GPU_GFX1151__`
* `__SYCL_TARGET_AMD_GPU_GFX1200__`
* `__SYCL_TARGET_AMD_GPU_GFX1201__`

If the user invokes the compiler driver with `-fsycl-targets=spir64_x86_64`,
the compiler driver must predefine the following C++ macro name:

* `__SYCL_TARGET_INTEL_X86_64__`

These macros are an internal implementation detail, so they should not be
documented to users, and user code should not make use of them.

### Changes to the device headers

The device headers implement the [sycl\_ext\_oneapi\_device\_architecture][2]
extension using these predefined macros and leverage `if constexpr` to discard
statements in the "if" or "else" body when the device does not match one of the
listed architectures.  The following code snippet illustrates the technique:

```
namespace sycl {
namespace ext::oneapi::experimental {

enum class architecture {
  x86_64,
  intel_gpu_bdw,
  intel_gpu_skl,
  intel_gpu_kbl
  // ...
};

} // namespace ext::oneapi::experimental

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

// This is true when the translation unit is compiled in AOT mode with target
// names that supports the "if_architecture_is" features.  If an unsupported
// target name is specified via "-fsycl-targets", the associated invocation of
// the device compiler will set this variable to false, and that will trigger
// an error for code that uses "if_architecture_is".
static constexpr bool is_allowable_aot_mode =
  (__SYCL_TARGET_INTEL_X86_64__ == 1) ||
  (__SYCL_TARGET_INTEL_GPU_BDW__ == 1) ||
  (__SYCL_TARGET_INTEL_GPU_SKL__ == 1) ||
  (__SYCL_TARGET_INTEL_GPU_KBL__ == 1)
  // ...
  ;

// One entry for each enumerator in "architecture" telling whether the AOT
// target matches that architecture.
static constexpr bool is_aot_for_architecture[] = {
  (__SYCL_TARGET_INTEL_X86_64__ == 1),
  (__SYCL_TARGET_INTEL_GPU_BDW__ == 1),
  (__SYCL_TARGET_INTEL_GPU_SKL__ == 1),
  (__SYCL_TARGET_INTEL_GPU_KBL__ == 1)
  // ...
};

// Read the value of "is_allowable_aot_mode" via a template to defer triggering
// static_assert() until template instantiation time.
template<ext::oneapi::experimental::architecture... Archs>
constexpr static bool allowable_aot_mode() {
  return is_allowable_aot_mode;
}

// Tells if the current device has one of the architectures in the parameter
// pack.
template<ext::oneapi::experimental::architecture... Archs>
constexpr static bool device_architecture_is() {
  return (is_aot_for_architecture[static_cast<int>(Archs)] || ...);
}

// Helper object used to implement "else_if_architecture_is" and "otherwise".
// The "MakeCall" template parameter tells whether a previous clause in the
// "if-elseif-elseif ..." chain was true.  When "MakeCall" is false, some
// previous clause was true, so none of the subsequent
// "else_if_architecture_is" or "otherwise" member functions should call the
// user's function.
template<bool MakeCall>
class if_architecture_helper {
 public:
  template<ext::oneapi::experimental::architecture ...Archs, typename T>
  constexpr auto else_if_architecture_is(T fnTrue) {
    if constexpr (MakeCall && device_architecture_is<Archs...>()) {
      fnTrue();
      return if_architecture_helper<false>{};
    } else {
      return if_architecture_helper<MakeCall>{};
    }
  }

  template<typename T>
  constexpr void otherwise(T fn) {
    if constexpr (MakeCall) {
      fn();
    }
  }
};

} // namespace detail

namespace ext::oneapi::experimental {

template<architecture ...Archs, typename T>
constexpr static auto if_architecture_is(T fnTrue) {
  static_assert(detail::allowable_aot_mode<Archs...>(),
    "The if_architecture_is function may only be used when AOT "
    "compiling with '-fsycl-targets=spir64_x86_64' or "
    "'-fsycl-targets=intel_gpu_*'");
  if constexpr (detail::device_architecture_is<Archs...>()) {
    fnTrue();
    return detail::if_architecture_helper<false>{};
  } else {
    return detail::if_architecture_helper<true>{};
  }
}

} // namespace ext::oneapi::experimental
} // namespace sycl
```

### Analysis of error checking for unsupported AOT modes

The header file code presented above triggers a `static_assert` if the
`if_architecture_is` function is used in a translation unit that is compiled
for an unsupported target. The supported targets are `spir64_x86_64`,
the new `intel_gpu_*`, `nvidia_gpu_*` and `amd_gpu_*` GPU device names.

The error checking relies on the fact that the device compiler is invoked
separately for each target listed in `-fsycl-target`.  If any target is
unsupported, the associated device compilation will compute
`is_allowable_aot_mode` as `false`, and this will trigger the `static_assert`
in that compilation phase.


## Phase 2

TBD.
