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
* `amd_gpu_gfx90a`
* `amd_gpu_gfx1010`
* `amd_gpu_gfx1011`
* `amd_gpu_gfx1012`
* `amd_gpu_gfx1013`
* `amd_gpu_gfx1030`
* `amd_gpu_gfx1031`
* `amd_gpu_gfx1032`
* `amd_gpu_gfx1034`

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
* `__SYCL_TARGET_AMD_GPU_GFX90A__`
* `__SYCL_TARGET_AMD_GPU_GFX1010__`
* `__SYCL_TARGET_AMD_GPU_GFX1011__`
* `__SYCL_TARGET_AMD_GPU_GFX1012__`
* `__SYCL_TARGET_AMD_GPU_GFX1013__`
* `__SYCL_TARGET_AMD_GPU_GFX1030__`
* `__SYCL_TARGET_AMD_GPU_GFX1031__`
* `__SYCL_TARGET_AMD_GPU_GFX1032__`
* `__SYCL_TARGET_AMD_GPU_GFX1034__`

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

The second phase supports both AOT and JIT modes, so this design replaces the
"phase 1" design described above.

This design requires changes to the headers, a new LLVM IR pass that runs at
the start of the pipeline, changes to the LLVM IR aspect propagation pass,
changes to the `sycl-post-link` tool, and changes to the DPC++ runtime.  The
bullets that follow are a high-level overview of the design.  After the
overview, each step is described in more detail.

* The extension API allows the application to conditionally call a C++ callable
  object based on whether the device has certain aspects or a certain
  architecture.  The callable is frequently a lambda expression.

* The header files use template metaprogramming which emits LLVM IR like this
  pseudocode for each of these callables:

  ```
  call void @call_if_on_device_conditionallyXXX(%callablethis, N1, N2, ...)

  define void @call_if_on_device_conditionallyXXX(%callablethis, %n1, %n2, ...)
    call void @CallableXXX(%callablethis)
  }
  ```

  where `%callablethis` is the "this" pointer for the application's callable
  object.  The parameters `N1`, `N2`, etc. are integer constants that define a
  simple expression language which tells the set of aspects or architectures
  that the device must have in order to enable the call.

* A new IR pass which runs before any optimizations translates this IR to
  pseudocode that looks like:

  ```
  call void @call_if_on_device_conditionallyXXX(@CallableXXX, %callablethis, N1, N2, ...)

  declare void @call_if_on_device_conditionallyXXX(%callable, %callablethis, %n1, %n2, ...)
  ```

  Note that the body of the function `@call_if_on_device_conditionallyXXX` has
  been removed, so there is no longer any call to `@CallableXXX`.  The IR can
  therefore be safely optimized without risk of inlining any of the callable's
  statements at its call site and without any danger of optimizing away the
  constants `N1`, `N2`, etc.

* After all of the device code IR has been optimized and linked together, the
  `sycl-post-link` tool does several things:

  * It translates the call site IR to look like:

    ```
    call void @CallableXXX(%callablethis)
    ```

  * If the target is SPIR-V:

    * It moves the definition of `@CallableXXX` and its entire call tree to a
      separate "add on" device image which defines this function as an exported
      symbol.

    * It creates a second "add on" device image that defines `@CallableXXX`
      with an empty body.

    * It creates a property set with information from `N1`, `N2`, etc., telling
      the condition under which the callable should be called.

  * If the target is for AOT:

    * It uses the device configuration file to determine if the condition
      expressed by `N1`, `N2`, etc. will be true for this device.

    * If the condition is true, it includes the definition of `@CallableXXX`
      and its entire call tree in the device image.

    * If the condition is false, it deletes the call to `@CallableXXX`.

* The logic for loading a SPIR-V module in the DPC++ runtime is changed.  When
  the module has associated "add on" device images, the following happens:

  * The runtime queries the device's aspects and architecture and then
    evaluates the metadata in the property set for the "add on" device image
    to determine whether the associated callable should be called.

  * If the call should be made, the runtime online-links the "add on" device
    image that has the application's callable.  If the call should not be made,
    it online-links the "add on" device image with an empty body.

### Definition of terms

The remaining sections provide a more detailed description of the changes in
each component.  That description uses the following terms:

* _Conditional Action_: The application's callable object, which is
  conditionally called based on the device's aspects or architecture.  The
  overview above refers to this as `@CallableXXX`.

* _Conditional Caller_: A helper function that represents a conditional call to
  the _Conditional Action_.  The overview above refers to this as the
  `@call_if_on_device_conditionallyXXX` function.

* _Condition Expression_: A sequence of integers representing a boolean
  expression tree composed of device aspects, device architectures, and the
  the logical operators AND, OR, and NOT.  This expression tree determines
  whether the _Conditional Caller_ calls the _Conditional Action_.

### Changes to the device headers

The metaprogramming in the device headers is best described in code.  The end
result of all this metaprogramming are the calls to the templated function
`call_if_on_device_conditionally`, which is the _Conditional Caller_.  This
function can be identified later in IR passes by its attribute
`sycl-call-if-on-device-conditionally`.

There are three important parts to this function:

* The body of the _Conditional Caller_ contains a single call, which is a
  direct call to the _Conditional Action_.

* The first parameter to the _Conditional Caller_ is the "this" pointer to the
  _Conditional Action_.

* The subsequent parameters to the _Conditional Caller_ are the literal integer
  values that define the _Condition Expression_.  This expression language is
  described in the comments of the code snippet below.

```
namespace sycl::ext::oneapi::experimental {
namespace detail {

// Call the callable object "fn" only when this code runs on a device which
// has a certain set of aspects or a particular architecture.
//
// Condition is a parameter pack of int's that define a simple expression
// language which tells the set of aspects or architectures that the device
// must have in order to enable the call.  See the "Condition*" values below.
template<typename T, typename ...Condition>
[[__sycl_detail__::add_ir_attributes_function("sycl-call-if-on-device-conditionally", true)]]
void call_if_on_device_conditionally(T fn, Condition...) {
  fn();
}

// The "Condition" parameter pack above is a sequence of int's that define an
// expression tree.  Each node represents a boolean subexpression:
//
// ConditionAspect -       Next int is a value from "enum aspect".  The
//                           subexpression is true if the device has this
//                           aspect.
// ConditionArchitecture - Next int is a value from "enum architecture".  The
//                           subexpression is true if the device has this
//                           architecture.
// ConditionNot -          Next int is the root of another subexpression S1.
//                           This subexpression is true if S1 is false.
// ConditionAnd -          Next int is the root of another subexpression S1.
//                           The int following that subexpression is the root
//                           of another subexpression S2.  This subexpression
//                           is true if both S1 and S2 are true.
// ConditionOr -           Next int is the root of another subexpression S1.
//                           The int following that subexpression is the root
//                           of another subexpression S2.  This subexpression
//                           is true if either S1 or S2 are true.
//
// These values are stored in the application's executable, so they are
// effectively part of the ABI.  Therefore, any change to an existing value
// is an ABI break.
//
// There is no programmatic reason for the values to be negative.  They are
// negative only by convention to make it easier for humans to distinguish them
// from aspect or architecture values (which are positive).
static constexpr int ConditionAspect = -1;
static constexpr int ConditionArchitecture = -2;
static constexpr int ConditionNot = -3;
static constexpr int ConditionAnd = -4;
static constexpr int ConditionOr = -5;

// Metaprogramming helper to construct a ConditionAnd expression for a sequence
// of aspects.  "ConditionAllAspectsBuilder<Aspects...>::seq" is an
// "std::integer_sequence" representing the expression.
template<aspect ...Aspects>
struct ConditionAllAspectsBuilder;

template<aspect Aspect, aspect ...Aspects>
struct ConditionAllAspectsBuilder<Aspect, Aspects...>  {
  template<int I1, int I2, int I3, int ...Is>
  static auto append(std::integer_sequence<int, Is...>) {
    return std::integer_sequence<int, I1, I2, I3, Is...>{};
  }
  using rest = typename ConditionAllAspectsBuilder<Aspects...>::seq;
  static constexpr int asp = static_cast<int>(Aspect);
  using seq = decltype(
    append<ConditionAnd, ConditionAspect, asp>(rest{}));
};

template<aspect Aspect>
struct ConditionAllAspectsBuilder<Aspect> {
  static constexpr int asp = static_cast<int>(Aspect);
  using seq = std::integer_sequence<int, ConditionAspect, asp>;
};

// Metaprogramming helper to construct a ConditionOr expression for a sequence
// of architectures.  "ConditionAnyArchitectureBuilder<Archs...>::seq" is an
// "std::integer_sequence" representing the expression.
template<architecture ...Archs>
struct ConditionAnyArchitectureBuilder;

template<architecture Arch, architecture ...Archs>
struct ConditionAnyArchitectureBuilder<Arch, Archs...>  {
  template<int I1, int I2, int I3, int ...Is>
  static auto append(std::integer_sequence<int, Is...>) {
    return std::integer_sequence<int, I1, I2, I3, Is...>{};
  }
  using rest = typename ConditionAnyArchitectureBuilder<Archs...>::seq;
  static constexpr int arch = static_cast<int>(Arch);
  using seq = decltype(
    append<ConditionOr, ConditionArchitecture, arch>(rest{}));
};

template<architecture Arch>
struct ConditionAnyArchitectureBuilder<Arch> {
  static constexpr int arch = static_cast<int>(Arch);
  using seq = std::integer_sequence<int, ConditionArchitecture, arch>;
};

// Metaprogramming helper to construct a ConditionNot expression.
// ConditionNotBuilder<Exp>::seq" is an "std::integer_sequence" representing
// the expression.
template<typename Exp>
struct ConditionNotBuilder {
  template<int I, int ...Is>
  static auto append(std::integer_sequence<int, Is...>) {
    return std::integer_sequence<int, I, Is...>{};
  }
  using rest = typename Exp::seq;
  using seq = decltype(append<ConditionNot>(rest{}));
};

// Metaprogramming helper to construct a ConditionAnd expression.
// "ConditionAndBuilder<Exp1, Exp2>::seq" is an "std::integer_sequence"
// representing the expression.
template<typename Exp1, typename Exp2>
struct ConditionAndBuilder {
  template<int I, int ...I1s, int ...I2s>
  static auto append(std::integer_sequence<int, I1s...>,
                     std::integer_sequence<int, I2s...>) {
    return std::integer_sequence<int, I, I1s..., I2s...>{};
  }
  using rest1 = typename Exp1::seq;
  using rest2 = typename Exp2::seq;
  using seq = decltype(append<ConditionAnd>(rest1{}, rest2{}));
};

// Metaprogramming helper to construct a ConditionOr expression.
// "ConditionOrBuilder<Exp1, Exp2>::seq" is an "std::integer_sequence"
// representing the expression.
template<typename Exp1, typename Exp2>
struct ConditionOrBuilder {
  template<int I, int ...I1s, int ...I2s>
  static auto append(std::integer_sequence<int, I1s...>,
                     std::integer_sequence<int, I2s...>) {
    return std::integer_sequence<int, I, I1s..., I2s...>{};
  }
  using rest1 = typename Exp1::seq;
  using rest2 = typename Exp2::seq;
  using seq = decltype(append<ConditionOr>(rest1{}, rest2{}));
};


// Helper function to call call_if_on_device_conditionally() while converting
// the "std::integer_sequence" for a condition expression into individual
// arguments of type int.
template<typename T, int ...Is>
void call_if_on_device_conditionally_helper(
    T fn, std::integer_sequence<int, Is...>) {
  call_if_on_device_conditionally(fn, Is...);
}

// Helper object used to implement "else_if_device_has" and "otherwise".
// The "MakeCall" template parameter tells whether a previous clause in the
// "if-elseif-elseif ..." chain was true.  When "MakeCall" is false, some
// previous clause was true, so none of the subsequent
// "else_if_device_has" or "otherwise" member functions should call the
// user's function.
template<typename MakeCallIf>
class if_device_has_helper {
 public:
  template<aspect ...Aspects, typename T,
           typename = std::enable_if<std::is_invocable_v<T>>>
  auto else_if_device_has(T fn) {
    using make_call_if = ConditionAndBuilder<
      MakeCallIf,
      ConditionAllAspectsBuilder<Aspects...>
      >;
    using make_else_call_if = ConditionAndBuilder<
      MakeCallIf,
      ConditionNotBuilder<ConditionAllAspectsBuilder<Aspects...>>
      >;

    using cond = typename make_call_if::seq;
    call_if_on_device_conditionally_helper(fn, cond{});
    return if_device_has_helper<make_else_call_if>{};
  }

  template<typename T>
  void otherwise(T fn) {
    using cond = typename MakeCallIf::seq;
    call_if_on_device_conditionally_helper(fn, cond{});
  }
};

// Same sort of helper object for "else_if_architecture_is".
template<typename MakeCallIf>
class if_architecture_is_helper {
 public:
  template<architecture ...Archs, typename T,
           typename = std::enable_if<std::is_invocable_v<T>>>
  auto else_if_architecture_is(T fn) {
    using make_call_if = ConditionAndBuilder<
      MakeCallIf,
      ConditionAnyArchitectureBuilder<Archs...>
      >;
    using make_else_call_if = ConditionAndBuilder<
      MakeCallIf,
      ConditionNotBuilder<ConditionAnyArchitectureBuilder<Archs...>>
      >;

    using cond = typename make_call_if::seq;
    call_if_on_device_conditionally_helper(fn, cond{});
    return if_architecture_is_helper<make_else_call_if>{};
  }

  template<typename T>
  void otherwise(T fn) {
    using cond = typename MakeCallIf::seq;
    call_if_on_device_conditionally_helper(fn, cond{});
  }
};

} // namespace detail

template<aspect ...Aspects, typename T>
static auto if_device_has(T fn) {
  using make_call_if = detail::ConditionAllAspectsBuilder<Aspects...>;
  using make_else_call_if = detail::ConditionNotBuilder<make_call_if>;

  using cond = typename make_call_if::seq;
  detail::call_if_on_device_conditionally_helper(fn, cond{});
  return detail::if_device_has_helper<make_else_call_if>{};
}

template<architecture ...Archs, typename T>
static auto if_architecture_is(T fn) {
  using make_call_if = detail::ConditionAnyArchitectureBuilder<Archs...>;
  using make_else_call_if = detail::ConditionNotBuilder<make_call_if>;

  using cond = typename make_call_if::seq;
  detail::call_if_on_device_conditionally_helper(fn, cond{});
  return detail::if_architecture_is_helper<make_else_call_if>{};
}

} // namespace sycl::ext::oneapi::experimental
```

### New IR pass

A new LLVM IR pass performs some simple transformations on each of the
_Conditional Caller_ functions.  This pass must run before any other passes
that perform optimizations across call boundaries, and it must be run before
the [SYCLPropagateAspectsUsagePass][3] pass.  The following transformations
are performed for each _Conditional Caller_ function _FCaller_:

[3]: <./OptionalDeviceFeatures.md#new-llvm-ir-pass-to-propagate-aspect-usage>

* The pass scans the body of function _FCaller_ looking for a `call`
  instruction.  The structure of the header file should ensure that there is
  exactly one such instruction.  The callee should be a literal (i.e. not a
  pointer) because SYCL forbids indirect function calls in device code.  If
  DPC++ is run in a mode that allows function pointers, we should still forbid
  the callable object parameter to `if_device_has` or `if_architecture_is` from
  being a function pointer.  This callee is the _Conditional Action_ function
  _FAction_.

* For each call to _FCaller_, the pass adds a new parameter at the beginning of
  the parameter list, which is the literal function name _FAction_.  The
  definition of function _FCaller_ is also changed, adding a new formal
  parameter whose type is pointer-to-function.

* The pass deletes the body of _FCaller_ and changes the function definition to
  a function declaration.  This requires changing the function from internal
  linkage to external linkage.

* Since the function now has external linkage, its name must be unique across
  all translation units.  The pass therefore renames the function to
  `call_if_on_device_conditionally_GUID_N`, where `GUID` comes from the
  `-fsycl-unique-prefix` option that is passed to clang and `N` is `1` for the
  first function _FCaller_, `2` for the second function _FCaller_, etc.

At the end of this IR pass, the call sites to the _Conditional Caller_
functions look like this, and each _Conditional Caller_ function is a function
declaration (not definition).

```
call void @call_if_on_device_conditionally_GUID_N(@CallableXXX, %callablethis,
            N1, N2, ...)
```

After this pass completes, the IR can be optimized by other passes.  There is
no danger that the body of the _Conditional Action_ will be inlined into the
_Conditional Caller_ because these call sites have been deleted.  There is also
no danger that the parameters to the _Conditional Caller_ calls will be
optimized away because _Conditional Caller_ function is now defined externally.

Note also that the _Conditional Action_ functions cannot be optimized away
because their addresses are passed to an external function.

### Normalization of condition expressions

Although it is not required for correctness, it is beneficial to rewrite the
_Condition Expressions_ at the _Conditional Caller_ call sites in a normalized
form as an additional part of the new IR pass.  Normalizing these expressions
causes equivalent expressions to have the same representation.  For example,
the condition (_ASP1_ AND _ASP2_) is equivalent to (_ASP2_ AND _ASP1_).  If
later IR passes think these are different expressions, it will lead to
unnecessary device code splits.

TODO: Describe algorithm to normalize expressions.

### Changes to the aspect propagation pass

The IR pass [SYCLPropagateAspectsUsagePass][3] propagates the set of aspects
used by each device function up the static call tree to the containing kernel.
We intentionally run this pass after the new pass described above because we do
not want aspects from the application's _Conditional Action_ functions to be
unconditionally propagated to the kernel.  These functions are called
conditionally, so the aspects they use are also conditional.

This IR pass is changed to perform the following additional aspect
propagations:

* Aspects used by each _Conditional Action_ function (and by functions it
  calls) are propagated up to the definition of the _Conditional Action_, and
  a `!sycl_used_aspects` metadata is added to the definition of that function.

* Each call to a _Conditional Caller_ function is considered to conditionally
  use the aspects in the associated _Conditional Action_ function.  The
  condition is the _Condition Expression_ at the call site.

* Aspect usage can be propagated through nested _Conditional Caller_ function
  calls.  To illustrate, consider the following example:

  * A _Conditional Caller_ named _FCallerA_ has the _Conditional Action_ named
    _FActionA_ and the _Condition Expression_ named _ExpA_.
  * The function _FActionA_ calls a different _Conditional Caller_ named
    _FCallerB_.
  * The call to _FCallerB_ has _Conditional Action_ named _FActionB_ and
    _Condition Expression_ named _ExpB_.

  When this happens, aspects used by _FActionA_ have the condition _ExpA_.
  Aspects used by _FActionB_ have the condition (_ExpA_ AND _ExpB_).

* Each kernel or exported device function is decorated with a
  `!sycl_used_aspects` metadata telling the aspects that it unconditionally
  uses and a `!sycl_conditionally_used_aspects` telling the aspects that it
  conditionally uses.  See [the section][sec-cond-meta] below for a description
  of this second metadata.

TODO: Describe interaction with the `-fsycl-fixed-targets` command line option.

### Format of the `!sycl_conditionally_used_aspects` metadata
[sec-cond-meta]: <#format-of-the-sycl_conditionally_used_aspects-metadata>

This metadata node has the following format:

```
define void @foo() !sycl_conditionally_used_aspects !0 {}
!0 = !{!1, !2, ...}         ; Each operand is one (condition, aspects) pair
!1 = !{!3, !4}              ; First operand is condition, second is aspects
!3 = !{i32 N1, i32 N2, ...} ; Condition is the series of integers in the
                            ; "Condition Expression"
!4 = !{i32 A1, i32 A2, ...} ; Aspects is a series of aspect numbers
!2 = !{!5, !6}              ; Next (condition, aspects) pair
...
```

### Simplification of conditionally used aspects

As described above, the [SYCLPropagateAspectsUsagePass][3] pass adds metadata
nodes named `!sycl_conditionally_used_aspects` to express the aspects that are
conditionally used by each kernel or exported device function.  Although it is
not required for correctness, it is beneficial to simplify this metadata by
removing "uninteresting" conditionally used aspects.  Doing this can reduce
the amount of device code splitting that occurs later.

To illustrate, consider an example where the _condition_ is "fp16 == true" and
the _aspects_ is "fp16".  In such a case, the conditional aspect usage is
uninteresting because any device where "fp16 == true" will definitely support
the "fp16" aspect.

In general, a conditionally used aspect (_ExpA_, _A_) can be removed whenever
we can prove that all devices where _ExpA_ is true will always have aspect _A_.
The following algorithm is easy to implement and will catch most of the common
cases:

* For each (_condition_, _aspects_) pair:
  * If _condition_ is an AND expression of several aspects _ASP1_ AND _ASP2_
    ..., remove all occurrences of _ASP1_, _ASP2_, etc. from the _aspects_
    list.
  * If _condition_ is an OR expression of several architectures _ARCH1_ OR
    _ARCH2_ ..., look at each aspect _ASP_ in the _aspects_ list.  Using the
    device configuration file, determine if every device architecture _ARCH1_,
    _ARCH2_, etc. has aspect _ASP_.  If so, remove _ASP_ from the _aspects_
    list.

After this simplification, some of the (_condition_, _aspects_) pairs may have
an empty set of aspects.  When this happens, remove the pair entirely from
`!sycl_conditionally_used_aspects`.  This may result in a
`!sycl_conditionally_used_aspects` metadata with no pairs.  If this happens,
remove the `!sycl_conditionally_used_aspects` metadata entirely.

### Changes to the `sycl-post-link` tool (non-AOT)

DPC++ already invokes `sycl-post-link` separately for AOT vs. non-AOT targets.
When there are both AOT and non-AOT targets specified on the command line,
DPC++ invokes the entire device compiler toolchain (including `sycl-post-link`)
multiple times.  Prior to this design, `sycl-post-link` performs the same steps
regardless of the target, but that changes with this design.  This section
describes the behavior of `sycl-post-link` when the target is not AOT.

The device code split algorithm operates normally except for the following
changes:

* When determining whether two kernels can be placed in the same device image,
  the `!used_aspects` must be the same and the
  `!sycl_conditionally_used_aspects` must be the same (the same set of
  conditions and the same set of conditionally used aspects).

* When traversing the static call tree of a kernel to form the device image,
  calls to the _Conditional Caller_ functions are handled specially.  These
  calls have a parameter which is a function pointer to the _Conditional
  Action_ function.  The device code split algorithm does *not* follow these
  function pointers when constructing the device image.  Thus, the _Conditional
  Action_ functions are not included in the device image (unless there is some
  call other than from the _Conditional Caller_ functions).

After device images are split, do the following for each device image:

* Iterate over each call to a _Conditional Caller_ function.  These calls have
  the following form:

  ```
  call void @call_if_on_device_conditionally_GUID_N(
              @CallableXXX, %callablethis, N1, N2, ...)
  ```

  In the description below, we refer to the _Conditional Action_
  (`@CallableXXX`) as _Action_ and the _Condition Expression_ (`N1`, `N2`,
  etc.) as _Condition_.

  * Maintain a table associating three pieces of information:

    * The _Condition_.
    * The _Action_ function.
    * The set of aspects used by the _Action_ function and its static call tree
      (i.e. from the `!sycl_used_aspects` metadata).
    * A generation number (described below).

  * If _Action_ is not in the table and if the device image does not already
    contain the function _Action_, add it to the table with _Condition_ and set
    the generation number to 1.  (The only time when the device image already
    contains _Action_ is when there is some call to _Action_ other than from a
    _Conditional Caller_ function.)

  * If _Action_ is in the table with the same _Condition_, get the generation
    number from that table entry.

  * Otherwise, one of two weird things has happened.  Either there are two (or
    more) calls to _Action_ with different conditions, or there is also a call
    to _Action_ from someplace other than a _Conditional Caller_.  Both cases
    are handled the same way.  Choose a generation number such that the name
    @CallableXXX.GENERATION is globally unique.  Add a new entry to the table
    with _Condition_ and that generation number.  Duplicate the function
    @CallableXXX as @CallableXXX.GENERATION.

  * Change the call to this (if the generation number is 1):

    ```
    call void @CallableXXX(%callablethis)
    ```

    or to this (if the generation number is something else):

    ```
    call void @CallableXXX.GENERATION(%callablethis)
    ```

  * Remember that the device code split algorithm did not include the
    definition of the _Action_ function in the image.  Therefore, this function
    is declared (not defined) with external linkage.

* The LLVM IR bitcode for this device image is now ready to be emitted.  We
  call this the _Main Image_.

* Using the association table described earlier, separate the _Conditional
  Action_ functions into groups such that all functions in the same group share
  the same _Condition_ and the same set of used aspects.  For each such group:

  * Construct a new device image that contains the definition of each
    _Conditional Action_ function from the group along with the static call
    tree of those functions.  This code may make additional calls to the
    _Conditional Caller_ functions, passing pointers to yet more _Conditional
    Action_ functions.  Do not follow these function pointers when constructing
    the static call tree.

  * Construct another new device image that contains stub definitions of each
    _Conditional Action_ function in the group.  Each stub defines the function
    with an empty body.

  * The definitions of the the _Conditional Action_ functions in these two
    images must have external linkage.  We call these new device images the
    _Add On Images_.

  * Each pair of _Add On Images_ (i.e. the one with the real function
    definitions and the one with the stub definitions) is assigned a unique
    name.  By convention this is just an integer in string form (e.g. "1").

  * Create a "SYCL/add on image" property set for each _Add On Image_, as
    specified below in the section [New property sets][sec-prop-sets].

  * Since the _Add On Image_ with the real function definitions may itself
    contain calls to a _Conditional Caller_ function, this _Add On Image_ is
    added to the list off all device images, such that it is processed as a
    _Main Image_ by this algorithm.

* Add a property set "SYCL/add on images" to the _Main Image_ with one entry
  for each pair of _Add On Images_ described above.  Each of these properties
  contains the associated _Condition Expression_.  See the section
  [New property sets][sec-prop-sets] for details of this property set.

* The association table is cleared before processing the next device image.

After each device image is processed by this algorithm, it goes through the
normal processing of aspects as described in the [Optional Device Features][4]
design, which includes generation of the "SYCL/device requirements" property
set.  Note that the `!sycl_conditionally_used_aspects` are not included in this
property set.

[4]: <./OptionalDeviceFeatures.md#error-checking-for-sycl_external-functions>

### Changes to the `sycl-post-link` tool (AOT)

The behavior of the `sycl-post-link` tool is different when it is invoked for
one or more AOT targets.  The following occurs for each AOT target before
device code is split into images:

* Iterate over each call to a _Conditional Caller_ function.  These calls have
  the following form:

  ```
  call void @call_if_on_device_conditionally_GUID_N(
              @CallableXXX, %callablethis, N1, N2, ...)
  ```

  * Using the device configuration file, determine if the _Condition
    Expression_ is true for this target.

  * If the condition is true, change the call site into a direct call to the
    _Conditional Action_ function like so:

    ```
    call void @CallableXXX(%callablethis)
    ```

  * If the condition is false, remove the call site entirely.

* Because we did not previously propagate the `!sycl_used_aspects` from the
  _Conditional Action_ functions up to their containing kernels, this must be
  done now.  For each call to a _Conditional Action_ function, add its
  `!sycl_used_aspects` set to each kernel that is a predecessor in the static
  call tree.

After this processing is complete, the device code is split into images in the
normal way, as described in the [Optional Device Features][5] design.

[5]: <./OptionalDeviceFeatures.md#changes-to-the-device-code-split-algorithm>

### New property sets
[sec-prop-sets]: <#new-property-sets>

When the `sycl-post-link` tool runs in non-AOT mode, it needs to communicate
information to the DPC++ runtime about the _Add On Images_ that contain the
_Conditional Action_ functions.  We do this by adding two new property sets.

The _Main Image_ contains a property set named "SYCL/add on images" which
contains one property for each of its _Add On Images_.  The name of each
property is a unique identifier for the _Add On Image_, which by convention is
just an integer in string form (e.g "1").  The value of the property has type
`PI_PROPERTY_TYPE_BYTE_ARRAY` containing a series of `int32` values `N1`, `N2`,
etc. telling the _Condition Expression_ for the _Conditional Actions_ in this
_Add On Image_.  The property's size (which is always divisible by 4) tells the
number of `int32` integers in this property.

```
[SYCL/add on images]
"1": N1, N2, N3, ...
"2": N1, N2, N3, ...
```

Each _Add On Image_ contains a property set named "SYCL/add on image" which
contains just one property.  The name of the property is the unique identifier
for this _Add On Image_ (e.g. "1").  The value of the property has type
`PI_PROPERTY_TYPE_UINT32`.  The value is `1` if the _Add On Image_ contains the
real function definitions, and it is `0` if the _Add On Image_ contains the
stub definitions.

```
[SYCL/add on image]
"1": 1
```

### Changes to the DPC++ runtime

The DPC++ runtime requires changes to identify and dynamically link the _Add On
Images_ when necessary.

When the application submits a kernel to a device, the runtime must already
find the device images from shared libraries that define exported device
functions as described in [Device Code Dynamic Linking][6].  This algorithm is
extended to look also for _Add On Images_.

If the _Main Image_ contains the "SYCL/add on images" property set, the runtime
does the following for each property in that set:

* The name of the property is the ID for a pair of _Add On Images_.  The
  runtime searches the other device images looking at their "SYCL/add on image"
  property sets in order to find the _Add On Images_ with the same ID.  There
  should be two such images: a primary image and a stub image.

* The value of the property is the _Condition Expression_ gating the
  _Conditional Actions_ in this _Add On Image_.  The runtime evaluates this
  expression for the device to which the kernel is being submitted.

* The runtime builds a set of _Add On Images_ to link against.  If the
  expression is true, the runtime adds the primary _Add On Image_ to this set.
  If the expression is false, the runtime adds the stub _Add On Image_ to this
  set.

* The selected _Add On Image_ may also contain a "SYCL/add on images" property
  set naming yet more _Add On Image_ pairs.  If so, the runtime adds these
  properties to the iteration set, causing their _Add On Images_ to be found
  also.

Once this completes, the runtime computes the union of the
"SYCL/device requirements" property sets from the _Main Image_, all the _Add On
Images_, and any images from shared libraries defining exported functions.  The
runtime then uses this combined set of device requirements to check if the
kernel is compatible with the device as described in the
[Optional Device Features][7] design.

If the kernel is compatible, the runtime creates a `pi_program` by compiling
and linking all these device images together.

[6]: <./SharedLibraries.md>
[7]: <./OptionalDeviceFeatures.md#changes-to-the-dpc-runtime>


## Alternate design for non-AOT SPIR-V targets

This is an alternate design that we can consider for SPIR-V targets that are
not AOT compiled.  This design proposes a [SPIR-V extension][8] that allows
code to be conditionally selected at JIT compilation time by setting the value
of a specialization constant.  The design then uses this feature to select the
_Conditional Actions_ when the program runs by setting specialization
constants.

Most of the design in this alternative is the same as the primary design
proposed above.  The only differences are in the `sycl-post-link` tool, the
property sets, and the DPC++ runtime.

[8]: <./spirv-extensions/SPV_INTEL_spec_conditional.asciidoc>

### Changes to the `sycl-post-link` tool (non-AOT)

The device code split algorithm must still be changed in the same way as the
primary design above describes.  After device code is split into images, the
following happens for each image.

(This part of the design is not very detailed yet and needs to be fleshed out.)

We need to compute the _Condition Expression_ that gates each of the device
functions.  This can be done by traversing the static call tree starting at the
kernel functions.  Each kernel function has an empty condition, indicating that
the kernel functions are not gated by any condition.  The following rules are
used to determine the condition for each remaining function _F_ in the call
tree:

* If all of the callers of _F_ have no condition, then _F_ has no condition.

* Otherwise, the condition for _F_ is the union of the conditions of all of
  its callers (i.e _C1_ OR _C2_ ...).

* A call to a _Conditional Caller_ propagates the _Condition Expression_ to
  the _Conditional Action_ function as follows.  Consider a function _F_ that
  has condition _CF_.  Function _F_ calls a _Conditional Caller_ with condition
  _CAction_ and a _Conditional Action_ function _G_.  The condition for
  function _G_ is (_CF_ AND _CAction_).

Iterate over each call to a _Conditional Caller_ function.  These calls have
the following form:

```
call void @call_if_on_device_conditionally_GUID_N(
            @CallableXXX, %callablethis, N1, N2, ...)
```

Change each such call site into a direct call to the _Conditional Action_
function like so:

```
call void @CallableXXX(%callablethis)
```

And also record a _Condition Expression_ that gates the call site, which is
the same _Condition Expression_ from the _Conditional Caller_.

We now have a gating condition for some of the device functions and for some of
the call sites.  Each of these conditions must be converted into a SPIR-V
specialization constant expression.  The leaves of this expression are
**OpSpecConstantFalse** instructions representing either an aspect or an
architecture.  The other nodes in this expression are **OpSpecConstantOp**
instructions representing the AND, OR, and NOT operations in the expression.

Although it is not necessary for correctness, it is advantageous to optimize
the specialization constant expressions to eliminate duplicate expression
nodes.  In particular, it is advantageous to eliminate duplicate leaf nodes
because this will reduce the number of elements in the
"SYCL/special specialization constants" property set (described below), and
thus reduce the amount of work the runtime needs to do when it loads a device
image.

When generating SPIR-V from the LLVM IR, the following must happen:

* Each function definition that has a gating condition _C_ must have its
  **OpFunction** instruction decorated with **ConditionalINTEL**.

* Each function call that has a gating condition _C_ (i.e. a
  _Conditional Caller_) must have its **OpCall** instruction decorated with
  **ConditionalINTEL**.

* Any capability that is used only from functions decorated with
  **ConditionalINTEL** must result in **OpConditionalCapabilityINTEL** rather
  than **OpCapability**.

* Any **OpTypeXXX** instruction that is referenced only from functions
  decorated with **ConditionalINTEL** must also be decorated with
  **ConditionalINTEL**.

The `sycl-post-link` tool must also generate the new property set named
"SYCL/special specialization constants", and it must generate new properties
in the "SYCL/device requirements" property set.  These properties are
described in the [section below][sec-alt-prop-sets].

The "SYCL/special specialization constants" property set is constructed from
the leaf nodes of the SPIR-V specialization constant expressions described
above.  Leaf nodes that represent aspects are added to the "aspects" property,
and leaf nodes that represent architectures are added to the "architectures"
property

The device code split algorithm ensures that all kernels in the same device
image have the same value for their `!sycl_conditionally_used_aspects`
metadata.  The value of this metadata can be used to create the
"conditional-aspects-_N_" and "condition-_N_" properties.

### New properties and property sets
[sec-alt-prop-sets]: <#new-properties-and-property-sets>

When the `sycl-post-link` tool runs in non-AOT mode, it needs to communicate
information to the DPC++ runtime about the specialization constants that must
be set in order to select the _Conditional Action_ functions.  We do this by
adding a new property set named "SYCL/special specialization constants".  There
are two properties in this set, one named "aspects" and another named
"architectures".  The value of each has type `PI_PROPERTY_TYPE_BYTE_ARRAY`
containing a series of `uint32` values.  The property's size (which is always
divisible by 4) tells the number of `uint32` integers.

The property "aspects" is a series of pairs (_SpecID_, _aspect_), where
_SpecID_ is the ID of an **OpSpecConstantFalse** instruction representing the
aspect whose numerical value in `enum aspect` is _aspect_.  The property
"architectures" is a series of pairs (_SpecID_, _arch_), where _SpecID_ is the
ID of an **OpSpecConstantFalse** instruction representing the architecture
whose numerical value in `enum architecture` is _arch_.

```
[SYCL/special specialization constants]
"aspects": SpecID1, aspect1, SpecID2, aspect2, ...
"architectures": SpecID3, arch1, SpecID4, arch2, ...
```

The `sycl-post-link` tool must also communicate information to the DPC++
runtime about the aspects that are conditionally used in an image.  We do this
by adding new properties to the existing "SYCL/device requirements" property
set.  These properties have names of the form "condition-_N_" and
"conditional-aspects-_N_", where _N_ is an integer in string form.  The value
of a "condition-_N_" property has type `PI_PROPERTY_TYPE_BYTE_ARRAY` containing
a series of `int32` values representing a _Condition Expression_.  The value
of a "conditional-aspects-_N_" property has type `PI_PROPERTY_TYPE_BYTE_ARRAY`
containing a series of `uint32` values representing aspect values.  In both
cases, the property's size (which is always divisible by 4) tells the number
of values.

These properties are grouped in pairs, matched by the value of _N_.  For
example, "condition-0" is matched with "conditional-aspects-0", etc.  The
"conditional-aspects-_N_" property tells the set of aspects that the image uses
when the device satisfies the condition expressed in "condition-_N_".

```
[SYCL/device requirements]
"condition-0": N1, N2, N3, ...
"conditional-aspects-0": aspect1, aspect2, ...
"condition-1": N4, N5, N6, ...
"conditional-aspects-1": aspect3, aspect4, ...
```

### Changes to the DPC++ runtime

The DPC++ runtime changes in several ways.  Before creating a `pi_program` from
a device image, the runtime examines the "SYCL/device requirements" property
set to see if the aspects used by the image are compatible with the device.
This logic is extended to make use of the "conditional-aspects-_N_" properties.
The runtime evaluates each _Condition Expression_ in "condition-_N_" and adds
the associated aspects to the requirements if the expression is true.  Existing
code in the DPC++ runtime then checks that the device has all of these aspects
before attempting to create a `pi_program` from the device image.

After creating the `pi_program`, the runtime examines the aspect and
architecture values in the "SYCL/special specialization constants" property
set.  Whenever the device has the aspect or architecture, the runtime calls
`piextProgramSetSpecializationConstant` to set the associated _SpecID_ to true.
Setting these specialization constants causes the _Conditional Actions_ to be
selected according to the device capabilities.

Ideally, each backend UMD (e.g. Level Zero, OpenCL) would support the proposed
[SPV\_INTEL\_spec\_conditional][8] extension.  However, if these backends
cannot be updated in time, the runtime can implement a preprocessing step as a
SPIR-V to SPIR-V translator that does the code specialization.  The output of
this translator can then be sent to the backend UMD even if it does not support
SPV\_INTEL\_spec\_conditional.


## Alternate design for AOT targets

This is an alternate design that we can consider for targets that are AOT
compiled.  This alternate design is orthogonal to the alternate design
presented above for non-AOT targets, so we could implement either alternate
design on its own or both together.

In any AOT design, we need to resolve the _Conditional Actions_ at some point
in the AOT compilation.  In the primary design, this is done in
`sycl-post-link`.  This has both advantages and disadvantages.  The
disadvantage is that we may lose some LLVM IR optimizations because the
optimization passes (which run before `sycl-post-link`) operate on the code
before the _Conditional Actions_ are resolved.  As an example, the optimizer
won't be able to inline the body of these _Conditional Action_ functions at
their call sites.  We think this will be mitigated for Intel GPU targets,
though, because `ocloc` (which runs after `sycl-post-link`) can perform these
optimizations.

As an alternative design, we could instead resolve the _Conditional Actions_ in
an LLVM IR pass that happens much earlier in the pipeline.  This would allow
the LLVM optimization passes to better optimize the code because the
_Conditional Action_ functions will either be direct calls or they will be
entirely eliminated by the time the optimizer passes run.  If we do this, it
makes sense to resolve the _Conditional Actions_ sometime before the
[SYCLPropagateAspectsUsagePass][3] runs.  The algorithm would be similar to
what `sycl-post-link` does in the primary design:

* Iterate over each call to a _Conditional Caller_ function.  These calls have
  the following form:

  ```
  call void @call_if_on_device_conditionally_GUID_N(
              @CallableXXX, %callablethis, N1, N2, ...)
  ```

* Using the device configuration file, determine if the _Condition
  Expression_ is true for this target.

* If the condition is true, change the call site into a direct call to the
  _Conditional Action_ function like so:

  ```
  call void @CallableXXX(%callablethis)
  ```

* If the condition is false, remove the call site entirely.

The SYCLPropagateAspectsUsagePass will then propagate aspects used by the
remaining _Conditional Actions_ up to the kernel functions as
`!sycl_used_aspects` metadata (i.e. unconditionally used aspects).

The disadvantage to this design is that it increases compilation time when
there are multiple targets.  Once the _Conditional Actions_ are resolved,
the LLVM IR is now specialized for one particular AOT target.  If the user has
asked to compile for multiple targets, we need to split the IR prior to
resolving the _Conditional Actions_ and run the LLVM IR pipeline separately for
each copy of the IR.

It would be possible to use this alternate design for some AOT targets but not
others.  For example, we may decide that `ocloc` can mitigate any lost
optimizations for Intel GPU targets, so the primary design may make sense for
those targets.  However, we might decide that other targets (e.g. Nvidia) work
better with the alternate design.
