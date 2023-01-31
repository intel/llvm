# Implementation design for `sycl::any_device_has` and `sycl::all_devices_have`

This design document describes the implementation of the SYCL 2020 device aspect
traits `any_device_has` and `all_devices_have` as described in the
[SYCL 2020 Specification Rev. 6 Section 4.6.4.3][1].

In summary, `any_device_has<aspect>` and `all_devices_have<aspect>` must inherit
from either `std::true_t` or `std::false_t` depending on whether the
corresponding compilation environment can guarantee that any and all the
supported devices support the `aspect`. Since DPC++ allows for compiling for
multiple targets, these traits can be different when compiling for the
individual targets and on host.

The design of these traits is inspired by the implementation of the
[sycl\_ext\_oneapi\_device\_if][2] and
[sycl\_ext\_oneapi\_device\_architecture][3] extensions as described in
[DeviceIf.md][4]. Additionally, it leverages part of the design for optional
kernel features, as described in [OptionalDeviceFeatures.md][5].

## Changes to the compiler driver

Using the `-fsycl-targets` options introduced in [DeviceIf.md][4] and the
configuration file introduced in [OptionalDeviceFeatures.md][5], the compiler
driver finds the set of all aspects supported by each specified target. Note
that in this section we refer to aspects as their integral representation as
specified in the device headers rather than by the names specified in the
[SYCL 2020 specification][1].

For each target $t$ in `-fsycl-targets`, let $A^{any}_t$ be the set of aspects
supported by any device supporting $t$ and let $A^{all}_t$ be the set of aspects
supported by all devices supporting $t$. If $t$ has an entry in the
configuration file, these sets are defined by the `aspects` list in that entry
and $A^{any}_t = A^{all}_t$. If there is no entry for $t$ in the configuration
file, then $A^{any}_t$ is the set of all aspects and $A^{all}_t = \emptyset$.

In the device-side compilation of a SYCL program for $t$ the driver defines the
following macros:
* `__SYCL_ALL_DEVICES_HAVE_`$i$`__` as `1` for all $i$ in $A^{all}_t$.
* `__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__` as `1` if $A^{any}_t$ is the set of all
aspects.
* `__SYCL_ANY_DEVICE_HAS_`$j$`__` as `1` for all $j$ in $A^{any}_t$ if
`__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__` was not defined.

In the host-side compilation of a SYCL program, where $[t1, t2, \ldots, tn]$ are
the $n$ targets specified in `-fsycl-targets`, the driver defines the following
macros:
* `__SYCL_ALL_DEVICES_HAVE_`$i$`__` as `1` for all $i$ in
${\bigcap}^n_{k=1} A^{all}_{tk}$.
* `__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__` as `1` if
${\bigcup}^n_{k=1} A^{any}_{tk}$ is the set of all aspects.
* `__SYCL_ANY_DEVICE_HAS_`$j$`__` as `1` for all $j$ in
${\bigcup}^n_{k=1} A^{any}_{tk}$ if `__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__` was not
defined.

Note that the need for the `__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__` macro is
due to the special case where the driver finds no configuration for a target and
must assume that there exists some device that supports any given aspect. Since
the driver has no way of knowing all possible aspects, we use a catch-all macro
to denote this case instead. This is not needed for $A^{all}_t$ for any target
$t$, as the driver will always know all relevant aspects.

## Changes to the device headers

Using the macros defined by the driver, the device headers define the traits
together with specializations for each aspect:

```c++
namespace sycl {
template <aspect Aspect> all_devices_have;
template<> all_devices_have<aspect::host> : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_0__ + 0> {};
template<> all_devices_have<aspect::cpu> : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_1__ + 0> {};
template<> all_devices_have<aspect::gpu> : std::bool_constant<__SYCL_ALL_DEVICES_HAVE_2__ + 0> {};
...

#ifdef __SYCL_ANY_DEVICE_HAS_ANY_ASPECT__
// Special case where any_device_has is trivially true.
template <aspect Aspect> any_device_has : std::true_t {};
#else
template <aspect Aspect> any_device_has;
template<> any_device_has<aspect::host> : std::bool_constant<__SYCL_ANY_DEVICE_HAS_0__ + 0> {};
template<> any_device_has<aspect::cpu> : std::bool_constant<__SYCL_ANY_DEVICE_HAS_1__ + 0> {};
template<> any_device_has<aspect::gpu> : std::bool_constant<__SYCL_ANY_DEVICE_HAS_2__ + 0> {};
...
#endif // __SYCL_ANY_DEVICE_HAS_ANY_ASPECT__

template <aspect Aspect> constexpr bool all_devices_have_v = all_devices_have<Aspect>::value;
template <aspect Aspect> constexpr bool any_device_has_v = any_device_has<Aspect>::value;
} // namespace sycl
```

Note that the driver may not define macros for all aspects, so the `+ 0` is
used to ensure the boolean constant value of the specializations become `false`
when the corresponding macro is undefined.

Since the specializations need to be explicitly specified, there is a high
probability of mistakes when new aspects are added. To avoid such mistakes, a
SYCL unit-test uses the [aspects.def](../../include/sycl/info/aspects.def) file
to generate test cases, ensuring that specializations exist for all aspects:

```c++
#define __SYCL_ASPECT(ASPECT, ASPECT_VAL)                                          \
  constexpr bool CheckAnyDeviceHas##ASPECT = any_devices_has_v<aspect::ASPECT>;    \
  constexpr bool CheckAllDevicesHave##ASPECT = all_devices_have_v<aspect::ASPECT>;

#include <sycl/info/aspects.def>

#undef __SYCL_ASPECT
```

This relies on the fact that unspecialized variants of `any_device_has` and
`all_devices_have` are undefined.

[1]: <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:device-aspects>
[2]: <../extensions/proposed/sycl_ext_oneapi_device_if.asciidoc>
[3]: <../extensions/proposed/sycl_ext_oneapi_device_architecture.asciidoc>
[4]: <DeviceIf.md>
[5]: <OptionalDeviceFeatures.md>
