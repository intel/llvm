# SYCL extension specifications: properties

Read together with `extension-specification.md` when a specification defines
properties or has APIs that accept them.

## Choosing a model

Three property models exist:

- SYCL 2020 (core): a class per property in a `property::<class>` namespace,
  passed in a `property_list` to a constructor or a USM allocation function
  and queried with `has_property`/`get_property` (`property::queue::in_order`;
  in extensions `ext::oneapi::property::queue::priority_high`).
- `sycl_ext_oneapi_properties`: `_key` structs, `property_value` and the
  `properties` container (`ext::intel::experimental::fp_control_key`).
- `sycl_khr_properties`: `khr::properties` with `property::key::` (draft spec
  https://github.com/KhronosGroup/SYCL-Docs/pull/980, implemented in
  `sycl/include/sycl/khr/properties.hpp`).

Which one to use:

- A new extension, or an update that adds properties to an extension that
  defines none: `sycl_khr_properties`.
- An update to an extension that already defines or accepts properties: keep
  the model it uses (SYCL 2020 or `sycl_ext_oneapi_properties`).
- Never mix the models in one extension; no container accepts another model's
  properties.

## Models and rules

- KHR model:
  * List `sycl_khr_properties` in Dependencies. It is a draft and DPC++ builds
    it only with `__DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS`, so the extension
    cannot be `supported` until the KHR is.
  * Property `foo` has its key in a nested namespace with the same name,
    `property::key::foo` (no `_key` suffix). The namespace for vendor-extension
    properties is not defined yet. Until it is, declare them in the
    extension's namespace, mirroring `khr::property` and `khr::property::key`;
    mark the location as provisional with a `[_Note:_ ... _{endnote}_]`, raise
    it in the PR and record it in "Issues".
  * Runtime values: a struct with a `constexpr` constructor and public members.
    Compile-time values: a variable template of unspecified type with
    `static constexpr` members (`using value_t` for a type value). A property
    with both kinds of value (hybrid): a class template whose template
    parameters carry the compile-time values and whose public members carry the
    runtime ones. A single value is named `value` (`value_t` for a type).
  * Boolean flags carry a boolean value (`enable_x(bool v = true)`,
    `enable_x<bool>`) rather than relying on presence, so the list has the same
    type either way. A small enum gets a variable per enumerator plus the
    variable template.
  * Each entry follows the KHR template below: the synopsis with
    `namespace key`, the effect of each value, the four `std::true_type`
    traits, and trivial copyability. `C` is the class constructed with the
    property; for any other use define a tag type that is declared but never
    defined (`khr::property::tag::reduction`).
  * An API that accepts properties takes `PropertyOrList props` (a template
    parameter) with "_Constraints:_ `khr::is_property_for_v<PropertyOrList, C>`
    is `true`, or `khr::is_property_list_for_v<PropertyOrList, C>` is `true`". A
    single property is treated as `khr::properties{props}`.
- oneAPI model: follow the oneAPI template below. Properties of this model go
  directly in `...::experimental`; `property::<class>::` nesting (Pass 3's
  `property::queue::discard_events`) is for the SYCL 2020 model.
  Vendor-specific properties live in the vendor namespace, but specializations
  of the property type traits must be in `sycl::ext::oneapi::experimental`.
  Restrict `is_property_key_of` to the intended classes. Document which
  properties each API accepts and constrain the property list: "_Constraints:_
  `PropertyListT` is one of the properties listed below ...; or
  `is_property_list_v<PropertyListT>` is true and contains no properties other
  than those listed below ...".
- SYCL 2020 model: follow the SYCL 2020 template below.
  * A property is a class in
    `sycl::ext::<vendor>::[experimental::]property::<class>`, where `<class>`
    is the SYCL class that accepts it (`queue`, `buffer`, `usm` for the USM
    allocation functions) or the extension's own class (`property::graph`).
  * A property without data has only a defaulted default constructor. A
    property with data takes it in a constructor and returns it from a
    `const` member function named after the value, like the core
    `use_mutex::get_mutex_ptr() const`.
  * Name the classes and functions whose `property_list` accepts the property.
    The core spec requires `is_property` and `is_property_of` specializations
    for every implementation property, so the specification need not list
    them.
  * Values are runtime values only; the model has no compile-time properties.
- All models:
  * Give an enum only the values supported today ("other values may be added in
    the future"). Prefer an integer parameter over an enum when future hardware
    may add values. No default value on a property that only exists to request
    non-default behavior; a separate `_automatic` property rather than a
    sentinel. Give a boolean property constructor a sensible default
    (`enable_ipc(bool enable = true)`).
  * A sub-parameter meaningful only inside one property is not a property of
    its own: make it another value of that property (KHR), a small wrapper
    class with a `static constexpr value` (oneAPI), or another constructor
    parameter (SYCL 2020).
  * State normatively whether a property is a kernel (compile-time) or launch
    property (a "runtime kernel property" is a launch property), and where it
    may not be used. Behavior on devices where it is meaningless goes in
    normative text, not a NOTE in the Status section.
  * Never define a property in terms of a specific backend extension; word it
    in terms of device capabilities.

## Templates

KHR property entry (`sycl_khr_properties`, the model for new extensions). It
shows a core property; a vendor extension substitutes its own namespace for
`sycl::khr::property` (see "Models and rules"):

```
.`khr::property::in_order`
[source,c++]
----
namespace sycl::khr::property {

struct in_order {
  constexpr in_order(bool v = true) : value{v} {}
  bool value;
};

namespace key {
struct in_order;
}

} // namespace sycl::khr::property
----

When a queue is constructed with this property and the `value` is `true`, the
effect is the same as if the queue was constructed with
`property::queue::in_order`. If the `value` is `false`, this property has no
effect on the constructed queue.

The following traits inherit from `std::true_type`:

* `khr::is_property<khr::property::in_order>`
* `khr::is_property_key<khr::property::key::in_order>`
* `khr::is_property_for<khr::property::in_order, queue>`
* `khr::is_property_key_for<khr::property::key::in_order, queue>`

`khr::property::in_order` is trivially copyable.
```

KHR compile-time property: a variable template of unspecified type holding the
value in a `static constexpr` member, plus a variable per enumerator.
`data_placement_t` is exposition only; the specification must say the
extension does not define it (Pass 4).

```
[source,c++]
----
namespace sycl::khr::property {

enum class placement_type : /*unspecified*/ { blocked, striped };

template <placement_type Placement>
struct data_placement_t { // exposition only
  static constexpr placement_type value = Placement;
};

template <placement_type Placement>
inline constexpr data_placement_t<Placement> data_placement;

inline constexpr data_placement_t<placement_type::blocked>
    data_placement_blocked;
inline constexpr data_placement_t<placement_type::striped>
    data_placement_striped;

namespace key {
struct data_placement;
}

} // namespace sycl::khr::property
----
```

oneAPI compile-time property (`sycl_ext_oneapi_properties`, for extensions that
already use it): the trait struct gets the `_key` suffix, the property variable
has the same name without it, the enum gets the `_enum` suffix, and there is a
shortcut variable per enumeration value.

```
[source,c++]
----
namespace sycl::ext::oneapi::experimental {

enum class data_placement_enum : /*unspecified*/ { blocked, striped };

struct data_placement_key {
  template <data_placement_enum Placement>
  using value_t = property_value<
      data_placement_key,
      std::integral_constant<data_placement_enum, Placement>>;
};

template <data_placement_enum Placement>
inline constexpr data_placement_key::value_t<Placement> data_placement;

inline constexpr data_placement_key::value_t<data_placement_enum::blocked>
    data_placement_blocked;
inline constexpr data_placement_key::value_t<data_placement_enum::striped>
    data_placement_striped;

} // namespace sycl::ext::oneapi::experimental
----
```

SYCL 2020 property entry (for extensions that already use the model), in the
Pass 5 per-API format:

```
.`ext::oneapi::experimental::property::queue::foo`
[source,c++]
----
namespace sycl::ext::oneapi::experimental::property::queue {

class foo {
 public:
  foo(int value);
  int get_value() const;
};

} // namespace sycl::ext::oneapi::experimental::property::queue
----

The `queue` constructors accept this property in their `property_list`
parameter. <effect of the property on the queue>.

'''

[source,c++]
----
foo(int value);
----

_Effects:_ Constructs a `foo` property holding `value`.

'''

[source,c++]
----
int get_value() const;
----

_Returns:_ The value this property was constructed with.
```
