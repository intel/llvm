# SYCL extension specifications: write, update, review

Covers new and updated specifications in `sycl/doc/extensions`, written and
reviewed the way DPC++ specification reviewers expect. Passes 1-8 apply to both
writing and reviewing.

## Writing or updating a specification

1. Start from `sycl/doc/extensions/template.asciidoc` and use its formatting. Do
   not copy an old extension; you would inherit its outdated format. Follow the
   lifecycle in `sycl/doc/extensions/README-process.md` (where the file goes,
   how a new version is added, how a spec is split into versions).
2. Consult the core SYCL specification as described below.
3. Consider whether the extension could be promoted to a Khronos extension.
   Compare with the `sycl_khr_*` extensions in SYCL-Docs `adoc/extensions/` and
   their headers in `sycl/include/sycl/khr/`.
4. Read the implementation. For an update, the extension's header under
   `sycl/include/sycl/ext/<vendor>/[experimental/]` is the current API
   surface: the synopsis must match it, and anything not implemented goes in
   "Future direction" or a NOTE (Pass 1). For a new extension, check the
   neighbouring headers and specs for API to reuse and for names, argument
   order and `const` conventions to stay consistent with (Pass 3, Pass 7).
5. Follow Passes 1-8 while drafting. The passes are phrased for a reviewer;
   when drafting, read every "flag/propose/push back on X" as "do not write X"
   and every "is Y present?" as "make Y present". Include at least one usage
   example. Capture open questions in an "Issues" section and deferred ideas in
   "Future direction".
6. Before posting the PR, self-review against Passes 1-8 and check the
   rendered document (see "Rendering").

## Reviewing a specification

1. Get the diff and the *whole* file — most defects are in the interaction
   between the change and the rest of the document (a renamed identifier left
   stale elsewhere, a table row that no longer matches the synopsis).
2. Read the rendered document, not only the source (see "Rendering").
3. Compare against `sycl/doc/extensions/template.asciidoc`. Some of the old
   extensions are *not* a style reference.
4. Walk the passes below in order. Passes 1-4 are cheap and catch most review
   comments; passes 5-8 are where the design discussions live.
5. Write comments as concrete replacement wording (an Asciidoc ```suggestion```
   block or a block quote of the proposed paragraph), not just as a complaint —
   that is how these reviews are conducted and it is what makes them converge.
   See also "Reporting" at the end.

## Core SYCL specification

The "core SYCL specification" is the SYCL 2020 revision named in the
extension's Dependencies section:
https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html. Its
sources are in https://github.com/KhronosGroup/SYCL-Docs (`adoc/chapters/`), and
the Khronos `sycl_khr_*` extensions are in `adoc/extensions/`. The spec is too
large to read whole — shallow-clone SYCL-Docs outside this repo, grep for the
sections, classes and terms the extension touches, and read only those.

## Rendering

Asciidoc defects (cut-off tables, `C++` rendering as `C`, notes rendering as
plain text) are invisible in the source and in the diff. Render locally with
`asciidoctor -o /tmp/<name>.html <file>.asciidoc` if `asciidoctor` is
installed; otherwise GitHub renders `.asciidoc` files in the PR file view and on
any pushed branch.

## Templates

KHR property entry (`sycl_khr_properties`, the model for new extensions; the
namespace for vendor extensions is still open, see "Properties"):

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

```c++
namespace sycl::khr::property {

enum class placement_type : /*unspecified*/ { blocked, striped };

template <placement_type Placement>
struct __unspecified_type__ {
  static constexpr placement_type value = Placement;
};

template <placement_type Placement>
inline constexpr __unspecified_type__<Placement> data_placement;

inline constexpr __unspecified_type__<placement_type::blocked>
    data_placement_blocked;
inline constexpr __unspecified_type__<placement_type::striped>
    data_placement_striped;

namespace key {
struct data_placement;
}

} // namespace sycl::khr::property
```

oneAPI compile-time property (`sycl_ext_oneapi_properties`, for extensions that
already use it): the trait struct gets the `_key` suffix, the property variable
has the same name without it, the enum gets the `_enum` suffix, and there is a
shortcut variable per enumeration value.

```c++
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
```

Information descriptor:

```
.`info::device::device_type`
[source,c++]
----
namespace sycl::info::device {
struct device_type {
  using return_type = info::device_type;
};
} // namespace sycl::info::device
----

_Remarks:_ Template parameter to `device::get_info`.

_Returns:_ The device type associated with the device.
```

Per-API entry: a `'''` horizontal rule, a `[source,c++]` synopsis, then the
`_Preconditions:_`/`_Constraints:_`/`_Effects:_`/`_Returns:_`/`_Throws:_`
paragraphs. Prefer this over tables; use a table only for properties,
descriptors and enumerators.

## Pass 1 — Structure and boilerplate

- Section order: Notice, Contact, Dependencies, **Status**, [Backend support
  status], Overview, Specification, [Implementation notes], Example(s),
  [Future direction], [Issues]. Dependencies before Status. "Usage Example",
  "Future Directions" and "Issues" must be *peers* of "Specification", not
  subsections. "Feature test macro" is the first subsection of Specification.
- `Contact` follows the template (GitHub issues).
- Status text matches the template for the actual state, including the bold
  "*Shipping software products should not rely on APIs defined in this
  specification.*". No extra "IMPORTANT: this is a draft" line.
- "This extension is written against the SYCL 2020 revision N specification" —
  is N current? Are the referenced section numbers still right?
- Dependencies list other extensions by extension name, with relative links
  (`link:../experimental/sycl_ext_oneapi_properties.asciidoc[...]`): relative
  links resolve to the same revision of the repo, absolute ones always point at
  the tip.
- Copyright = year of first publication, **not** bumped by this PR. No revision
  history table, no contributors list.
- Feature-test macro name is exactly the extension name in upper case, and is
  not reused for anything else. For an unversioned experimental extension the
  value table says "The APIs of this experimental extension are not versioned,
  so the feature-test macro always has this value."
- Extension name == title line == file name, all lower case, numbers included
  (`memcpy2d`). Backend-specific extension names include the backend.
- Location matches the state: `proposed/` if unimplemented, `experimental/`
  (and `...::experimental` namespaces) if implemented but unstable, `supported/`
  if stable. A "supported" extension may not depend on an experimental one, and
  may not break source compatibility. Moves between directories, version bumps
  and deprecation signposts follow `sycl/doc/extensions/README-process.md`.
- "Backend support status" section: present only if there is a real limitation,
  placed right after Status, and preferably expressed as a device aspect with
  `errc::kernel_not_supported` rather than as undefined behavior. With no aspect
  to express it, state plainly that use on other backends is undefined behavior
  with no diagnostic. Delete the section if it would only say "not yet
  implemented".
- Implementation gaps are a different thing from backend limitations.
  `sycl/doc/extensions/README-process.md` *requires*, also for experimental
  extensions, a non-normative `[_Note:_ ... _{endnote}_]` for every API, device
  or backend not yet implemented — in Status when a whole backend is missing,
  next to the API otherwise. Do not ask for such a note to be removed; do ask
  for one when the implementation PR leaves something out. Normative behavior
  on devices where the feature is meaningless is specified in the API text,
  never in such a note (see "Properties").
- New APIs added by a new version: is there a new row in the feature-test macro
  table, and a "This API is available starting in version N of this
  specification." line under each new section (the wording from
  `sycl/doc/extensions/README-process.md`; say "version", not "revision", which
  is reserved for the core SYCL specification)?
- Lines wrapped at 80 columns; links relative (not absolute); every link still
  resolves after any file move.

## Pass 2 — Mechanical defects (fast, always productive)

- Unclosed or missing backticks; `Note:` vs `[_Note:_ ... _{endnote}_]` (inside
  a bullet list, start the note as a second paragraph with a leading `+`);
  `[source,c++]` (not `[source,dpcpp]`, not a bare triple-backtick block).
- `{cpp}` (a built-in attribute; no `:cpp:` definition needed, the template
  has none) instead of a literal `C++`, which can render as `C`; `+...+`
  escaping for `+__FILE__+`, `+__shared__+`, `+T...+`, `+<=+`, `+this->x()+`.
- Asciidoctor link syntax (`https://example.com[text]`), not Markdown. No
  `stem:`/`latexmath:`; write `O(N*log_2(N))` as plain text.
- Doubled words, typos, stale identifiers after a rename, leftovers from the
  template or from a copy-pasted extension, `Built On: {docdate}`, "this doc is
  better viewed with asciidoctor", pasted web-page cruft.
- Synopsis vs description table: every declaration described exactly once and
  nothing described that isn't declared; rows in the same order as the synopsis;
  `const`/`noexcept` consistent between the two; parameter names identical.
- Terminology: "member function" not "method"; "linearized" not "flattened";
  "work-item"/"work-group"/"sub-group" hyphenated; "implementation" not
  "backend"; "the core SYCL specification" not "SYCL 2020"; "a `std::string`";
  no possessive/plural forms of C++ type names ("`external_mem` objects").
  "Command" covers kernels, explicit memory operations and host tasks. No
  vocabulary of the layer beneath ("shader", "invocation", "thread block").
- Section titles lower case except the first word, phrased symmetrically, using
  core-spec terminology ("information descriptor", "property", "aspect",
  "command").

## Pass 3 — Naming

- Vendor string right? `oneapi` for generally-useful features, `intel` (or the
  vendor) for hardware-tied ones. Namespace matches vendor + experimental status,
  and backend-specific APIs have the backend level
  (`sycl::ext::oneapi::experimental::cuda`).
- Members added to **core** classes carry the `ext_<vendor>_` prefix; members of
  the extension's own classes do not. New core enumerators carry the prefix.
  `experimental` appears only in namespaces.
- Information descriptors are in the extension namespace
  (`ext::oneapi::experimental::info::device::foo`), *not*
  `info::device::ext_oneapi_foo` (deprecated), and the namespace order is
  `...::info::device`.
- Identifiers lower case including abbreviations (cache level names `L1`, `L2`
  are the accepted exception); no acronyms; no abbreviations where the full word
  fits (`if_architecture_is`, not `if_arch_is`).
- Does the name describe the concept and read well at the use site? Push back on
  names that describe a mechanism, repeat their enclosing scope, or read
  ambiguously (`wait_request_not_requested`, `read_write_mode_readwrite`,
  `build_log(log)`). Good: `device_image_scope` not `device_image_life`,
  `host_access` not `copy_access`, `init_mode` not `init_via`, `grf_size` not
  `register_size`. Words dropped because the enclosing scope provides them
  (`property::queue::discard_events`) stay in the feature-test macro
  (`SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS`).
- Group APIs: `joint_*` (all work-items pass the same range) vs
  `*_over_group`/`group_*` (per-work-item value) — does the name match the
  semantics as currently specified?

## Pass 4 — Synopsis hygiene

- Any implementation detail visible? `detail::`, `enable_if`/SFINAE, checker
  traits, dispatch specializations, private members, `friend`, data members,
  function bodies, a destructor other specs omit. All of these should be replaced
  by prose, a comment, or `/*unspecified*/` / `/* implementation-defined */` /
  `/*return-type*/`.
- Exposition-only traits flagged with a note that the extension does not define
  them.
- No namespace aliases inside the synopsis. `namespace sycl::ext::...` one-liner
  form, declarations at column 0, closing comment, balanced braces.
- Members added to an existing class: summary synopsis inside a
  `class X { /* ... */ };` skeleton first, then abbreviated per-API entries. The
  whole core class must not be reprinted.
- Numeric callouts written `// (1)`, aligned to the right, only where one block
  declares several functions.
- Signature review: pointless `const` on by-value parameters; `const&` vs `&`
  used consistently; query functions `const`; callables perfect-forwarded;
  copy/move constructors listed and `= default`-ed if trivial copyability is
  required; no out-parameters filled in after the call; `size_t` for sizes;
  `ptrdiff_t`/`size_t` for pointer arithmetic; `std::vector` for `get_info`
  results; no different signature in host vs device code.
- A `const` pointer wrapper means the *pointer* is const, not the pointee.
  Classes with a deleted copy constructor are passed by reference. Say
  "trivially copyable" (stronger than "device copyable") when that is required.
- Say "null pointer value" rather than `nullptr`. Use C++ terms of art correctly
  ("trivial type", "implicit-lifetime type", "fundamental alignment", "extended
  alignment").
- Allowed access modes enumerated explicitly (`read` or `read_write`), not
  `!= write`, which admits deprecated modes. `id<Dimensions>` rather than a
  linearized index for multi-dimensional accessors.

## Pass 5 — Normative wording

- Clause order per API: `_Preconditions:_`, `_Constraints:_`, `_Effects:_`,
  `_Returns:_`, `_Throws:_` (+ `_Remarks:_`, `_Mandates:_`, `_Complexity:_`),
  the same order everywhere. Is each clause the right *kind*?
  * A precondition is an unchecked requirement — violating it is UB by
    definition, so "violation results in undefined behavior" and "optionally
    diagnosed" must be deleted.
  * A diagnosed condition is a `_Throws:_`, not a precondition.
  * A template-argument requirement is a `_Constraint:_`, written positively.
  * `_Returns:_ Nothing` on a `void` function, and clauses that state nothing
    observable, should be deleted.
- "must" for requirements, "may" for implementation freedom; weak words removed.
- Ambiguity and verbosity: flag any sentence that can be understood in more
  than one way (e.g. an unclear "it", or a condition whose scope is unclear) and
  propose a rewrite that can only be understood one way; flag prose that
  restates the synopsis or the core spec, repeats itself, or is longer than
  needed, and propose the shorter text.
- **Hint vs assertion**: does the description promise only performance, or does
  it require the application to guarantee something? If the latter, it is an
  assertion — the wording must say "This property is an assertion by the
  application that ... The behavior is undefined if ...", and the word "hint"
  must not appear in the name, description or section title.
- Implementation detail in normative text? Anything about low-level events, when
  a copy happens, whether scratch memory is used, which backend call is made,
  belongs in a non-normative note or "Implementation notes". Ask: "if we change
  the implementation, does this sentence force us to break applications?"
- Errors: is the failure detected as early as possible, at the call that causes
  it? Does `what()` identify the offender? Is there an `errc` for every
  exception, and is it an existing one? Is a *new* error code really needed, or
  will `errc::invalid` plus a `what()` string do? Never both "throws" and
  "undefined behavior" for the same condition.
  * Prefer an exception over undefined behavior where the implementation can
    reasonably detect the problem. Where it can't, specify undefined behavior
    and tighten it later.
  * If a check is expensive for some users, add a property that disables it
    rather than weakening the diagnostic for everyone.
  * Existing codes to prefer: `errc::invalid`, `errc::feature_not_supported`,
    `errc::kernel_not_supported`, `errc::memory_allocation`, `errc::runtime`
    (unmapped native failures).
  * "must issue a diagnostic" implies compile time. For a runtime failure write
    "throws a synchronous `exception` with the `errc::X` error code from the
    kernel invocation command (e.g. `parallel_for`)". No `_Throws:_` on a device
    function that cannot throw.
- Completeness: every affected API covered (including `get_wait_list()`,
  `get_profiling_info()`, destructors); property incompatibilities and their
  errors; defaults ("equivalent to one of the values above, but the choice is
  implementation-defined"); host-vs-device availability; cross-TU /
  `SYCL_EXTERNAL` rules; value/default initialization; "when the last copy of X
  is destroyed" for reference-counted classes. Attributes across TUs follow the
  core rules: the first declaration in a TU must carry it, redeclarations may,
  and all TUs must agree.
- Over-specification: exact diagnostic text; enumerator values (especially for
  an enum modelling a hierarchy); the underlying type of an `enum class`;
  preconditions nobody will enforce; guarantees stronger than the layer beneath
  (Level Zero / SPIR-V / OpenCL) provides.
- Terms: defined only if used, active voice, with an example, and not a synonym
  for something the core spec already names. Any vague normative term
  ("topologically identical", "active control flow", "same command") must be
  defined exhaustively or replaced with something checkable.
- Overview: does it motivate, or does it just list the API / explain how to use
  it? Is anything normative hiding in it? Advanced performance knobs should carry
  the caveat: "These properties are intended for advanced performance tuning ...
  applications should generally rely on the compiler's default selection unless
  profiling demonstrates a benefit."
- Internal contradictions — a precondition that contradicts a definition, an
  example that uses a feature the synopsis doesn't have, a restriction stated in
  two places with different content. Writing every API's clauses explicitly
  exposes contradictions that prose paragraphs hide.

## Pass 6 — Relationship to the core spec and other extensions

- Does the extension try to deprecate or modify core SYCL? It may not — it can
  only *add* (with the `ext_oneapi_` prefix on core classes).
- Is it worded as a diff against sections/tables of the core spec, or does it
  speculate about a future core spec? Both should be removed.
- Does it restate rules the core spec already gives (USM/context accessibility,
  device-code type rules, kernel-argument rules)? Replace with a reference.
- Does it document interactions with other extensions, especially where it
  *restricts* existing behavior? "Undefined in combination with other extensions"
  is never acceptable. Mutual exclusivity must be stated in both extensions.
  Only the limitations need listing, not every compatible extension.
- Duplicating a *small* API in two extensions is acceptable (wording must be
  identical); once the shared surface grows, factor it into a common extension.
- Is it trying to be an "extension to an extension"? That should be a new
  version of the existing extension instead. Is a new extension usable on its
  own? One that only defines, say, a type trait isn't; put such pieces in the
  extension that needs them.
- Does the spec match what will actually be implemented, or is it aspirational?
  Aspirational parts belong in "Future direction".
- If an unimplemented feature blocks promotion to `supported`, propose removing
  it now and adding it back in version 2.

## Pass 7 — API design

### Rules

- Prefer one class per operation over overloads that are easy to confuse
  silently; real types over tag types; template parameters on the class when
  that keeps declarations in sync; the most commonly overridden template
  parameters first. Don't default a template parameter deduced from a function
  argument. State when CTAD is guaranteed ("All constructors for
  `launch_config` are guaranteed to deduce the class template parameters").
- Don't add narrow convenience functions a generic mechanism already covers, or
  a free function longer than the member function it wraps.
- Mirror the standard library's names, defaults, promotion rules and overloads.
  Follow SYCL: `copy(src, dest)` with element counts, `memcpy(dest, src)` with
  byte counts.
- Use strong/named types where adjacent scalar parameters are easy to swap
  (`src_stride{2}` / `dest_stride{2}`). Classes that own memory follow SYCL
  common reference semantics rather than deleting copies, and never throw from
  a destructor.
- Don't extend a class the core spec removed; define a new one.
- New group types implement the whole group concept
  (`get_group_id`/`get_group_linear_id`/`get_group_range`/
  `get_group_linear_range`/`get_local_id`/`get_local_range`/
  `get_local_linear_id`/`get_local_linear_range`, `leader()`, `fence_scope`,
  the `is_group`/`is_user_constructed_group` traits), usually plus a new
  `memory_scope` enumerator; `sycl_ext_oneapi_non_uniform_groups` is the model.
  State whether the functions are group functions (core §4.17.3) and therefore
  require convergent control flow and uniform arguments.
- Type traits: "an implementation may add specializations for
  implementation-defined types; users must not add their own".

### Properties

- Two models. New extensions use `sycl_khr_properties` (`sycl::khr::properties`;
  draft spec https://github.com/KhronosGroup/SYCL-Docs/pull/980, implemented in
  `sycl/include/sycl/khr/properties.hpp`). Keep `sycl_ext_oneapi_properties`
  only when updating an extension that already uses it. Never mix the models in
  one extension; neither container accepts the other's properties.
- KHR model:
  * List `sycl_khr_properties` in Dependencies. It is a draft and DPC++ builds
    it only with `__DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS`, so the extension
    cannot be `supported` until the KHR is.
  * Property `foo` has its key in a nested namespace with the same name,
    `property::key::foo` (no `_key` suffix). The namespace for vendor-extension
    properties is not defined yet: don't invent one; raise it in the PR and
    record it in "Issues".
  * Runtime values: a struct with a `constexpr` constructor and public members.
    Compile-time values: a variable template of unspecified type with
    `static constexpr` members (`using value_t` for a type value). Both: a class
    template. A single value is named `value` (`value_t` for a type).
  * Boolean flags carry a boolean value (`enable_x(bool v = true)`,
    `enable_x<bool>`) rather than relying on presence, so the list has the same
    type either way. A small enum gets a variable per enumerator plus the
    variable template.
  * Each entry gives the synopsis including `namespace key { struct foo; }`, the
    effect for each value, the four traits that inherit from `std::true_type`
    (`is_property`, `is_property_key`, `is_property_for<foo, C>`,
    `is_property_key_for<key::foo, C>`), and whether it is trivially copyable.
    `C` is the class constructed with the property; for any other use define a
    tag type that is declared but never defined
    (`khr::property::tag::reduction`).
  * An API that accepts properties takes `PropertyOrList props` (a template
    parameter) with "_Constraints:_ `khr::is_property_for_v<PropertyOrList, C>`
    is `true`, or `khr::is_property_list_for_v<PropertyOrList, C>` is `true`". A
    single property is treated as `khr::properties{props}`.
- oneAPI model: follow the pattern in "Templates". Properties go directly in
  `...::experimental` (not under `property::`). Vendor-specific properties live
  in the vendor namespace, but specializations of the property type traits must
  be in `sycl::ext::oneapi::experimental`. Restrict `is_property_key_of` to the
  intended classes. Document which properties each API accepts and constrain the
  property list: "_Constraints:_ `PropertyListT` is one of the properties listed
  below ...; or `is_property_list_v<PropertyListT>` is true and contains no
  properties other than those listed below ...".
- Both models:
  * Give an enum only the values supported today ("other values may be added in
    the future"). Prefer an integer parameter over an enum when future hardware
    may add values. No default value on a property that only exists to request
    non-default behavior; a separate `_automatic` property rather than a
    sentinel. Give a boolean property constructor a sensible default
    (`enable_ipc(bool enable = true)`).
  * A sub-parameter meaningful only inside one property is not a property of
    its own: make it another value of that property (KHR) or a small wrapper
    class with a `static constexpr value` (oneAPI).
  * State normatively whether a property is a kernel (compile-time) or launch
    property, and where it may not be used. Behavior on devices where it is
    meaningless goes in normative text, not a NOTE in the Status section.
  * Never define a property in terms of a specific backend extension; word it
    in terms of device capabilities.

### Queries and aspects

- `aspect` for a boolean capability, information descriptor for a richer answer.
  One aspect per scope/mode where support may differ. Never add an aspect that
  means "the info descriptor is supported".
- Throw `errc::feature_not_supported` from a query gated by an aspect, as a
  guardrail for applications that forgot to check it. Consider pairing a
  set-returning descriptor with `supports_X()` defined as "Equivalent to
  `...contains(name)`".
- Query the highest-level user-reachable concept (`device::can_compile(lang)`,
  not something taking a `backend`). Keep architecture selection
  (`if_architecture_is`) separate from aspects (`if_device_has`).
- Add new enumerators before any alias enumerators, and record in the table
  which extension version introduced each entry.
- Every optional device feature needs an `aspect`.

### Questions to ask

These questions recur across these reviews. When writing, make sure the spec
answers the ones that apply. When reviewing, raise only the ones that apply to
the change under review, and only where the spec doesn't already answer them —
never paste the list:

- Is this required or merely allowed? What happens if the user does it anyway?
- Which translation unit must carry the decoration? Must the implementation do
  cross-TU analysis? Is a diagnostic required?
- Is this implementation-defined, unspecified, or undefined behavior? Which did
  you mean?
- Do we have a use case? (Adding API later is easy; removing it isn't.)
- How will a user discover what is supported — is there an aspect or a query?
- Can we word this so we can extend it later without breaking code?
- What happens on a device/backend where this is meaningless — silently ignored,
  or an optional kernel feature that throws on submit?
- Is this an assertion or a hint?
- Could misuse be a compile error instead of silent misbehavior? (Distinct type
  instead of a mode on an existing class; `_Constraints:_` instead of
  `_Preconditions:_` where a trait makes it possible.)
- Would a sentinel enumerator (`architecture::unknown`) be friendlier than
  throwing from a query?
- Does the result of this query have two different meanings?
- Would separate overloads be clearer than a parameter pack or a wall of
  constraints — and what will the error message look like when the user gets it
  wrong?
- Is `sycl::foo(Group g, ...)` the right shape? That signature implies a
  cooperative operation.
- For a new group type: does it implement the full group concept, and does it
  need a new `memory_scope` enumerator?
- Should this be a member function or a free function? Does the standard library
  have a precedent (`std::real`) we should follow?
- Are the property patterns right (see "Templates" and "Properties")?
- Compile-time or runtime property — and is a "runtime kernel property" really a
  launch property?

## Pass 8 — Examples

- No `using namespace`; namespace aliases (`syclex`, `intelex`) defined at the
  top and every identifier fully qualified.
- Would it plausibly compile? `#include <sycl/sycl.hpp>`, code inside
  `int main()`, every identifier defined, no ill-formed C++ (bare `return;` in
  `main`, uninitialized pointer dereference), decoration ordering as the compiler
  actually accepts, C++17 baseline respected.
- Accidental semantics: `2.71` pulling in `fp64`; units not obvious
  (`1024 * sizeof(int)`); all work-items writing the same element; hard-coded
  index where the prose says a choice is computed; library calls that device code
  may not make (`std::launder`).
- Is it a real usage example, or a lightly edited synopsis? (The latter is not an
  example.)
- Does it verbosely reimplement something SYCL provides (a loop over devices
  instead of `platform::has(aspect)`)?
- Is any non-obvious C++ the example relies on explained?

## Reporting

- One thread per issue, anchored on the offending line, with proposed
  replacement text.
- Separate "must fix before merge" from bikeshedding, and say which is which.
- If a discussion is going long, propose recording it as an entry in the "Issues"
  section (marked `[UNRESOLVED]`) and merging, rather than blocking — but only if
  the open question doesn't change the API that users will write against. Keep
  resolved issues in the document, marked `[RESOLVED]` with the rationale, so the
  question isn't reopened later.
- Close each thread with an explicit resolution ("resolving because we chose
  another design"), and reference the commit that addressed it.
