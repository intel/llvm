---
description: Reviews a SYCL extension specification (new or updated) the way the DPC++ specification reviewers do. Use when the user asks to review, critique or check a SYCL extension spec, an extension PR, or a draft of one.
---

## How to run a review

1. Get the diff and the *whole* file — most defects are in the interaction
   between the change and the rest of the document (a renamed identifier left
   stale elsewhere, a table row that no longer matches the synopsis).
2. Read the rendered document, not only the source. Asciidoc defects
   (cut-off tables, `C++` rendering as `C`, notes rendering as plain text) are
   invisible in the diff.
3. Compare against `sycl/doc/extensions/template.asciidoc`. Some of the old
   extensions are *not* a style reference.
4. Walk the passes below in order. Passes 1-4 are cheap and catch most review
   comments; passes 5-8 are where the design discussions live.
5. Write comments as concrete replacement wording (an Asciidoc ```suggestion```
   block or a block quote of the proposed paragraph), not just as a complaint —
   that is how these reviews are conducted and it is what makes them converge.

## Pass 1 — Structure and boilerplate

- Section order: Notice, Contact, Dependencies, **Status**, [Backend support
  status], Overview, Specification, [Implementation notes], Example(s),
  [Future direction], [Issues]. Dependencies before Status. "Usage Example",
  "Future Directions" and "Issues" must be *peers* of "Specification", not
  subsections. "Feature test macro" is the first subsection of Specification.
- Status text matches the template for the actual state, including the bold
  "*Shipping software products should not rely on APIs defined in this
  specification.*". No extra "IMPORTANT: this is a draft" line.
- "This extension is written against the SYCL 2020 revision N specification" —
  is N current? Are the referenced section numbers still right?
- Copyright = year of first publication, **not** bumped by this PR. No revision
  history table, no contributors list.
- Feature-test macro name is exactly the extension name in upper case. For an
  unversioned experimental extension the value table says "The APIs of this
  experimental extension are not versioned, so the feature-test macro always has
  this value."
- Extension name == title line == file name, all lower case, numbers included
  (`memcpy2d`). Backend-specific extension names include the backend.
- Location matches the state: `proposed/` if unimplemented, `experimental/`
  (and `...::experimental` namespaces) if implemented but unstable, `supported/`
  if stable. A "supported" extension may not depend on an experimental one, and
  may not break source compatibility.
- "Backend support status" section: present only if there is a real limitation,
  placed right after Status, and preferably expressed as a device aspect with
  `errc::kernel_not_supported` rather than as undefined behavior.
- New APIs added by a revision: is there a new row in the feature-test macro
  table, and an "Added in revision N of this specification." line under each new
  section?
- Lines wrapped at 80 columns; links relative (not absolute); every link still
  resolves after any file move.

## Pass 2 — Mechanical defects (fast, always productive)

- Unclosed or missing backticks; `Note:` vs `[_Note:_ ... _{endnote}_]`;
  `[source,c++]` (not `[source,dpcpp]`, not a bare triple-backtick block).
- `:cpp: pass:[C++]` present and `{cpp}` used; `+...+` escaping for
  `+__FILE__+`, `+__shared__+`, `+T...+`, `+<=+`, `+this->x()+`.
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
- Section titles lower case except the first word, phrased symmetrically, using
  core-spec terminology ("information descriptor", "property", "aspect",
  "command").

## Pass 3 — Naming

- Vendor string right? `oneapi` for generally-useful features, `intel` (or the
  vendor) for hardware-tied ones. Namespace matches vendor + experimental status,
  and backend-specific APIs have the backend level.
- Members added to **core** classes carry the `ext_<vendor>_` prefix; members of
  the extension's own classes do not. New core enumerators carry the prefix.
  `experimental` appears only in namespaces.
- Information descriptors are in the extension namespace
  (`ext::oneapi::experimental::info::device::foo`), *not*
  `info::device::ext_oneapi_foo` (deprecated), and the namespace order is
  `...::info::device`.
- Identifiers lower case including abbreviations; no acronyms; no abbreviations
  where the full word fits.
- Does the name describe the concept and read well at the use site? Push back on
  names that describe a mechanism, repeat their enclosing scope, or read
  ambiguously (`wait_request_not_requested`, `read_write_mode_readwrite`,
  `build_log(log)`).
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
- Completeness: every affected API covered (including `get_wait_list()`,
  `get_profiling_info()`, destructors); property incompatibilities and their
  errors; defaults ("equivalent to one of the values above, but the choice is
  implementation-defined"); host-vs-device availability; cross-TU /
  `SYCL_EXTERNAL` rules; value/default initialization; "when the last copy of X
  is destroyed" for reference-counted classes.
- Over-specification: exact diagnostic text; enumerator values (especially for
  an enum modelling a hierarchy); the underlying type of an `enum class`;
  preconditions nobody will enforce; guarantees stronger than the layer beneath
  (Level Zero / SPIR-V / OpenCL) provides.
- Terms: defined only if used, active voice, with an example, and not a synonym
  for something the core spec already names. Any vague normative term
  ("topologically identical", "active control flow", "same command") must be
  pinned down or removed.
- Overview: does it motivate, or does it just list the API / explain how to use
  it? Is anything normative hiding in it? Advanced performance knobs should carry
  the "most applications should not need this" caveat.
- Internal contradictions — a precondition that contradicts a definition, an
  example that uses a feature the synopsis doesn't have, a restriction stated in
  two places with different content.

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
- Is it trying to be an "extension to an extension"? That should be a new
  revision of the existing extension instead. Is a new extension usable on its
  own?
- Does the spec match what will actually be implemented, or is it aspirational?
  Aspirational parts belong in "Future direction".
- If an unimplemented feature blocks promotion to `supported`, propose removing
  it now and adding it back in revision 2.

## Pass 7 — API design questions to ask

These are the questions that recur across hundreds of these reviews. Ask them
explicitly:

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
- Are the property patterns right: `_key` suffix on the trait, matching property
  variable, `_enum` suffix, shortcut per enum value, properties directly in
  `...::experimental` (not under `property::`), `is_property_key_of` restricted
  to the intended classes, no default value on a property that only exists to
  request non-default behavior, a separate `_automatic` property rather than a
  sentinel?
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
