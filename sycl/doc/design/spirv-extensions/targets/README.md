# Targets Registry

**Version: 0**

This directory contains a registry of enumerator values for compute device targets, features and architectures used in the SPV_INTEL_function_variants extension.
Other extensions can also adapt this registry

The authoritative reference file is `core.json`.
From this file the following files are generated using `generate.py`:
* `architectures.asciidoc` - to be used with the `OpSpecConditionalArchitectureINTEL` instruction
* `targets.asciidoc` - to be used with the `OpSpecConditionalTargetINTEL` instruction
* `registry.h` - contains all the values in a C header format

## Types of entries

There are two main types of entries, organized as unorded _sets_, or ordered _lists_.
In both cases, each entry has an associated integer value.

### Targets

Targets represent the device's Instruction Set Architecture (ISA), for example x86.
* The **Targets** _set_ contains the recognized targets.
* Each target has also a _set_ of **Features** which represent features/extensions of a particular target.
For example, AVX2 is an extension of the x86/x86_64 ISA and thus is available only for those targets.

### Architectures

Architectures represent the processor's model or microarchitecture.
* Architecture **categories** define a _set_ of device types, such as CPU, GPU, or other domain-specific accelerators (eg. an AI accelerator).
* Architecture **families** define a _set_ of distinct lines of products within each category. This generally means a vendor, but can also represent two types of devices of the same category within one vendor that are not directly comparable (eg. Vendor X has two lines of CPUs: high-performance CPUs for servers and low-power CPUs for embedded.).
* **Architecture** is an ordered _list_ of architectures.
Within the same category and family, it is meaningful to compare architecture to say, for example, arch. X larger than arch. Y.
Architectures with a larger enumerator value are evaluated as larger than those with smaller values.
The meaning of the ordering is defined by the vendor, but generally, newer architectures are added after older ones.

## Adding new entries & Versioning

Adding a new entry to one of the sets or lists is done by incrementing the enumerator and adding the entry at the end of the set / list.
Adding entries this way is backwards-compatible and does not require incrementing the version number.
Likewise, adding a new alias to an existing enumerator value is backwards-compatible.

Any changes to the meaning of existing enumerator values are backwards-incompatible and should be as rare as possible.
This includes, for example, removing an entry, adding an architecture somewhere else than at the end of the ordered list, or reordering the architectures.
In such cases, the version should be incremented, specifications using this repository will need to update the version, and finally the implementations of the specifications need to adapt to the changed version.

Version 0 is used for the initial draft and within this version breaking changes can occur.
