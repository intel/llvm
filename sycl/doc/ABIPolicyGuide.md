# ABI Policy Guide

## Intro

Application Binary Interface is a contract between binary modules, that defines
how structures and routines are accessed in machine code. Changing the ABI may
break backwards compatibility of user application with the DPC++ runtime library
for user-developed applications, resulting in need to rebuild such applications.
The goal of this document is to provide guidelines for maintaining the current
ABI of the DPC++ runtime library and mechanisms of notifying users about ABI
changes.

All ABI changes can be divided into two large groups: breaking and non-breaking.
A breaking change means that the new binary is incompatible with the previous
version (i.e. it can not be used as a drop-in replacement). A non-breaking
change means that the forward compatibility is broken (i.e. the old library
can be replaced with newer version, but not vice versa).

The following non-exhaustive list contains changes that are considered to be
breaking:

1. Changing the size of exported symbol (for example, adding new member field
   to the exported class).
1. Removing the exported symbol (that includes both changing the signature of
   exported routine and removing it).
1. Changing the alignment of exported symbol.
1. Changing the layout of exported symbol (for example, reordering class field
   members).
1. Adding or removing base classes.

Adding a new exported symbol is considered to be non-breaking change.

## ABI Versioning Policy

The release version of the DPC++ runtime library follows
[Semantic Versioning](https://semver.org/) scheme: `MAJOR.MINOR.PATCH`. `MAJOR`
version indicates breaking change. Version `X` is backwards incompatible with
version `X-1`. `MINOR` indicates a non-breaking change. The development version
of the library has a postfix `-V` that indicates breaking changes between
releases. Every time a pull request introduces a breaking change, it must also
uplift `V`. It is pull request author responsibility to accordingly update
`V` part of library version. If `V > 0` on release date, then `MAJOR_VERSION`
is uplifted and `V` is set to zero. `MAJOR`, `MINOR`, and `PATCH` versions are
not updated between releases.

## `__SYCL_EXPORT` Macro

The `__SYCL_EXPORT` provides facilities for fine-grained control over exported
symbols. Mark symbols that are supposed to be accessible by the user and that
are implemented in the SYCL Runtime library with this macro. Template
specializations also must be explicitly marked with `__SYCL_EXPORT` macro.
Symbols not marked `__SYCL_EXPORT` have internal linkage.

A few examples of when it is necessary to mark symbols with the macro:

* The `device` class:
  - It is defined as API by the SYCL spec.
  - It is implemented in `device.cpp` file.
* The `SYCLMemObjT` class:
  - It is not defined in the SYCL spec, but it is an implementation detail that
    is accessible by the user (buffer and image inherit from this class).
  - It has symbols that are implemented in the Runtime library.

When it is not necessary to mark symbols with `__SYCL_EXPORT`:
* The `buffer` class:
  - It is defined by the SYCL spec, but it is fully implemented in the headers.
* The `ProgramManager` class:
  - It is an implementation detail.
  - It is not accessed from the header files that are available to users.

## Automated ABI Changes Testing

> The automated tests deal with the most commonly occurring problems, but they
> may not catch some corner cases. If you believe your PR breaks ABI, but the
> test does not indicate that, please, notify the reviewers.

There is a set of tests to help identifying ABI changes:

* `test/abi/sycl_symbols_*.dump` contains dump of publicly available symbols.
  If you add a new symbol, it is considered non-breaking change. When the test
  reports missing symbols, it means you have either changed or remove some of
  existing API methods. In both cases you need to adjust the dump file. You
  can do it either manually, or by invoking the following command:
  ```shell
  python3 sycl/tools/abi_check.py --mode dump_symbols --output path/to/output.dump path/to/sycl.so(.dll)
  ```
  Please, prefer updating the test files with the above command. The checker
  script automatically sorts symbols. This would allow developers to avoid
  large diffs and help maintainers identify the nature of ABI changes.
* `test/abi/layout*` and `test/abi/symbol_size_alignment.cpp` are a group of 
  tests to check the internal layout of some classes. The `layout*` tests check 
  some of API classes for layout changes, while `symbol_size_alignment` only 
  checks `sizeof` and `alignof` for API classes. Changing the class layout is a 
  breaking change.

## Changing ABI

**Note (October, 2020)**: DPC++ runtime and compiler ABI is currently in frozen
state. This means that no ABI-breaking changes will be accepted by default.
Project maintainers may still approve breaking changes in some cases. Please,
try to avoid any breaking changes until freeze is lifted. If you need to change
existing functionality, consider adding new APIs instead of replacing them.
Also, please, avoid any changes, mentioned in the [Intro](#intro) section as
breaking. Refer to the above guide to distinguish breaking and non-breaking
changes. If not sure, do not hesitate to ask code owners for help.

Whenever you need to change the existing ABI, please, follow these steps:

1. Adjust you PR description to reflect (non-)breaking ABI changes. Make sure
   it is clear, why breaking ABI is necessary.
2. Fix failing ABI tests in your Pull Request. Use aforementioned techniques to
   update test files.
3. ~~If Pull Request introduces a breaking change, update the library version~~
   ~~according to the policies.~~ **(See note above)**

## BKMs on avoiding changing ABI

1. If there is a need to add a new field in `sycl::handler` or/and
   `sycl::detail::CG` classes it can be done without breaking ABI using the
   approach described in the comment at the beggining of
   [cg.hpp](../include/CL/sycl/detail/cg.hpp)
