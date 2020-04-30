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

TBD

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
* `test/abi/layout*` and `test/abi/symbol_size*` are a group of tests to check
  the internal layout of some classes. The layout tests check Clang AST for
  changes, while symbol_size check `sizeof` for objects. Changing the class
  layout is a breaking change.

## Breaking ABI

Whenever you need to change the existing ABI, please, follow these steps:

1. Adjust you PR description to reflect (non-)breaking ABI changes. Make sure
   it is clear, why breaking ABI is necessary.
2. Fix failing ABI tests in your Pull Request. Use aforementioned techniques to
   update test files.
3. Update the library version according to the policies.
