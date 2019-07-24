===================================================
Extra Clang Tools 9.0.0 (In-Progress) Release Notes
===================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 9 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 9.0.0. Here we describe the status of the Extra Clang Tools in
some detail, including major improvements from the previous release and new
feature work. All LLVM releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or
the `LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Subversion checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Extra Clang Tools 9.0.0?
======================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

Major New Features
------------------

...

Improvements to clangd
----------------------

The improvements are...

Improvements to clang-doc
-------------------------

The improvements are...

Improvements to clang-query
---------------------------

- ...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- New OpenMP module.

  For checks specific to `OpenMP <https://www.openmp.org/>`_ API.

- New :doc:`abseil-duration-addition
  <clang-tidy/checks/abseil-duration-addition>` check.

  Checks for cases where addition should be performed in the ``absl::Time``
  domain.

- New :doc:`abseil-duration-conversion-cast
  <clang-tidy/checks/abseil-duration-conversion-cast>` check.

  Checks for casts of ``absl::Duration`` conversion functions, and recommends
  the right conversion function instead.

- New :doc:`abseil-duration-unnecessary-conversion
  <clang-tidy/checks/abseil-duration-unnecessary-conversion>` check.

  Finds and fixes cases where ``absl::Duration`` values are being converted to
  numeric types and back again.

- New :doc:`abseil-time-comparison
  <clang-tidy/checks/abseil-time-comparison>` check.

  Prefer comparisons in the ``absl::Time`` domain instead of the integer
  domain.

- New :doc:`abseil-time-subtraction
  <clang-tidy/checks/abseil-time-subtraction>` check.

  Finds and fixes ``absl::Time`` subtraction expressions to do subtraction
  in the Time domain instead of the numeric domain.

- New :doc:`android-cloexec-pipe
  <clang-tidy/checks/android-cloexec-pipe>` check.

  This check detects usage of ``pipe()``.

- New :doc:`android-cloexec-pipe2
  <clang-tidy/checks/android-cloexec-pipe2>` check.

  This checks ensures that ``pipe2()`` is called with the O_CLOEXEC flag.

- New :doc:`bugprone-unhandled-self-assignment
  <clang-tidy/checks/bugprone-unhandled-self-assignment>` check.

  Finds user-defined copy assignment operators which do not protect the code
  against self-assignment either by checking self-assignment explicitly or
  using the copy-and-swap or the copy-and-move method.

- New :doc:`bugprone-branch-clone
  <clang-tidy/checks/bugprone-branch-clone>` check.

  Checks for repeated branches in ``if/else if/else`` chains, consecutive
  repeated branches in ``switch`` statements and indentical true and false
  branches in conditional operators.

- New :doc:`bugprone-posix-return
  <clang-tidy/checks/bugprone-posix-return>` check.

  Checks if any calls to POSIX functions (except ``posix_openpt``) expect negative
  return values.

- New :doc:`fuchsia-default-arguments-calls
  <clang-tidy/checks/fuchsia-default-arguments-calls>` check.

  Warns if a function or method is called with default arguments.
  This was previously done by `fuchsia-default-arguments check`, which has been
  removed.

- New :doc:`fuchsia-default-arguments-calls
  <clang-tidy/checks/fuchsia-default-arguments-calls>` check.

  Warns if a function or method is declared with default parameters.
  This was previously done by `fuchsia-default-arguments check` check, which has
  been removed.

- New :doc:`google-readability-avoid-underscore-in-googletest-name
  <clang-tidy/checks/google-readability-avoid-underscore-in-googletest-name>`
  check.

  Checks whether there are underscores in googletest test and test case names in
  test macros, which is prohibited by the Googletest FAQ.

- New :doc:`google-objc-avoid-nsobject-new
  <clang-tidy/checks/google-objc-avoid-nsobject-new>` check.

  Checks for calls to ``+new`` or overrides of it, which are prohibited by the
  Google Objective-C style guide.

- New :doc:`objc-super-self <clang-tidy/checks/objc-super-self>` check.

  Finds invocations of ``-self`` on super instances in initializers of
  subclasses of ``NSObject`` and recommends calling a superclass initializer
  instead.

- New alias :doc:`cert-oop54-cpp
  <clang-tidy/checks/cert-oop54-cpp>` to
  :doc:`bugprone-unhandled-self-assignment
  <clang-tidy/checks/bugprone-unhandled-self-assignment>` was added.

- New alias :doc:`cppcoreguidelines-explicit-virtual-functions
  <clang-tidy/checks/cppcoreguidelines-explicit-virtual-functions>` to
  :doc:`modernize-use-override
  <clang-tidy/checks/modernize-use-override>` was added.

- The :doc:`bugprone-argument-comment
  <clang-tidy/checks/bugprone-argument-comment>` now supports
  `CommentBoolLiterals`, `CommentIntegerLiterals`, `CommentFloatLiterals`,
  `CommentUserDefiniedLiterals`, `CommentStringLiterals`,
  `CommentCharacterLiterals` & `CommentNullPtrs` options.

- The :doc:`bugprone-too-small-loop-variable
  <clang-tidy/checks/bugprone-too-small-loop-variable>` now supports
  `MagnitudeBitsUpperLimit` option. The default value was set to 16,
  which greatly reduces warnings related to loops which are unlikely to
  cause an actual functional bug.

- The ‘fuchsia-default-arguments’ check has been removed.

  Warnings of function or method calls and declarations with default arguments
  were moved to :doc:`fuchsia-default-arguments-calls
  <clang-tidy/checks/fuchsia-default-arguments-calls>` and
  :doc:`fuchsia-default-arguments-calls
  <clang-tidy/checks/fuchsia-default-arguments-calls>` checks respectively.

- The :doc:`google-runtime-int <clang-tidy/checks/google-runtime-int>`
  check has been disabled in Objective-C++.

- The `Acronyms` and `IncludeDefaultAcronyms` options for the
  :doc:`objc-property-declaration <clang-tidy/checks/objc-property-declaration>`
  check have been removed.

- The :doc:`modernize-use-override
  <clang-tidy/checks/modernize-use-override>` now supports `OverrideSpelling`
  and `FinalSpelling` options.

- New :doc:`llvm-prefer-isa-or-dyn-cast-in-conditionals
  <clang-tidy/checks/llvm-prefer-isa-or-dyn-cast-in-conditionals>` check.

  Looks at conditionals and finds and replaces cases of ``cast<>``,
  which will assert rather than return a null pointer, and
  ``dyn_cast<>`` where the return value is not captured. Additionally,
  finds and replaces cases that match the pattern ``var &&
  isa<X>(var)``, where ``var`` is evaluated twice.

- New :doc:`modernize-use-trailing-return-type
  <clang-tidy/checks/modernize-use-trailing-return-type>` check.

  Rewrites function signatures to use a trailing return type.

- The :doc:`misc-throw-by-value-catch-by-reference
  <clang-tidy/checks/misc-throw-by-value-catch-by-reference>` now supports
  `WarnOnLargeObject` and `MaxSize` options to warn on any large trivial
  object caught by value.

- Added `UseAssignment` option to :doc:`cppcoreguidelines-pro-type-member-init
  <clang-tidy/checks/cppcoreguidelines-pro-type-member-init>`

  If set to true, the check will provide fix-its with literal initializers
  (``int i = 0;``) instead of curly braces (``int i{};``).

- New :doc:`readability-convert-member-functions-to-static
  <clang-tidy/checks/readability-convert-member-functions-to-static>` check.

  Finds non-static member functions that can be made ``static``.

Improvements to include-fixer
-----------------------------

- New :doc:`openmp-exception-escape
  <clang-tidy/checks/openmp-exception-escape>` check.

  Analyzes OpenMP Structured Blocks and checks that no exception escapes
  out of the Structured Block it was thrown in.

- New :doc:`openmp-use-default-none
  <clang-tidy/checks/openmp-use-default-none>` check.

  Finds OpenMP directives that are allowed to contain a ``default`` clause,
  but either don't specify it or the clause is specified but with the kind
  other than ``none``, and suggests to use the ``default(none)`` clause.

Improvements to clang-include-fixer
-----------------------------------

The improvements are...

Improvements to modularize
--------------------------

The improvements are...

Improvements to pp-trace
------------------------

- Added a new option `-callbacks` to filter preprocessor callbacks. It replaces
  the `-ignore` option.
