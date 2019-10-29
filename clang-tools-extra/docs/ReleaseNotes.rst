====================================================
Extra Clang Tools 10.0.0 (In-Progress) Release Notes
====================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

.. warning::

   These are in-progress notes for the upcoming Extra Clang Tools 10 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release 10.0.0. Here we describe the status of the Extra Clang Tools in
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

What's New in Extra Clang Tools 10.0.0?
=======================================

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

- :doc:`clang-doc <clang-doc>` now generates documentation in HTML format.

Improvements to clang-query
---------------------------

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- New :doc:`bugprone-dynamic-static-initializers
  <clang-tidy/checks/bugprone-dynamic-static-initializers>` check.

  Finds instances where variables with static storage are initialized
  dynamically in header files.

- New :doc:`bugprone-infinite-loop
  <clang-tidy/checks/bugprone-infinite-loop>` check.

  Finds obvious infinite loops (loops where the condition variable is not
  changed at all).

- New :doc:`bugprone-not-null-terminated-result
  <clang-tidy/checks/bugprone-not-null-terminated-result>` check

  Finds function calls where it is possible to cause a not null-terminated
  result. Usually the proper length of a string is ``strlen(str) + 1`` or equal
  length of this expression, because the null terminator needs an extra space.
  Without the null terminator it can result in undefined behaviour when the
  string is read.

- New :doc:`cppcoreguidelines-init-variables
  <clang-tidy/checks/cppcoreguidelines-init-variables>` check.

- New :doc:`darwin-dispatch-once-nonstatic
  <clang-tidy/checks/darwin-dispatch-once-nonstatic>` check.

  Finds declarations of ``dispatch_once_t`` variables without static or global
  storage.

- New :doc:`google-upgrade-googletest-case
  <clang-tidy/checks/google-upgrade-googletest-case>` check.

  Finds uses of deprecated Googletest APIs with names containing ``case`` and
  replaces them with equivalent APIs with ``suite``.

- New :doc:`linuxkernel-must-use-errs
  <clang-tidy/checks/linuxkernel-must-use-errs>` check.

  Checks Linux kernel code to see if it uses the results from the functions in
  ``linux/err.h``.

- New :doc:`llvm-prefer-register-over-unsigned
  <clang-tidy/checks/llvm-prefer-register-over-unsigned>` check.

  Finds historical use of ``unsigned`` to hold vregs and physregs and rewrites
  them to use ``Register``

- New :doc:`objc-missing-hash
  <clang-tidy/checks/objc-missing-hash>` check.

  Finds Objective-C implementations that implement ``-isEqual:`` without also
  appropriately implementing ``-hash``.

- Improved :doc:`bugprone-posix-return
  <clang-tidy/checks/bugprone-posix-return>` check.

  Now also checks if any calls to ``pthread_*`` functions expect negative return
  values.

- The 'objc-avoid-spinlock' check was renamed to :doc:`darwin-avoid-spinlock
  <clang-tidy/checks/darwin-avoid-spinlock>`

Improvements to include-fixer
-----------------------------

The improvements are...

Improvements to clang-include-fixer
-----------------------------------

The improvements are...

Improvements to modularize
--------------------------

The improvements are...

Improvements to pp-trace
------------------------

The improvements are...

Clang-tidy visual studio plugin
-------------------------------

The clang-tidy-vs plugin has been removed from clang, as
it's no longer maintained. Users should migrate to
`Clang Power Tools <https://marketplace.visualstudio.com/items?itemName=caphyon.ClangPowerTools>`_
instead.
