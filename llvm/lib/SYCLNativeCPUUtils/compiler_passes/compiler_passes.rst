Compiler passes
===============

Introduction
------------

Files under this directory are integrated from the `oneAPI Construction Kit`_
using `git-filter-repo`. They are used by Native CPU to help create a pipeline for
turning a base kernel into something which can be executed across multiple work
items, including auto-vectorization.

These files are largely from the sub-directories
**modules/compiler/compiler_pipeline**, **modules/compiler/vecz** and
**modules/compiler/multi_llvm**. Only files that are used have been integrated
and the **CMake** files have been updated to fit in with LLVM components.

These sub-directories are used as follows:

* **compiler_pipeline** provides the passes to build a pipeline from the initial
  kernel, including generating working item loops, handling local memory,
  handling metadata and calling the vectorizer **vecz**.

* **vecz** provides a full function vectorizer, which generates a copy of the
  original function but vectorized across the work group, taking into account
  subgroups.

* **multi_llvm**. This provides some support for these functions to work across
  multiple LLVM versions. Although this is not strictly needed in LLVM, it has
  been integrated to allow the integration to go smoothly, without changing files
  directly. Note this is header only and exists under
  **compiler_pipeline/include/multi_llvm**.

**compiler_pipeline** and **vecz** will be documented under `sycl/docs`. Note
that there are several limitations in the code that are a result of the initial
integration. These should be addressed over time for maintainability reasons,
they are not necessary for correctness or performance reasons.

General limitations
-------------------

To simplify the integration and reduce risk, most of the files were integrated
with no changes at all. This means there are currently the following limitations:

* The namespace in **compiler_pipeline** is **compiler/utils**, the namespace in
  multi_llvm is **multi_llvm** and the namespace in **vecz** is **vecz**. These should
  be updated to reflect being under **LLVM**.
* include files should ideally be moved to under **llvm/include** but remain under
  these directories after the integration.
* **vecz** has a test tool **veczc** and associated **lit** tests. This tool if
  required should be moved under **llvm/tools** or **llvm/test**. This is also
  requires `NATIVE_CPU_BUILD_VECZ_TEST_TOOLS` **CMake** option to build. This can be
  run using the target `check-sycl-vecz`.
* **compiler_pipeline** has lit tests for the passes which have not been integrated.
  This is because they use a tool **muxc**, but these passes should be
  able to be tested using **opt**. These lit tests can be found in the
  `pipeline pass tests`_.
* There are many integrated files that are unlikely to have any code coverage but because
  there are referred to in other files which we do need, they exist here. These
  should be pruned over time as a better understanding is made of what is
  essential.

.. _oneAPI Construction Kit: https://github.com/uxlfoundation/oneapi-construction-kit
.. _pipeline pass tests: https://github.com/uxlfoundation/oneapi-construction-kit/tree/main/modules/compiler/test/lit/passes
