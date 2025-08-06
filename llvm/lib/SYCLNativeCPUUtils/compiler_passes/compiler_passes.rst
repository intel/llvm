Compiler passes
===============

Introduction
------------

Files under this directory are ported from the [oneAPI Construction
Kit](https://github.com/uxlfoundation/oneapi-construction-kit). They are used by
NativeCPU to help create a pipeline for turning a base kernel into something
which can be executed across multiple work items, including auto-vectorization.

These files are largely from the sub-directories
**modules/compiler/compiler_pipeline**, **modules/compiler/vecz** and
**modules/compiler/multi_llvm**. Only files that are used have been copied and
the **Cmake** files have been updated to fit in with LLVM components.

These sub-directories are used as follows:

* **compiler_pipeline** provides the passes to build a pipeline from the initial
  kernel, including generating working item loops, handling local memory,
  handling metadata and calling the vectorizer **vecz**.

* **vecz** provides a full function vectorizer, which generates a copy of the
  original function but vectorized across the work group, taking into account
  subgroups.

* **multi_llvm**. This provides some support for these functions to work across
  multiple LLVM versions. Although this is not strictly needed in LLVM, it has
  been ported across to allow the port to go smoothly, without changing files
  directly. Note this is header only and exists under
  **compiler_pipeline/include/multi_llvm**.

**compiler_pipeline** is documented in :doc:`/compiler_pipeline/docs/compiler`
and **vecz** is documented in :doc:`/vecz/vecz`. Note that there are several
limitations both in the documentation and in the code that are a result of the
initial port. These should be addressed.

General limitations
-------------------

There is the following limitations in the current port:

* The documentation makes a lot of references to **oneAPI Construction Kit**
constructs such as ComputeMux. It also references files that have not been
copied across.
* The namespace in **compiler_pipeline** is **compiler/utils**, the namespace in
  multi_llvm is **multi_llvm** and the namespace in **vecz** is **vecz**. These should
  be updated to reflect being under **LLVM**.
* include files should ideally be moved to under **llvm/include** but remain under
  these directories after the port.
* **vecz** has a test tool **veczc** and associated **lit** tests. This tool if
  required should be moved under **llvm/tools** or **llvm/test**. This is also not
  built by default. It also refers to **@CA_COMMON_LIT_BINARY_PATH@** which is not
  defined.
* **compiler_pipeline** has lit tests for the passes which have not been ported
  across. This is because they use a tool **muxc**, but these passes should be
  able to be tested using opt. These lit tests can be found
  [here](https://github.com/uxlfoundation/oneapi-construction-kit/tree/main/modules/compiler/test/lit/passes).
* There are many files that are unlikely to have any code coverage but because
  there are referred to in other files which we do need, they exist here. These
  should be pruned over time as a better understanding is made of what is
  required.

.. _oneAPI Construction Kit: https://github.com/uxlfoundation/oneapi-construction-kit