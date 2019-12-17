Device library extensions
===================================

Device compiler that indicates support for a particular extension is
supposed to support *all* the corresponding functions.

cl_intel_devicelib_cassert
==========================

.. code:
   void __devicelib_assert_fail(__generic const char *expr,
                                __generic const char *file,
                                int32_t line,
                                __generic const char *func,
                                size_t gid0, size_t gid1, size_t gid2,
                                size_t lid0, size_t lid1, size_t lid2);
Semantic:
the function is called when an assertion expression `expr` is false,
and it indicates that a program does not execute as expected.
The function should print a message containing the information
provided in the arguments. In addition to that, the function is free
to terminate the current kernel invocation.

Arguments:

  - `expr` is a string representation of the assertion condition
  - `file` and `line` are the source code location of the assertion
  - `func` (optional, may be NULL)  name of the function containing the assertion
  - `gidX` current work-item global id
  - `lidX` current work-item local id

Example of a message:
.. code:
   foo.cpp:42: void foo(int): local id: [0,0,0], global id: [0,0,0] Assertion `buf[wiID] == 0 && "Invalid value"` failed.
