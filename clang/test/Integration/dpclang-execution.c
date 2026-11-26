// Exercise real frontend subprocesses and builtin-header discovery through the
// wrappers without supplying a resource directory or using system headers.
// RUN: %dpclang -target x86_64-unknown-unknown -canonical-prefixes -fno-integrated-cc1 -fsyntax-only -nostdlibinc %s
// RUN: %dpclang -target x86_64-unknown-unknown -no-canonical-prefixes -fno-integrated-cc1 -fsyntax-only -nostdlibinc %s
// RUN: %if clang++ %{ %dpclangxx -target x86_64-unknown-unknown -canonical-prefixes -fno-integrated-cc1 -fsyntax-only -nostdlibinc -x c++ %s %}
// RUN: %if clang++ %{ %dpclangxx -target x86_64-unknown-unknown -no-canonical-prefixes -fno-integrated-cc1 -fsyntax-only -nostdlibinc -x c++ %s %}

#include <stddef.h>
size_t size;
