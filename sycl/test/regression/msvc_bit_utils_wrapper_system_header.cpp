// To properly get the STL headers on MSVC, the
// stl_wrappers/__msvc_bit_utils.hppp wrapper decorates std::__isa_available
// with sycl_global_var and qualifies with #pragma clang system_header.  This
// test verifies that that compilation proceeds correctly when the shim is
// included via a system or non-system path. A failure will see
// "'sycl_global_var' attribute only supported within a system header"

//
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -D_MSC_VER=1900 \
// RUN:   -include %sycl_include/sycl/stl_wrappers/__msvc_bit_utils.hpp \
// RUN:   -x c++ /dev/null
//
// Reproduce the original: preprocess with -E and recompile the .ii. Mirrors
// what --offload-new-driver does internally.
//
// RUN: %if system-windows %{ \
// RUN:   %clangxx -fsycl -fsycl-targets=spir64 --offload-new-driver -E %s -o %t.ii && \
// RUN:   %clangxx -fsycl -fsycl-targets=spir64 --offload-new-driver -fsyntax-only %t.ii \
// RUN: %}

#include <sycl/detail/core.hpp>

int main() { return 0; }
