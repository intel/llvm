// The stl_wrappers/__msvc_bit_utils.hpp shim provides a device-side
// definition of `std::__isa_available`. An earlier revision decorated that
// definition with `sycl_global_var` and gated the header with
// `#pragma clang system_header`. That combination did not survive a
// `-E` preprocess / recompile round-trip on MSVC targets: MSVC-style
// `#line` directives emitted by `-E` do not carry the system-header flag,
// so on the second parse the definition landed outside a system header and
// Sema rejected it with:
//   'sycl_global_var' attribute only supported within a system header
// This test verifies the round-trip works — preprocess to a `.ii`, then
// compile the `.ii`.
//
// RUN: %clangxx -fsycl -fsycl-device-only \
// RUN:   -U_MSC_VER -D_MSC_VER=1900 -Wno-macro-redefined \
// RUN:   -include %sycl_include/sycl/stl_wrappers/__msvc_bit_utils.hpp \
// RUN:   -E -x c++ %s -o %t.ii
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only \
// RUN:   -Wno-macro-redefined %t.ii
