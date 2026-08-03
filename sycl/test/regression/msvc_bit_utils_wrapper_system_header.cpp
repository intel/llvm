// To properly get the STL headers on MSVC, the
// stl_wrappers/__msvc_bit_utils.hpp wrapper decorates std::__isa_available
// with sycl_global_var and qualifies with #pragma clang system_header. This
// test verifies that compilation proceeds correctly when the shim is
// included via non-system path. A failure will show
// "'sycl_global_var' attribute only supported within a system header".

//
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only \
// RUN:   -U_MSC_VER -D_MSC_VER=1900 -Wno-macro-redefined \
// RUN:   -include %sycl_include/sycl/stl_wrappers/__msvc_bit_utils.hpp \
// RUN:   -x c++ %s
