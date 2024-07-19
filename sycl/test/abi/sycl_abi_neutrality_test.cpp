// REQUIRES: linux
// UNSUPPORTED: libcxx
// RUN: sort %S/sycl_symbols_linux.dump | FileCheck %s --implicit-check-not=cxx11

// The purpose of this test is to check that all symbols that are visible from
// SYCL library are ABI neutral (see
// https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html). It
// means that SYCL library must not export symbols in "__cxx11" namespace or
// with "cxx11" tag because such symbols correspond to the new ABI entries
// (_GLIBCXX_USE_CXX11_ABI=1, default) and won't work with a program that uses
// the old ABI (_GLIBCXX_USE_CXX11_ABI=0). All APIs exported from SYCL RT must
// avoid using classes like std::string and std::list impacted by the dual ABI
// issue and have to use their ABI-neutral counterparts provided by SYCL RT (e.g
// sycl::detail::string, etc.).

// New exclusions are NOT ALLOWED to this file. All remaining cases that need
// to be fixed are listed below.
// CHECK:_ZN4sycl3_V13ext5intel12experimental15online_compilerILNS3_15source_languageE0EE7compileIJSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaISE_EEEEES8_IhSaIhEERKSE_DpRKT_
// CHECK:_ZN4sycl3_V13ext5intel12experimental15online_compilerILNS3_15source_languageE1EE7compileIJSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaISE_EEEEES8_IhSaIhEERKSE_DpRKT_
// CHECK:_ZN4sycl3_V13ext5intel12experimental9pipe_base13get_pipe_nameB5cxx11EPKv
// CHECK:_ZN4sycl3_V16detail6OSUtil10getDirNameB5cxx11EPKc
// CHECK:_ZN4sycl3_V16detail6OSUtil16getCurrentDSODirB5cxx11Ev
// CHECK:_ZN4sycl3_V16opencl13has_extensionERKNS0_6deviceERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
// CHECK:_ZN4sycl3_V16opencl13has_extensionERKNS0_8platformERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
// CHECK:_ZNK4sycl3_V13ext6oneapi12experimental6detail24modifiable_command_graph11print_graphENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb
// CHECK:_ZNK4sycl3_V16device21ext_oneapi_cl_profileB5cxx11Ev
