// RUN: env LLVM_BIN_PATH=%llvm_build_bin_dir python %sycl_tools_src_dir/abi_check.py --mode check_symbols --reference %S/pi_opencl_symbol_check.txt %sycl_libs_dir/libpi_opencl.so
// REQUIRES: linux
// expected-no-diagnostics
//
//===----------------------------------------------------------------------===//
// This test checks if there is any change in export symbols in libpi_opencl.so
//===----------------------------------------------------------------------===//
