// RUN: %clangxx %s -o %t
// RUN: env LLVM_BIN_PATH=%llvm_build_bin_dir %python %sycl_tools_src_dir/abi_check.py --mode check_symbols --reference %S/abi_check_positive_dump.txt %t
// REQUIRES: linux

__attribute__((weak)) void foo() {}

int main() {}
