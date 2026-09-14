// Input for clang-linker-wrapper-syclbin-link.cpp: uses a SYCL_EXTERNAL
// function that is defined in syclbin-link-b.cpp.
__attribute__((sycl_device)) int syclbin_link_foo(int X);

__attribute__((sycl_device)) int syclbin_link_bar(int X) {
  return syclbin_link_foo(X) + 1;
}
