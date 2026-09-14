// Input for clang-linker-wrapper-syclbin-link.cpp: defines the SYCL_EXTERNAL
// function used in syclbin-link-a.cpp.
__attribute__((sycl_device)) int syclbin_link_foo(int X) { return X * 2; }
