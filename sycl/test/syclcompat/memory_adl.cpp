// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -fsyntax-only
// Test that no syclcompat:: functions clash with global namespace fns due to ADL
#include <sycl/sycl.hpp>
#include <syclcompat/syclcompat.hpp>

int main(){
  syclcompat::device_info dummy_info;
  syclcompat::device_info dummy_info_2;
  memset(&dummy_info, 0, sizeof(syclcompat::device_info));
  memcpy(&dummy_info, &dummy_info_2, sizeof(syclcompat::device_info));
  free(&dummy_info);
}
