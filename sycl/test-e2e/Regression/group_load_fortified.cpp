// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Checks that group_load runs even when the source code is fortified. This
// failed at one point due to the use of std::memcpy in the implementation,
// which would hold an assert in device code when fortified, which would fail
// to JIT compile.

#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

int main(void) {
  sycl::queue Q;

  constexpr std::size_t N = 256;
  constexpr std::uint32_t LWS = 64;
  constexpr std::uint32_t VecSize = 4;
  constexpr std::size_t NGroups = (N + VecSize * LWS - 1) / (VecSize * LWS);

  int *Ptr = sycl::malloc_device<int>(N, Q);

  Q.submit([&](sycl::handler &CGH) {
     CGH.parallel_for(
         sycl::nd_range<1>{sycl::range<1>{NGroups * LWS}, sycl::range<1>{LWS}},
         [=](sycl::nd_item<1> It) {
           const std::size_t GID = It.get_global_id();
           const sycl::sub_group &SG = It.get_sub_group();

           constexpr auto Striped = syclexp::properties{
               syclexp::data_placement_striped, syclexp::full_group};

           auto MPtr = sycl::address_space_cast<
               sycl::access::address_space::global_space,
               sycl::access::decorated::yes>(Ptr);

           sycl::vec<int, VecSize> X{};
           syclexp::group_load(SG, MPtr, X, Striped);
         });
   }).wait();

  sycl::free(Ptr, Q);

  return 0;
}
