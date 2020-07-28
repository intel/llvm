// UNSUPPORTED: cuda
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <cassert>
#include <tuple>
using namespace sycl;
using namespace sycl::intel;

template <typename Group> class copy_kernel;

// Returns a tuple of:
// - The requested group
// - The size of the requested group
// - The global offset of the start of the group, for accessing global memory
// - The local offset of the start of the group, for accessing local memory
template <typename Group>
std::tuple<Group, size_t, size_t, size_t> get_copy_args(nd_item<1> it);

template <>
std::tuple<group<1>, size_t, size_t, size_t>
get_copy_args<group<1>>(nd_item<1> it) {
  return {it.get_group(), it.get_local_range()[0],
          it.get_group(0) * it.get_local_range()[0], 0};
}

template <>
std::tuple<sub_group, size_t, size_t, size_t>
get_copy_args<sub_group>(nd_item<1> it) {
  sub_group sg = it.get_sub_group();
  return {sg, sg.get_local_range()[0],
          it.get_group(0) * it.get_local_range()[0] +
              sg.get_group_id()[0] * sg.get_max_local_range()[0],
          sg.get_group_id()[0] * sg.get_max_local_range()[0]};
}

template <typename Group> void test(queue q, bool async) {

  constexpr size_t N = 32;
  constexpr size_t L = 16;
  std::array<int, N> in, out;
  std::iota(in.begin(), in.end(), 0);
  std::fill(out.begin(), out.end(), 0);
  {
    buffer<int> in_buf(in.data(), range<1>{N});
    buffer<int> out_buf(out.data(), range<1>{N});
    q.submit([&](handler &cgh) {
      auto tmp =
          accessor<int, 1, access::mode::read_write, access::target::local>(
              L, cgh);
      auto in = in_buf.get_access<access::mode::read_write>(cgh);
      auto out = out_buf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class copy_kernel<Group>>(
          nd_range<1>(N, L), [=](nd_item<1> it) {
        auto [g, size, goffset, loffset] = get_copy_args<Group>(it);
        if (async) {
          device_event e1 = async_copy_n(g, in.get_pointer() + goffset, size,
                                         tmp.get_pointer() + loffset);
          wait_for(g, e1);
          tmp[it.get_local_linear_id()] += 1;
          device_event e2 = async_copy_n(g, tmp.get_pointer() + loffset, size,
                                         out.get_pointer() + goffset);
          wait_for(g, e2);
        } else {
          copy_n(g, in.get_pointer() + goffset, size,
                 tmp.get_pointer() + loffset);
          tmp[it.get_local_linear_id()] += 1;
          copy_n(g, tmp.get_pointer() + loffset, size,
                 out.get_pointer() + goffset);
        }
          });
    });
  }

  // Each result should be one greater than before
  std::array<int, N> gold;
  std::iota(gold.begin(), gold.end(), 1);
  assert(std::equal(out.begin(), out.end(), gold.begin()));
}

int main() {
  queue q;
  test<group<1>>(q, true);
  test<group<1>>(q, false);
  test<sub_group>(q, true);
  test<sub_group>(q, false);
  std::cout << "Test passed." << std::endl;
}
