// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o -

// Test checks for that no compile errors occur for
// builtin async_work_group_copy
#include <sycl/sycl.hpp>

using namespace cl::sycl;

// Define the number of work items to enqueue.
const size_t NElems = 32;
const size_t WorkGroupSize = 8;
const size_t NWorkGroups = NElems / WorkGroupSize;

template <typename T> void async_work_group_test() {
  queue Q;

  buffer<T, 1> InBuf(NElems);
  buffer<T, 1> OutBuf(NElems);

  Q.submit([&](handler &CGH) {
     auto In = InBuf.template get_access<access::mode::read>(CGH);
     auto Out = OutBuf.template get_access<access::mode::write>(CGH);
     accessor<T, 1, access::mode::read_write, access::target::local> Local(
         range<1>{WorkGroupSize}, CGH);

     nd_range<1> NDR{range<1>(NElems), range<1>(WorkGroupSize)};
     CGH.parallel_for(NDR, [=](nd_item<1> NDId) {
       auto GrId = NDId.get_group_linear_id();
       auto Group = NDId.get_group();
       size_t Offset = GrId * WorkGroupSize;
       auto E = NDId.async_work_group_copy(
           Local.get_pointer(), In.get_pointer() + Offset, WorkGroupSize);
       E = NDId.async_work_group_copy(Out.get_pointer() + Offset,
                                      Local.get_pointer(), WorkGroupSize);
     });
   }).wait();
}

template <typename T> void test() {
  async_work_group_test<T>();
  async_work_group_test<vec<T, 2>>();
  async_work_group_test<vec<T, 3>>();
  async_work_group_test<vec<T, 4>>();
  async_work_group_test<vec<T, 8>>();
  async_work_group_test<vec<T, 16>>();
  async_work_group_test<detail::make_unsigned_t<T>>();
  async_work_group_test<vec<detail::make_unsigned_t<T>, 2>>();
  async_work_group_test<vec<detail::make_unsigned_t<T>, 3>>();
  async_work_group_test<vec<detail::make_unsigned_t<T>, 4>>();
  async_work_group_test<vec<detail::make_unsigned_t<T>, 8>>();
  async_work_group_test<vec<detail::make_unsigned_t<T>, 16>>();
}

int main() {
  test<int8_t>();
  test<int16_t>();
  test<int32_t>();
  test<int64_t>();
  test<cl::sycl::cl_half>();
  test<float>();
  test<double>();
  return 1;
}
