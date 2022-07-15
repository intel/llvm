// RUN: %clangxx -fsycl %s -o %t.out
// The purpose of this test is to check that the following code can be
// successfully compiled

#include <CL/sycl.hpp>
#include <ext/intel/esimd.hpp>
using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

#define SIMD 16
#define THREAD_NUM 512
template <typename T0, typename T1, typename T2,
          class Sat = __ESIMD_NS::saturation_off_tag>
void test_rol(cl::sycl::queue q) {
  std::vector<T1> A(THREAD_NUM * SIMD, 0xFFFFFFFF);
  buffer<T1, 1> A_buf(A.data(), range<1>(A.size()));

  nd_range<1> Range((range<1>(THREAD_NUM)), (range<1>(16)));
  auto e = q.submit([&](handler &cgh) {
    auto A_acc = A_buf.template get_access<access::mode::read_write>(cgh);

    cgh.parallel_for(Range, [=](nd_item<1> it) SYCL_ESIMD_FUNCTION {
      T1 scalar_argument = 0xFFFFFFFF;
      Sat sat;
      T2 arg = 1;

      __ESIMD_NS::simd<T1, SIMD> A_load_vec;
      A_load_vec.copy_from(A_acc, 0);

      T0 scalar_result;
      __ESIMD_NS::simd<T0, SIMD> result;

      result = shl<T1>(A_load_vec, arg, sat);
      result = shr<T1>(A_load_vec, arg, sat);
      result = rol<T1>(A_load_vec, arg);
      result = ror<T1>(A_load_vec, arg);
      result = lsr<T1>(A_load_vec, arg, sat);
      result = asr<T1>(A_load_vec, arg, sat);

      scalar_result = shl<T1>(scalar_argument, arg, sat);
      scalar_result = shr<T1>(scalar_argument, arg, sat);
      scalar_result = rol<T1>(scalar_argument, arg);
      scalar_result = ror<T1>(scalar_argument, arg);
      scalar_result = lsr<T1>(scalar_argument, arg, sat);
      scalar_result = asr<T1>(scalar_argument, arg, sat);

      result.copy_to(A_acc, 0);
    });
  });
  e.wait();
}

int main(int argc, char *argv[]) {
  sycl::property_list properties{sycl::property::queue::enable_profiling()};
  auto q = sycl::queue(properties);

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();

  test_rol<uint32_t, uint32_t, int32_t>(q);
  test_rol<uint32_t, int32_t, int32_t>(q);
  test_rol<int32_t, uint32_t, int32_t>(q);
  test_rol<int32_t, int32_t, int32_t>(q);

  test_rol<uint32_t, uint32_t, int32_t, __ESIMD_NS::saturation_on_tag>(q);
  test_rol<uint32_t, int32_t, int32_t, __ESIMD_NS::saturation_on_tag>(q);
  test_rol<int32_t, uint32_t, int32_t, __ESIMD_NS::saturation_on_tag>(q);
  test_rol<int32_t, int32_t, int32_t, __ESIMD_NS::saturation_on_tag>(q);
  return 0;
}
