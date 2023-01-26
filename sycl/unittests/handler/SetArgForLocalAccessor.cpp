#include <gtest/gtest.h>
#include <helpers/KernelInteropCommon.hpp>
#include <helpers/PiMock.hpp>

#include <sycl/sycl.hpp>

// This test checks that we pass the correct buffer size value when setting
// local_accessor as an argument through handler::set_arg to a kernel created
// using OpenCL interoperability methods.

namespace {

size_t LocalBufferArgSize = 0;

pi_result redefined_piKernelSetArg(pi_kernel kernel, pi_uint32 arg_index,
                                   size_t arg_size, const void *arg_value) {
  LocalBufferArgSize = arg_size;

  return PI_SUCCESS;
}

TEST(HandlerSetArg, LocalAccessor) {
  sycl::unittest::PiMock Mock;
  redefineMockForKernelInterop(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piKernelSetArg>(
      redefined_piKernelSetArg);

  constexpr size_t Size = 128;
  sycl::queue Q;

  DummyHandleT handle;
  auto KernelCL = reinterpret_cast<typename sycl::backend_traits<
      sycl::backend::opencl>::template input_type<sycl::kernel>>(&handle);
  auto Kernel =
      sycl::make_kernel<sycl::backend::opencl>(KernelCL, Q.get_context());

  Q.submit([&](sycl::handler &CGH) {
     sycl::local_accessor<float, 1> Acc(Size, CGH);
     CGH.set_arg(0, Acc);
     CGH.single_task(Kernel);
   }).wait();

  ASSERT_EQ(LocalBufferArgSize, Size * sizeof(float));
}
} // namespace
