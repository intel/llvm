#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>

#include <sycl/sycl.hpp>

// This test checks that we pass the correct buffer size value when setting
// local_accessor as an argument through handler::set_arg to a kernel created
// using OpenCL interoperability methods.

namespace {

struct TestContext {
  size_t localBufferArgSize = 0;

  // SYCL RT has number of checks that all devices and contexts are consistent
  // between kernel, kernel_bundle and other objects.
  //
  // To ensure that those checks pass, we intercept some PI calls to extract
  // the exact PI handles of device and context used in queue creation to later
  // return them when program/context/kernel info is requested.
  pi_device deviceHandle;
  pi_context contextHandle;

  pi_program programHandle = createDummyHandle<pi_program>();

  ~TestContext() { releaseDummyHandle<pi_program>(programHandle); }
};

TestContext GlobalContext;

} // namespace

pi_result redefined_piKernelSetArg(pi_kernel kernel, pi_uint32 arg_index,
                                   size_t arg_size, const void *arg_value) {
  GlobalContext.localBufferArgSize = arg_size;

  return PI_SUCCESS;
}

pi_result after_piContextGetInfo(pi_context context, pi_context_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_CONTEXT_INFO_DEVICES:
    if (param_value)
      *static_cast<pi_device *>(param_value) = GlobalContext.deviceHandle;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(GlobalContext.deviceHandle);
    break;
  default:;
  }

  return PI_SUCCESS;
}

pi_result after_piProgramGetInfo(pi_program program, pi_program_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {

  switch (param_name) {
  case PI_PROGRAM_INFO_DEVICES:
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(GlobalContext.deviceHandle);
    if (param_value)
      *static_cast<pi_device *>(param_value) = GlobalContext.deviceHandle;
    break;
  default:;
  }

  return PI_SUCCESS;
}

pi_result redefined_piProgramGetBuildInfo(pi_program program, pi_device device,
                                          _pi_program_build_info param_name,
                                          size_t param_value_size,
                                          void *param_value,
                                          size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_PROGRAM_BUILD_INFO_BINARY_TYPE:
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_program_binary_type);
    if (param_value)
      *static_cast<pi_program_binary_type *>(param_value) =
          PI_PROGRAM_BINARY_TYPE_EXECUTABLE;
    break;
  default:;
  }

  return PI_SUCCESS;
}

pi_result after_piContextCreate(const pi_context_properties *properties,
                                pi_uint32 num_devices, const pi_device *devices,
                                void (*pfn_notify)(const char *errinfo,
                                                   const void *private_info,
                                                   size_t cb, void *user_data),
                                void *user_data, pi_context *ret_context) {
  if (ret_context)
    GlobalContext.contextHandle = *ret_context;
  GlobalContext.deviceHandle = *devices;
  return PI_SUCCESS;
}

pi_result after_piKernelGetInfo(pi_kernel kernel, pi_kernel_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_KERNEL_INFO_CONTEXT:
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(GlobalContext.contextHandle);
    if (param_value)
      *static_cast<pi_context *>(param_value) = GlobalContext.contextHandle;
    break;
  case PI_KERNEL_INFO_PROGRAM:
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(GlobalContext.programHandle);
    if (param_value)
      *(pi_program *)param_value = GlobalContext.programHandle;
    break;
  default:;
  }

  return PI_SUCCESS;
}

TEST(HandlerSetArg, LocalAccessor) {
  sycl::unittest::PiMock Mock;

  Mock.redefine<sycl::detail::PiApiKind::piKernelSetArg>(
      redefined_piKernelSetArg);
  Mock.redefineAfter<sycl::detail::PiApiKind::piContextCreate>(
      after_piContextCreate);
  Mock.redefineAfter<sycl::detail::PiApiKind::piProgramGetInfo>(
      after_piProgramGetInfo);
  Mock.redefineAfter<sycl::detail::PiApiKind::piContextGetInfo>(
      after_piContextGetInfo);
  Mock.redefineAfter<sycl::detail::PiApiKind::piKernelGetInfo>(
      after_piKernelGetInfo);
  Mock.redefine<sycl::detail::PiApiKind::piProgramGetBuildInfo>(
      redefined_piProgramGetBuildInfo);

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

  ASSERT_EQ(GlobalContext.localBufferArgSize, Size * sizeof(float));
}
