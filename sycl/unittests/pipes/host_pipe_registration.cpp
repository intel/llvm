// //==-------------- host_pipe_registration.cpp - Host pipe tests------------==//
// //
// // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// // See https://llvm.org/LICENSE.txt for license information.
// // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// //
// //===----------------------------------------------------------------------===//

// #include <cstring>
// #include <sycl/sycl.hpp>

// #include <gtest/gtest.h>
// #include <helpers/PiImage.hpp>
// #include <helpers/PiMock.hpp>

// template <size_t KernelSize = 1> class TestKernel;

// namespace sycl {
// __SYCL_INLINE_VER_NAMESPACE(_V1) {
// namespace detail {
// template <size_t KernelSize> struct KernelInfo<TestKernel<KernelSize>> {
//   static constexpr unsigned getNumParams() { return 0; }
//   static const kernel_param_desc_t &getParamDesc(int) {
//     static kernel_param_desc_t Dummy;
//     return Dummy;
//   }
//   static constexpr const char *getName() { return "TestKernel"; }
//   static constexpr bool isESIMD() { return false; }
//   static constexpr bool callsThisItem() { return false; }
//   static constexpr bool callsAnyThisFreeFunction() { return false; }
//   static constexpr int64_t getKernelSize() { return KernelSize; }
// };

// } // namespace detail
// } // __SYCL_INLINE_VER_NAMESPACE(_V1)
// } // namespace sycl

// static sycl::unittest::PiImage generateDefaultImage() {
//   using namespace sycl::unittest;

//   PiPropertySet PropSet;

//   std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

//   PiArray<PiOffloadEntry> Entries = makeEmptyKernels({"TestKernel"});

//   PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
//               __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
//               "",                                     // Compile options
//               "",                                     // Link options
//               std::move(Bin),
//               std::move(Entries),
//               std::move(PropSet)};

//   return Img;
// }

// using namespace sycl;
// using default_pipe_properties = decltype(sycl::ext::oneapi::experimental::properties(sycl::ext::intel::experimental::uses_valid<true>));

// class PipeID;
// using Pipe = sycl::ext::intel::experimental::pipe<PipeID, int, 10, default_pipe_properties>;

// pi_event READ = reinterpret_cast<pi_event>(0);
// pi_event WRITE = reinterpret_cast<pi_event>(1);
// static constexpr int PipeReadVal = 8;
// static int PipeWriteVal = 0;
// pi_result redefinedEnqueueReadHostPipe(pi_queue, pi_program, const char *,
//                                        pi_bool, void *ptr, size_t, pi_uint32,
//                                        const pi_event *, pi_event *event) {
//   *(((test_data_type *)ptr)) = PipeReadVal;
//   *event = READ;
//   return PI_SUCCESS;
// }
// pi_result redefinedEnqueueWriteHostPipe(pi_queue, pi_program, const char *,
//                                         pi_bool, void *ptr, size_t, pi_uint32,
//                                         const pi_event *, pi_event *event) {
//   PipeWriteVal = 9;
//   *event = WRITE;
//   return PI_SUCCESS;
// }

// void preparePiMock(unittest::PiMock &Mock) {
//   Mock.redefine<detail::PiApiKind::piextEnqueueReadHostPipe>(
//       redefinedEnqueueReadHostPipe);
//   Mock.redefine<detail::PiApiKind::piextEnqueueWriteHostPipe>(
//       redefinedEnqueueWriteHostPipe);
// }

// class PipeTest : public ::testing::Test {
// public:
//   PipeTest() : Mock{}, Plt{Mock.getPlatform()} {}

// protected:
//   void SetUp() override {
//     std::clog << "Zibai started the setup()\n";
//     preparePiMock(Mock);
//     const sycl::device Dev = Plt.get_devices()[0];
//     sycl::context Ctx{Dev};
//     sycl::queue Q{Ctx, Dev};
//     ctx = Ctx;
//     q = Q;
//   }

// protected:
//   unittest::PiMock Mock;
//   sycl::platform Plt;
//   context ctx;
//   queue q;
// };

// TEST_F(PipeTest, Basic) {

//   // Fake registration of host pipes
//   sycl::detail::host_pipe_map::add(Pipe::get_host_ptr(),
//                                    "test_host_pipe_unique_id");

//   // Device registration
//   std::clog << "Zibai started the Device registration\n";
//   static sycl::unittest::PiImage Img = generateDefaultImage();
//   static sycl::unittest::PiImageArray<1> ImgArray{&Img};

//   std::clog << "Zibai started the get_host_ptr\n";
//   const void *HostPipePtr = Pipe::get_host_ptr();
//   std::clog << "Zibai started the hostPipeEntry\n";
//   detail::HostPipeMapEntry *hostPipeEntry =
//       detail::ProgramManager::getInstance().getHostPipeEntry(HostPipePtr);
//   const std::string pipe_name = hostPipeEntry->MUniqueId;
//   std::clog << "Zibai what is the pipe_name " << pipe_name
//             << "\n"; // this part is fine
//   int host_pipe_read_data;
//   void *data_ptr = &host_pipe_read_data;
//   std::clog << "Zibai started the q submit for read\n";
//   event e = q.submit([&](handler &CGH) {
//     CGH.read_write_host_pipe(pipe_name, data_ptr, sizeof(int), false,
//                              true /* read */);
//   });

//   std::clog << "Zibai started the wait for read\n";
//   e.wait();
//   std::clog << "Zibai started the assert\n";
//   assert(host_pipe_read_data == PipeReadVal);
//   int tmp = {9};
//   data_ptr = &tmp;
//   std::clog << "Zibai started the q submit for write\n";
//   event e_write = q.submit([&](handler &CGH) {
//     CGH.read_write_host_pipe(pipe_name, data_ptr, sizeof(int), false,
//                              false /* write */);
//   });
//   e_write.wait();
//   assert(PipeWriteVal == 9);
// }