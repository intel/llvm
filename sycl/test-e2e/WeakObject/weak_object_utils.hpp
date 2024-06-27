// Utilities for weak_object testing

#include <sycl/accessor_image.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/weak_object.hpp>
#include <sycl/image.hpp>
#include <sycl/sampler.hpp>
#include <sycl/usm.hpp>

class TestKernel1;
class TestKernel2;

void MaterializeTestKernels(sycl::queue Q) {
  if (false) {
    Q.single_task<TestKernel1>([]() {});
    Q.single_task<TestKernel2>([]() {});
  }
}

template <template <typename> typename CallableT> void runTest(sycl::queue Q) {
  MaterializeTestKernels(Q);

  // Auxiliary variables.
  sycl::half Data[2 * 3 * 4];
  sycl::image_sampler Sampler{sycl::addressing_mode::none,
                              sycl::coordinate_normalization_mode::unnormalized,
                              sycl::filtering_mode::linear};

  sycl::context Ctx = Q.get_context();
  sycl::device Dev = Q.get_device();
  sycl::platform Plt = Dev.get_platform();
  sycl::event E;
  sycl::kernel_id KId = sycl::get_kernel_id<TestKernel1>();
  sycl::kernel_bundle KB =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  sycl::buffer<int, 1> Buf1D{1};
  sycl::buffer<int, 2> Buf2D{sycl::range<2>{1, 2}};
  sycl::buffer<int, 3> Buf3D{sycl::range<3>{1, 2, 3}};
  sycl::unsampled_image<1> UImg1D{sycl::image_format::r8g8b8a8_uint,
                                  sycl::range<1>{1}};
  sycl::unsampled_image<2> UImg2D{sycl::image_format::r8g8b8a8_uint,
                                  sycl::range<2>{1, 2}};
  sycl::unsampled_image<3> UImg3D{sycl::image_format::r8g8b8a8_uint,
                                  sycl::range<3>{1, 2, 3}};
  sycl::sampled_image<1> SImg1D{Data, sycl::image_format::r8g8b8a8_uint,
                                Sampler, sycl::range<1>{1}};
  sycl::sampled_image<2> SImg2D{Data, sycl::image_format::r8g8b8a8_uint,
                                Sampler, sycl::range<2>{1, 2}};
  sycl::sampled_image<3> SImg3D{Data, sycl::image_format::r8g8b8a8_uint,
                                Sampler, sycl::range<3>{1, 2, 3}};
  sycl::accessor PAcc1D{Buf1D, sycl::read_write};
  sycl::accessor PAcc2D{Buf2D, sycl::read_write};
  sycl::accessor PAcc3D{Buf3D, sycl::read_write};
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                 sycl::access::target::host_buffer>
      HAcc1D;
  sycl::accessor<int, 2, sycl::access::mode::read_write,
                 sycl::access::target::host_buffer>
      HAcc2D;
  sycl::accessor<int, 3, sycl::access::mode::read_write,
                 sycl::access::target::host_buffer>
      HAcc3D;
  sycl::host_accessor<int, 1> HAcc1D_2020;
  sycl::host_accessor<int, 2> HAcc2D_2020;
  sycl::host_accessor<int, 3> HAcc3D_2020;
  sycl::host_unsampled_image_accessor<sycl::int4, 1> UImgHAcc1D{UImg1D};
  sycl::host_unsampled_image_accessor<sycl::int4, 2> UImgHAcc2D{UImg2D};
  sycl::host_unsampled_image_accessor<sycl::int4, 3> UImgHAcc3D{UImg3D};
  sycl::host_sampled_image_accessor<sycl::int4, 1> SImgHAcc1D{SImg1D};
  sycl::host_sampled_image_accessor<sycl::int4, 2> SImgHAcc2D{SImg2D};
  sycl::host_sampled_image_accessor<sycl::int4, 3> SImgHAcc3D{SImg3D};

  CallableT<decltype(Plt)>()(Plt);
  CallableT<decltype(Dev)>()(Dev);
  CallableT<decltype(Ctx)>()(Ctx);
  CallableT<decltype(Q)>()(Q);
  CallableT<decltype(E)>()(E);
  CallableT<decltype(KId)>()(KId);
  CallableT<decltype(KB)>()(KB);
  CallableT<decltype(Buf1D)>()(Buf1D);
  CallableT<decltype(Buf2D)>()(Buf2D);
  CallableT<decltype(Buf3D)>()(Buf3D);
  CallableT<decltype(UImg1D)>()(UImg1D);
  CallableT<decltype(UImg2D)>()(UImg2D);
  CallableT<decltype(UImg3D)>()(UImg3D);
  CallableT<decltype(SImg1D)>()(SImg1D);
  CallableT<decltype(SImg2D)>()(SImg2D);
  CallableT<decltype(SImg3D)>()(SImg3D);
  CallableT<decltype(PAcc1D)>()(PAcc1D);
  CallableT<decltype(PAcc2D)>()(PAcc2D);
  CallableT<decltype(PAcc3D)>()(PAcc3D);
  CallableT<decltype(HAcc1D)>()(HAcc1D);
  CallableT<decltype(HAcc2D)>()(HAcc2D);
  CallableT<decltype(HAcc3D)>()(HAcc3D);
  CallableT<decltype(HAcc1D_2020)>()(HAcc1D_2020);
  CallableT<decltype(HAcc2D_2020)>()(HAcc2D_2020);
  CallableT<decltype(HAcc3D_2020)>()(HAcc3D_2020);
  CallableT<decltype(UImgHAcc1D)>()(UImgHAcc1D);
  CallableT<decltype(UImgHAcc2D)>()(UImgHAcc2D);
  CallableT<decltype(UImgHAcc3D)>()(UImgHAcc3D);
  CallableT<decltype(SImgHAcc1D)>()(SImgHAcc1D);
  CallableT<decltype(SImgHAcc2D)>()(SImgHAcc2D);
  CallableT<decltype(SImgHAcc3D)>()(SImgHAcc3D);

  Q.submit([&](sycl::handler &CGH) {
    sycl::accessor DAcc1D{Buf1D, CGH, sycl::read_only};
    sycl::accessor DAcc2D{Buf2D, CGH, sycl::read_only};
    sycl::accessor DAcc3D{Buf3D, CGH, sycl::read_only};
    sycl::local_accessor<int> LAcc1D{1, CGH};
    sycl::local_accessor<int, 2> LAcc2D{sycl::range<2>{1, 2}, CGH};
    sycl::local_accessor<int, 3> LAcc3D{sycl::range<3>{1, 2, 3}, CGH};
    sycl::stream Stream{1024, 32, CGH};
    sycl::unsampled_image_accessor<sycl::int4, 1, sycl::access_mode::read,
                                   sycl::image_target::host_task>
        UImgAcc1D{UImg1D, CGH};
    sycl::unsampled_image_accessor<sycl::int4, 2, sycl::access_mode::read,
                                   sycl::image_target::host_task>
        UImgAcc2D{UImg2D, CGH};
    sycl::unsampled_image_accessor<sycl::int4, 3, sycl::access_mode::read,
                                   sycl::image_target::host_task>
        UImgAcc3D{UImg3D, CGH};
    sycl::sampled_image_accessor<sycl::int4, 1, sycl::image_target::host_task>
        SImgAcc1D{SImg1D, CGH};
    sycl::sampled_image_accessor<sycl::int4, 2, sycl::image_target::host_task>
        SImgAcc2D{SImg2D, CGH};
    sycl::sampled_image_accessor<sycl::int4, 3, sycl::image_target::host_task>
        SImgAcc3D{SImg3D, CGH};

    CallableT<decltype(DAcc1D)>()(DAcc1D);
    CallableT<decltype(DAcc2D)>()(DAcc2D);
    CallableT<decltype(DAcc3D)>()(DAcc3D);
    CallableT<decltype(LAcc1D)>()(LAcc1D);
    CallableT<decltype(LAcc2D)>()(LAcc2D);
    CallableT<decltype(LAcc3D)>()(LAcc3D);
    CallableT<decltype(Stream)>()(Stream);
    CallableT<decltype(UImgAcc1D)>()(UImgAcc1D);
    CallableT<decltype(UImgAcc2D)>()(UImgAcc2D);
    CallableT<decltype(UImgAcc3D)>()(UImgAcc3D);
    CallableT<decltype(SImgAcc1D)>()(SImgAcc1D);
    CallableT<decltype(SImgAcc2D)>()(SImgAcc2D);
    CallableT<decltype(SImgAcc3D)>()(SImgAcc3D);
  });
}

template <template <typename> typename CallableT>
void runTestMulti(sycl::queue Q1) {
  MaterializeTestKernels(Q1);

  // Auxiliary variables.
  sycl::half Data[2 * 3 * 4];
  sycl::image_sampler Sampler{sycl::addressing_mode::none,
                              sycl::coordinate_normalization_mode::unnormalized,
                              sycl::filtering_mode::linear};

  sycl::context Ctx1 = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  sycl::platform Plt = Dev.get_platform();

  sycl::context Ctx2{Dev};
  sycl::queue Q2{Dev};
  sycl::event E1;
  sycl::event E2;
  sycl::kernel_id KId1 = sycl::get_kernel_id<TestKernel1>();
  sycl::kernel_id KId2 = sycl::get_kernel_id<TestKernel2>();
  sycl::kernel_bundle KB1 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx1, {Dev});
  sycl::kernel_bundle KB2 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx2, {Dev});
  sycl::buffer<int, 1> Buf1D1{1};
  sycl::buffer<int, 1> Buf1D2{1};
  sycl::buffer<int, 2> Buf2D1{sycl::range<2>{1, 2}};
  sycl::buffer<int, 2> Buf2D2{sycl::range<2>{1, 2}};
  sycl::buffer<int, 3> Buf3D1{sycl::range<3>{1, 2, 3}};
  sycl::buffer<int, 3> Buf3D2{sycl::range<3>{1, 2, 3}};
  sycl::unsampled_image<1> UImg1D1{sycl::image_format::r8g8b8a8_uint,
                                   sycl::range<1>{1}};
  sycl::unsampled_image<1> UImg1D2{sycl::image_format::r8g8b8a8_uint,
                                   sycl::range<1>{1}};
  sycl::unsampled_image<2> UImg2D1{sycl::image_format::r8g8b8a8_uint,
                                   sycl::range<2>{1, 2}};
  sycl::unsampled_image<2> UImg2D2{sycl::image_format::r8g8b8a8_uint,
                                   sycl::range<2>{1, 2}};
  sycl::unsampled_image<3> UImg3D1{sycl::image_format::r8g8b8a8_uint,
                                   sycl::range<3>{1, 2, 3}};
  sycl::unsampled_image<3> UImg3D2{sycl::image_format::r8g8b8a8_uint,
                                   sycl::range<3>{1, 2, 3}};
  sycl::sampled_image<1> SImg1D1{Data, sycl::image_format::r8g8b8a8_uint,
                                 Sampler, sycl::range<1>{1}};
  sycl::sampled_image<1> SImg1D2{Data, sycl::image_format::r8g8b8a8_uint,
                                 Sampler, sycl::range<1>{1}};
  sycl::sampled_image<2> SImg2D1{Data, sycl::image_format::r8g8b8a8_uint,
                                 Sampler, sycl::range<2>{1, 2}};
  sycl::sampled_image<2> SImg2D2{Data, sycl::image_format::r8g8b8a8_uint,
                                 Sampler, sycl::range<2>{1, 2}};
  sycl::sampled_image<3> SImg3D1{Data, sycl::image_format::r8g8b8a8_uint,
                                 Sampler, sycl::range<3>{1, 2, 3}};
  sycl::sampled_image<3> SImg3D2{Data, sycl::image_format::r8g8b8a8_uint,
                                 Sampler, sycl::range<3>{1, 2, 3}};
  sycl::accessor PAcc1D1{Buf1D1, sycl::read_write};
  sycl::accessor PAcc1D2{Buf1D2, sycl::read_write};
  sycl::accessor PAcc2D1{Buf2D1, sycl::read_write};
  sycl::accessor PAcc2D2{Buf2D2, sycl::read_write};
  sycl::accessor PAcc3D1{Buf3D1, sycl::read_write};
  sycl::accessor PAcc3D2{Buf3D2, sycl::read_write};
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                 sycl::access::target::host_buffer>
      HAcc1D1;
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                 sycl::access::target::host_buffer>
      HAcc1D2;
  sycl::accessor<int, 2, sycl::access::mode::read_write,
                 sycl::access::target::host_buffer>
      HAcc2D1;
  sycl::accessor<int, 2, sycl::access::mode::read_write,
                 sycl::access::target::host_buffer>
      HAcc2D2;
  sycl::accessor<int, 3, sycl::access::mode::read_write,
                 sycl::access::target::host_buffer>
      HAcc3D1;
  sycl::accessor<int, 3, sycl::access::mode::read_write,
                 sycl::access::target::host_buffer>
      HAcc3D2;
  sycl::host_accessor<int, 1> HAcc1D1_2020;
  sycl::host_accessor<int, 2> HAcc2D1_2020;
  sycl::host_accessor<int, 3> HAcc3D1_2020;
  sycl::host_accessor<int, 1> HAcc1D2_2020;
  sycl::host_accessor<int, 2> HAcc2D2_2020;
  sycl::host_accessor<int, 3> HAcc3D2_2020;
  sycl::host_unsampled_image_accessor<sycl::int4, 1> UImgHAcc1D1{UImg1D1};
  sycl::host_unsampled_image_accessor<sycl::int4, 2> UImgHAcc2D1{UImg2D1};
  sycl::host_unsampled_image_accessor<sycl::int4, 3> UImgHAcc3D1{UImg3D1};
  sycl::host_unsampled_image_accessor<sycl::int4, 1> UImgHAcc1D2{UImg1D2};
  sycl::host_unsampled_image_accessor<sycl::int4, 2> UImgHAcc2D2{UImg2D2};
  sycl::host_unsampled_image_accessor<sycl::int4, 3> UImgHAcc3D2{UImg3D2};
  sycl::host_sampled_image_accessor<sycl::int4, 1> SImgHAcc1D1{SImg1D1};
  sycl::host_sampled_image_accessor<sycl::int4, 2> SImgHAcc2D1{SImg2D1};
  sycl::host_sampled_image_accessor<sycl::int4, 3> SImgHAcc3D1{SImg3D1};
  sycl::host_sampled_image_accessor<sycl::int4, 1> SImgHAcc1D2{SImg1D2};
  sycl::host_sampled_image_accessor<sycl::int4, 2> SImgHAcc2D2{SImg2D2};
  sycl::host_sampled_image_accessor<sycl::int4, 3> SImgHAcc3D2{SImg3D2};

  CallableT<decltype(Ctx1)>()(Ctx1, Ctx2);
  CallableT<decltype(Q1)>()(Q1, Q2);
  CallableT<decltype(E1)>()(E1, E2);
  CallableT<decltype(KId1)>()(KId1, KId2);
  CallableT<decltype(KB1)>()(KB1, KB2);
  CallableT<decltype(Buf1D1)>()(Buf1D1, Buf1D2);
  CallableT<decltype(Buf2D1)>()(Buf2D1, Buf2D2);
  CallableT<decltype(Buf3D1)>()(Buf3D1, Buf3D2);
  CallableT<decltype(UImg1D1)>()(UImg1D1, UImg1D2);
  CallableT<decltype(UImg2D1)>()(UImg2D1, UImg2D2);
  CallableT<decltype(UImg3D1)>()(UImg3D1, UImg3D2);
  CallableT<decltype(SImg1D1)>()(SImg1D1, SImg1D2);
  CallableT<decltype(SImg2D1)>()(SImg2D1, SImg2D2);
  CallableT<decltype(SImg3D1)>()(SImg3D1, SImg3D2);
  CallableT<decltype(PAcc1D1)>()(PAcc1D1, PAcc1D2);
  CallableT<decltype(PAcc2D1)>()(PAcc2D1, PAcc2D2);
  CallableT<decltype(PAcc3D1)>()(PAcc3D1, PAcc3D2);
  CallableT<decltype(HAcc1D1)>()(HAcc1D1, HAcc1D2);
  CallableT<decltype(HAcc2D1)>()(HAcc2D1, HAcc2D2);
  CallableT<decltype(HAcc3D1)>()(HAcc3D1, HAcc3D2);
  CallableT<decltype(HAcc1D1_2020)>()(HAcc1D1_2020, HAcc1D2_2020);
  CallableT<decltype(HAcc2D1_2020)>()(HAcc2D1_2020, HAcc2D2_2020);
  CallableT<decltype(HAcc3D1_2020)>()(HAcc3D1_2020, HAcc3D2_2020);
  CallableT<decltype(UImgHAcc1D1)>()(UImgHAcc1D1, UImgHAcc1D2);
  CallableT<decltype(UImgHAcc2D1)>()(UImgHAcc2D1, UImgHAcc2D2);
  CallableT<decltype(UImgHAcc3D1)>()(UImgHAcc3D1, UImgHAcc3D2);
  CallableT<decltype(SImgHAcc1D1)>()(SImgHAcc1D1, SImgHAcc1D2);
  CallableT<decltype(SImgHAcc2D1)>()(SImgHAcc2D1, SImgHAcc2D2);
  CallableT<decltype(SImgHAcc3D1)>()(SImgHAcc3D1, SImgHAcc3D2);

  Q1.submit([&](sycl::handler &CGH) {
    sycl::accessor DAcc1D1{Buf1D1, CGH, sycl::read_only};
    sycl::accessor DAcc1D2{Buf1D2, CGH, sycl::read_only};
    sycl::accessor DAcc2D1{Buf2D1, CGH, sycl::read_only};
    sycl::accessor DAcc2D2{Buf2D2, CGH, sycl::read_only};
    sycl::accessor DAcc3D1{Buf3D1, CGH, sycl::read_only};
    sycl::accessor DAcc3D2{Buf3D2, CGH, sycl::read_only};
    sycl::local_accessor<int, 1> LAcc1D1{1, CGH};
    sycl::local_accessor<int, 1> LAcc1D2{1, CGH};
    sycl::local_accessor<int, 2> LAcc2D1{sycl::range<2>{1, 2}, CGH};
    sycl::local_accessor<int, 2> LAcc2D2{sycl::range<2>{1, 2}, CGH};
    sycl::local_accessor<int, 3> LAcc3D1{sycl::range<3>{1, 2, 3}, CGH};
    sycl::local_accessor<int, 3> LAcc3D2{sycl::range<3>{1, 2, 3}, CGH};
    sycl::stream Stream1{1024, 32, CGH};
    sycl::stream Stream2{1024, 32, CGH};
    sycl::unsampled_image_accessor<sycl::int4, 1, sycl::access_mode::read,
                                   sycl::image_target::host_task>
        UImgAcc1D1{UImg1D1, CGH};
    sycl::unsampled_image_accessor<sycl::int4, 2, sycl::access_mode::read,
                                   sycl::image_target::host_task>
        UImgAcc2D1{UImg2D1, CGH};
    sycl::unsampled_image_accessor<sycl::int4, 3, sycl::access_mode::read,
                                   sycl::image_target::host_task>
        UImgAcc3D1{UImg3D1, CGH};
    sycl::unsampled_image_accessor<sycl::int4, 1, sycl::access_mode::read,
                                   sycl::image_target::host_task>
        UImgAcc1D2{UImg1D2, CGH};
    sycl::unsampled_image_accessor<sycl::int4, 2, sycl::access_mode::read,
                                   sycl::image_target::host_task>
        UImgAcc2D2{UImg2D2, CGH};
    sycl::unsampled_image_accessor<sycl::int4, 3, sycl::access_mode::read,
                                   sycl::image_target::host_task>
        UImgAcc3D2{UImg3D2, CGH};
    sycl::sampled_image_accessor<sycl::int4, 1, sycl::image_target::host_task>
        SImgAcc1D1{SImg1D1, CGH};
    sycl::sampled_image_accessor<sycl::int4, 2, sycl::image_target::host_task>
        SImgAcc2D1{SImg2D1, CGH};
    sycl::sampled_image_accessor<sycl::int4, 3, sycl::image_target::host_task>
        SImgAcc3D1{SImg3D1, CGH};
    sycl::sampled_image_accessor<sycl::int4, 1, sycl::image_target::host_task>
        SImgAcc1D2{SImg1D2, CGH};
    sycl::sampled_image_accessor<sycl::int4, 2, sycl::image_target::host_task>
        SImgAcc2D2{SImg2D2, CGH};
    sycl::sampled_image_accessor<sycl::int4, 3, sycl::image_target::host_task>
        SImgAcc3D2{SImg3D2, CGH};

    CallableT<decltype(DAcc1D1)>()(DAcc1D1, DAcc1D2);
    CallableT<decltype(DAcc2D1)>()(DAcc2D1, DAcc2D2);
    CallableT<decltype(DAcc3D1)>()(DAcc3D1, DAcc3D2);
    CallableT<decltype(LAcc1D1)>()(LAcc1D1, LAcc1D2);
    CallableT<decltype(LAcc2D1)>()(LAcc2D1, LAcc2D2);
    CallableT<decltype(LAcc3D1)>()(LAcc3D1, LAcc3D2);
    CallableT<decltype(Stream1)>()(Stream1, Stream2);
    CallableT<decltype(UImgAcc1D1)>()(UImgAcc1D1, UImgAcc1D2);
    CallableT<decltype(UImgAcc2D1)>()(UImgAcc2D1, UImgAcc2D2);
    CallableT<decltype(UImgAcc3D1)>()(UImgAcc3D1, UImgAcc3D2);
    CallableT<decltype(SImgAcc1D1)>()(SImgAcc1D1, SImgAcc1D2);
    CallableT<decltype(SImgAcc2D1)>()(SImgAcc2D1, SImgAcc2D2);
    CallableT<decltype(SImgAcc3D1)>()(SImgAcc3D1, SImgAcc3D2);
  });
}
