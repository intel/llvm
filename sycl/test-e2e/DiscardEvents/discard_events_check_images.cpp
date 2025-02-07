// UNSUPPORTED: hip
// REQUIRES: aspect-ext_intel_legacy_image
//
// RUN: %{build} -o %t.out
//
// RUN: %{run} %t.out image
// RUN: %{run} %t.out mixed
//
// Note that the tests use image functionality and if you have problems with
// the tests, please check if they pass without the discard_events property, if
// they don't pass then it's most likely a general issue unrelated to
// discard_events.

#include "../helpers.hpp" // for printableVec
#include <cassert>
#include <iostream>
#include <sycl/accessor_image.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/image.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

using namespace sycl;
static constexpr size_t BUFFER_SIZE = 1024;
static constexpr int MAX_ITER_NUM1 = 10;
static constexpr int MAX_ITER_NUM2 = 10;
static constexpr int InitialVal = MAX_ITER_NUM1;

void TestHelper(sycl::queue Q,
                const std::function<void(sycl::range<2> ImgSize, int *Harray,
                                         sycl::image<2> Img)> &Function) {
  int *Harray = sycl::malloc_shared<int>(BUFFER_SIZE, Q);
  assert(Harray != nullptr);
  for (size_t i = 0; i < BUFFER_SIZE; ++i) {
    Harray[i] = 0;
  }

  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType =
      sycl::image_channel_type::signed_int32;

  const sycl::range<2> ImgSize(sycl::sqrt(static_cast<float>(BUFFER_SIZE)),
                               sycl::sqrt(static_cast<float>(BUFFER_SIZE)));
  std::vector<sycl::int4> ImgHostData(
      ImgSize.size(), {InitialVal, InitialVal, InitialVal, InitialVal});
  sycl::image<2> Img(ImgHostData.data(), ChanOrder, ChanType, ImgSize);

  Function(ImgSize, Harray, Img);

  free(Harray, Q);
}

void IfTrueIncrementUSM(sycl::queue Q, sycl::range<1> Range, int *Harray,
                        int ValueToCheck) {
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class increment_usm>(Range, [=](sycl::item<1> itemID) {
      size_t i = itemID.get_id(0);
      if (Harray[i] == ValueToCheck) {
        Harray[i] += 1;
      }
    });
  });
}

void IfTrueIncrementImageAndUSM(sycl::queue Q, sycl::range<2> ImgSize,
                                int *Harray, sycl::image<2> Img,
                                int HarrayValueToCheck, int ImageValueToCheck) {
  Q.submit([&](sycl::handler &CGH) {
    auto Img1Acc = Img.get_access<sycl::int4, sycl::access::mode::read>(CGH);
    auto Img2Acc = Img.get_access<sycl::int4, sycl::access::mode::write>(CGH);
    CGH.parallel_for<class ImgCopy>(ImgSize, [=](sycl::item<2> Item) {
      size_t i = Item.get_linear_id();
      if (Harray[i] == HarrayValueToCheck) {
        sycl::int4 Data = Img1Acc.read(sycl::int2{Item[0], Item[1]});
        if (Data[0] == ImageValueToCheck && Data[1] == ImageValueToCheck &&
            Data[2] == ImageValueToCheck && Data[3] == ImageValueToCheck) {
          Data[0]++;
          Data[3] = Data[2] = Data[1] = Data[0];
          Img2Acc.write(sycl::int2{Item[0], Item[1]}, Data);
        }
        ++Harray[i];
      }
    });
  });
}

void RunTest_ImageTest(sycl::queue Q) {
  TestHelper(Q, [&](sycl::range<2> ImgSize, int *Harray, sycl::image<2> Img) {
    sycl::range<1> Range(BUFFER_SIZE);
    for (int i = 0; i < MAX_ITER_NUM1; ++i)
      IfTrueIncrementUSM(Q, Range, Harray, (i));

    for (int i = 0; i < MAX_ITER_NUM2; ++i)
      IfTrueIncrementImageAndUSM(Q, ImgSize, Harray, Img, (MAX_ITER_NUM1 + i),
                                 (InitialVal + i));
    Q.wait();

    // check results
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
      int expected = MAX_ITER_NUM1 + MAX_ITER_NUM2;
      assert(Harray[i] == expected);
    }

    {
      auto HostAcc =
          Img.template get_access<sycl::int4, sycl::access::mode::read>();
      int expected = InitialVal + MAX_ITER_NUM2;
      for (int X = 0; X < ImgSize[0]; ++X)
        for (int Y = 0; Y < ImgSize[1]; ++Y) {
          sycl::int4 Vec1 = sycl::int4(expected);
          sycl::int4 Vec2 = HostAcc.read(sycl::int2{X, Y});
          if (Vec1[0] != Vec2[0] || Vec1[1] != Vec2[1] || Vec1[2] != Vec2[2] ||
              Vec1[3] != Vec2[3]) {
            std::cerr << "Failed" << std::endl;
            std::cerr << "Element [ " << X << ", " << Y << " ]" << std::endl;
            std::cerr << "Expected: " << printableVec(Vec1) << std::endl;
            std::cerr << " Got    : " << printableVec(Vec2) << std::endl;
            assert(false && "ImageTest failed!");
          }
        }
    }
  });
}

void RunTest_ImageTest_Mixed(sycl::queue Q) {
  TestHelper(Q, [&](sycl::range<2> ImgSize, int *Harray, sycl::image<2> Img) {
    sycl::range<1> Range(BUFFER_SIZE);

    for (int i = 0; i < MAX_ITER_NUM1; ++i) {
      IfTrueIncrementUSM(Q, Range, Harray, (i * 2));
      IfTrueIncrementImageAndUSM(Q, ImgSize, Harray, Img, (i * 2 + 1),
                                 (InitialVal + i));
    }

    for (int i = 0; i < MAX_ITER_NUM2; ++i) {
      IfTrueIncrementImageAndUSM(Q, ImgSize, Harray, Img,
                                 (MAX_ITER_NUM1 * 2 + i * 2),
                                 (InitialVal + MAX_ITER_NUM1 + i));
      IfTrueIncrementUSM(Q, Range, Harray, (MAX_ITER_NUM1 * 2 + i * 2 + 1));
    }

    Q.wait();

    // check results
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
      int expected = MAX_ITER_NUM1 * 2 + MAX_ITER_NUM2 * 2;
      assert(Harray[i] == expected);
    }

    {
      auto HostAcc =
          Img.template get_access<sycl::int4, sycl::access::mode::read>();
      int expected = InitialVal + MAX_ITER_NUM1 + MAX_ITER_NUM2;
      for (int X = 0; X < ImgSize[0]; ++X)
        for (int Y = 0; Y < ImgSize[1]; ++Y) {
          sycl::int4 Vec1 = sycl::int4(expected);
          sycl::int4 Vec2 = HostAcc.read(sycl::int2{X, Y});
          if (Vec1[0] != Vec2[0] || Vec1[1] != Vec2[1] || Vec1[2] != Vec2[2] ||
              Vec1[3] != Vec2[3]) {
            std::cerr << "Failed" << std::endl;
            std::cerr << "Element [ " << X << ", " << Y << " ]" << std::endl;
            std::cerr << "Expected: " << printableVec(Vec1) << std::endl;
            std::cerr << " Got    : " << printableVec(Vec2) << std::endl;
            assert(false && "ImageTest_Mixed failed!");
          }
        }
    }
  });
}

int main(int Argc, const char *Argv[]) {
  assert(Argc == 2 && "Invalid number of arguments");
  std::string TestType(Argv[1]);

  sycl::property_list props{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue Q(props);

  auto dev = Q.get_device();
  if (TestType == "image") {
    std::cerr << "RunTest_ImageTest" << std::endl;
    RunTest_ImageTest(Q);
  } else if (TestType == "mixed") {
    std::cerr << "RunTest_ImageTest_Mixed" << std::endl;
    RunTest_ImageTest_Mixed(Q);
  } else {
    assert(0 && "Unsupported test type!");
  }

  std::cout << "The test passed." << std::endl;
  return 0;
}
