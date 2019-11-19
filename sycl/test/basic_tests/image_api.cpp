// RUN: %clangxx -fsycl %s -o %t1.out
// RUN: %clangxx %s -o %t3.out -lsycl
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t3.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

#include <CL/sycl.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <utility>

namespace s = cl::sycl;
namespace d = cl::sycl::detail;

struct FakeHandler {
  d::NDRDescT MNDRDesc;
  s::unique_ptr_class<d::HostKernelBase> MHostKernel;
  s::shared_ptr_class<d::kernel_impl> MSyclKernel;
  s::vector_class<s::vector_class<char>> MArgsStorage;
  s::vector_class<d::AccessorImplPtr> MAccStorage;
  s::vector_class<d::Requirement *> MRequirements;
  s::vector_class<d::ArgDesc> MArgs;
  s::vector_class<s::shared_ptr_class<const void>> MSharedPtrStorage;
  s::string_class MKernelName;
  d::OSModuleHandle MOSModuleHandle;
  s::vector_class<s::shared_ptr_class<d::stream_impl>> MStreamStorage;
};

int main() {
  constexpr size_t Size = 256;
  std::array<s::float4, Size> Src;
  std::array<s::float4, Size> Dest;
  std::fill(Src.begin(), Src.end(), s::float4{1.0f, 2.0f, 3.0f, 4.0f});
  std::fill(Dest.begin(), Dest.end(), s::float4{0.0f, 0.0f, 0.0f, 0.0});

  try {
    constexpr int Dimensions = 2;
    constexpr s::image_channel_order ChannelOrder =
        s::image_channel_order::rgba;
    constexpr s::image_channel_type ChannelType = s::image_channel_type::fp32;
    const s::range<Dimensions> Range{16, 16};
    s::image<Dimensions> SrcImage(Src.data(), ChannelOrder, ChannelType, Range);
    s::image<Dimensions> DestImage(Dest.data(), ChannelOrder, ChannelType,
                                   Range);

    s::queue Queue;
    s::device Device = Queue.get_device();
    if(!Device.get_info<s::info::device::image_support>()) {
      std::cout << "Images are not supported. The result can not be checked."
                << std::endl;
      return 0;
    }

    FakeHandler Handler;
    Handler.MNDRDesc.set(Range);
    if (Queue.is_host()) {
      auto KernelFunc = [&](s::id<Dimensions> ID) {
        s::int2 Coords(ID[0], ID[1]);
        int linearId = ID[1] + ID[0] * Range[0];
        s::float4 Color = Src[linearId];
        Color *= 10.0f;
        Dest[linearId] = Color;
      };

      Handler.MHostKernel.reset(
          new d::HostKernel<decltype(KernelFunc),
                            d::lambda_arg_type<decltype(KernelFunc)>,
                            Dimensions>(KernelFunc));
    } else {
      s::context Context = Queue.get_context();
      s::program Program(Context);
      Program.build_with_source(
          "__kernel void ImageTest(__read_only image2d_t Input, __write_only "
          "image2d_t Output, sampler_t sampler) {"
          "  int2 Coords = (int2)(get_global_id(0), get_global_id(1));"
          "  float4 Color = read_imagef(Input, sampler, Coords);"
          "  Color *= 10.0f;"
          "  write_imagef(Output, Coords, Color);"
          "}\n");
      s::kernel Kernel = Program.get_kernel("ImageTest");
      Handler.MSyclKernel = d::getSyclObjImpl(Kernel);
      Handler.MKernelName =
          Handler.MSyclKernel->get_info<s::info::kernel::function_name>();
      Handler.MOSModuleHandle =
          d::OSUtil::getOSModuleHandle(Handler.MKernelName.c_str());
    }

    auto addFakeImageAccessor = [&Handler, Dimensions](s::image<Dimensions> Image,
                                           s::access::mode Mode, int Index) {
      const s::id<3> Offset{0, 0, 0};
      const s::range<3> AccessRange{Image.get_range()[0], Image.get_range()[1], 1};
      const s::range<3> MemoryRange{Image.get_range()[0], Image.get_range()[1], 1};
      d::SYCLMemObjI *SYCLMemObject = static_cast<d::SYCLMemObjI *>(d::getSyclObjImpl(Image).get());
      const int ElemSize = d::getSyclObjImpl(Image)->getElementSize();

      d::AccessorImplPtr AccImpl = std::make_shared<d::Requirement>(Offset,
        AccessRange,MemoryRange, Mode, SYCLMemObject, Dimensions, ElemSize);

      d::Requirement *Req = AccImpl.get();
      Handler.MRequirements.push_back(Req);
      Handler.MAccStorage.push_back(AccImpl);
      Handler.MArgs.emplace_back(d::kernel_param_kind_t::kind_accessor, Req,
                                 static_cast<int>(s::access::target::image),
                                 Index);
    };

    addFakeImageAccessor(SrcImage, s::access::mode::read, 0);
    addFakeImageAccessor(DestImage, s::access::mode::write, 1);

    s::sampler Sampler(s::coordinate_normalization_mode::unnormalized,
                       s::addressing_mode::clamp, s::filtering_mode::nearest);
    Handler.MArgsStorage.emplace_back(sizeof(s::sampler));
    s::sampler *SamplerPtr =
        reinterpret_cast<s::sampler *>(Handler.MArgsStorage.back().data());
    *SamplerPtr = Sampler;
    Handler.MArgs.emplace_back(d::kernel_param_kind_t::kind_sampler,
                               static_cast<void *>(SamplerPtr),
                               sizeof(s::sampler), 2);

    s::unique_ptr_class<d::CG> CommandGroup;
    CommandGroup.reset(new d::CGExecKernel(
        std::move(Handler.MNDRDesc), std::move(Handler.MHostKernel),
        std::move(Handler.MSyclKernel), std::move(Handler.MArgsStorage),
        std::move(Handler.MAccStorage), std::move(Handler.MSharedPtrStorage),
        std::move(Handler.MRequirements), /*DepsEvents*/ {},
        std::move(Handler.MArgs), std::move(Handler.MKernelName),
        std::move(Handler.MOSModuleHandle), std::move(Handler.MStreamStorage),
        d::CG::KERNEL));

    d::EventImplPtr Event = d::Scheduler::getInstance().addCG(
        std::move(CommandGroup), d::getSyclObjImpl(Queue));

    s::event EventRet = d::createSyclObjFromImpl<s::event>(Event);
    EventRet.wait();
  } catch (const s::exception &E) {
    std::cout << "SYCL exception caught: " << E.what() << std::endl;
  }

  s::float4 Expected{10.f, 20.f, 30.f, 40.f};

  bool Result = std::all_of(Dest.cbegin(), Dest.cend(),
                            [Expected](const s::float4 &Value) -> bool {
                              return s::all(s::isequal(Value, Expected));
                            });

  if (Result) {
    std::cout << "The result is correct." << std::endl;
  } else {
    std::cout << "The result is incorrect." << std::endl;
    assert(Result);
  }
  return 0;
}
