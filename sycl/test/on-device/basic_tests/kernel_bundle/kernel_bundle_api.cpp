// RUN: %clangxx -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
//
// -fsycl-device-code-split is not supported for cuda
// UNSUPPORTED: cuda

#include <CL/sycl.hpp>
#include <detail/device_image_impl.hpp>

#include <vector>

class Kernel1Name;
class Kernel2Name;
class Kernel3Name;

int main() {
  sycl::queue Q;

  // No support for host device so far.
  if (Q.is_host())
    return 0;

  const sycl::context Ctx = Q.get_context();
  const sycl::device Dev = Q.get_device();

  if (0) {
    Q.submit([](sycl::handler &CGH) { CGH.single_task<Kernel1Name>([]() {}); });
    Q.submit([](sycl::handler &CGH) { CGH.single_task<Kernel2Name>([]() {}); });
  }

  sycl::kernel_id Kernel1ID = sycl::get_kernel_id<Kernel1Name>();
  sycl::kernel_id Kernel2ID = sycl::get_kernel_id<Kernel2Name>();

#if 0
  {
    sycl::kernel_bundle KernelBundle1 =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});

    sycl::kernel_bundle KernelBundleCopy = KernelBundle1;
    assert(KernelBundleCopy == KernelBundle1);
    assert(!(KernelBundleCopy != KernelBundle1));
    assert(false == KernelBundle1.empty());
    assert(Ctx.get_platform().get_backend() == KernelBundle1.get_backend());
    assert(KernelBundle1.get_context() == Ctx);
    assert(KernelBundle1.get_devices() == (std::vector<sycl::device>){Dev});
    assert(KernelBundle1.has_kernel(Kernel1ID));
    assert(KernelBundle1.has_kernel(Kernel2ID));
    assert(KernelBundle1.has_kernel(Kernel1ID, Dev));
    assert(KernelBundle1.has_kernel(Kernel2ID, Dev));

    assert(std::any_of(
        KernelBundle1.begin(), KernelBundle1.end(),
        [&Kernel1ID](
            const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel1ID);
        }));

    assert(std::any_of(
        KernelBundle1.begin(), KernelBundle1.end(),
        [&Kernel2ID](
            const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel2ID);
        }));

    assert(std::any_of(
        KernelBundle1.begin(), KernelBundle1.end(),
        [&Kernel1ID, &Dev](
            const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel1ID, Dev);
        }));

    assert(std::any_of(
        KernelBundle1.begin(), KernelBundle1.end(),
        [&Kernel2ID, &Dev](
            const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel2ID, Dev);
        }));
  }
#endif

  // The following check relies on "-fsycl-device-code-split=per_kernel" option,
  // so it is expected that each kernel in a separate device image.
  // Verify that get_kernel_bundle filters out device images based on vector
  // of kernel_id's and Selector.

  {
    //sycl::kernel_bundle KernelBundle2 =
        //sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                           //{Kernel1ID});
    //assert(KernelBundle2.has_kernel(Kernel1ID));
    //assert(!KernelBundle2.has_kernel(Kernel2ID));

    //auto Selector =
        //[&Kernel2ID](
            //const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          //return DevImage.has_kernel(Kernel2ID);
        //};

    //sycl::kernel_bundle KernelBundle3 =
        //sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                           //Selector);
    //assert(!KernelBundle3.has_kernel(Kernel1ID));
    //assert(KernelBundle3.has_kernel(Kernel2ID));

    //sycl::kernel_bundle KernelBundle4 =
        //sycl::join(std::vector<sycl::kernel_bundle<sycl::bundle_state::input>>{
            //KernelBundle2, KernelBundle3});

    //assert(KernelBundle4.has_kernel(Kernel1ID));
    //assert(KernelBundle4.has_kernel(Kernel2ID));

    //sycl::kernel_bundle KernelBundle12 =
        //sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                           //{Kernel1ID});

    //sycl::kernel_bundle KernelBundle5 =
        //sycl::join(std::vector<sycl::kernel_bundle<sycl::bundle_state::input>>{
            //KernelBundle4, KernelBundle12});

    //std::vector<sycl::kernel_id> KernelBundle5KernelIDs =
        //KernelBundle5.get_kernel_ids();

    //assert(KernelBundle5KernelIDs.size() == 2);

    sycl::kernel_bundle KernelBundle6 =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});

    //std::cout << "KernelBundle6 programs before compile:" << std::endl;
    //for (const sycl::device_image<sycl::bundle_state::input> &DevImage :
         //KernelBundle6) {
      //auto Impl = sycl::detail::getSyclObjImpl(DevImage);
      //std::cout << "\tProgram ptr = " << Impl->get_program_ref() << std::endl;
    //}

    //sycl::kernel_bundle<sycl::bundle_state::object> KernelBundle6Compiled =
        //sycl::compile(KernelBundle6, KernelBundle6.get_devices());

    //std::cout << "KernelBundle6 programs after compile:" << std::endl;
    //for (const sycl::device_image<sycl::bundle_state::input> &DevImage :
         //KernelBundle6) {
      //auto Impl = sycl::detail::getSyclObjImpl(DevImage);
      //std::cout << "\tProgram ptr = " << Impl->get_program_ref() << std::endl;
    //}


    //std::cout << "KernelBundle6Compiled programs after compile:" << std::endl;
    //for (const sycl::device_image<sycl::bundle_state::object> &DevImage :
         //KernelBundle6Compiled) {
      //auto Impl = sycl::detail::getSyclObjImpl(DevImage);
      //std::cout << "\tProgram ptr = " << Impl->get_program_ref() << std::endl;
    //}

    //sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle7Linked =
        //sycl::link({KernelBundle6Compiled}, KernelBundle6.get_devices());

    //std::cout << "KernelBundle7Linked programs after link:" << std::endl;
    //for (const sycl::device_image<sycl::bundle_state::executable> &DevImage :
         //KernelBundle7Linked) {
      //auto Impl = sycl::detail::getSyclObjImpl(DevImage);
      //std::cout << "\tProgram ptr = " << Impl->get_program_ref() << std::endl;
    //}

    //sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle6Built =
        //sycl::build(KernelBundle6, KernelBundle6.get_devices());
    //std::cout << "KernelBundle7Linked programs after build:" << std::endl;
    //for (const sycl::device_image<sycl::bundle_state::executable> &DevImage :
         //KernelBundle6Built) {
      //auto Impl = sycl::detail::getSyclObjImpl(DevImage);
      //std::cout << "\tProgram ptr = " << Impl->get_program_ref() << std::endl;
    //}

    //static constexpr inline int SpecConstEmulationVar = 0;
    sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle6Built2 =
        sycl::build(KernelBundle6, KernelBundle6.get_devices());

    cl::sycl::buffer<int, 1> Buf(sycl::range<1>{16});

    Q.submit([&](sycl::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::write>(CGH);
      CGH.use_kernel_bundle(KernelBundle6Built2);
      CGH.single_task<Kernel3Name>([=]() { 
          Acc[0] = 42;
          });
    });
    auto HostAcc = Buf.get_access<sycl::access::mode::write>();
    std::cout << "HostAcc[0] = " << HostAcc[0] << std::endl;
  }

  return 0;
}
