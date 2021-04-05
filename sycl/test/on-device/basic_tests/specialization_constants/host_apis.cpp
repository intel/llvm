#include <sycl/sycl.hpp>

class Kernel1Name;
class Kernel2Name;

static sycl::specialization_id<int> SpecConst1{42};
static sycl::specialization_id<float> SpecConst2{42.f};
static sycl::specialization_id<double> SpecConst3{42.f};
static sycl::specialization_id<short> SpecConst4{42};

int main() {
  sycl::queue Q;

  // No support for host device so far
  if (Q.is_host())
    return 0;

  // The code is needed to just have device images in the executable
  if (0) {
    Q.submit([](sycl::handler &CGH) { CGH.single_task<Kernel1Name>([]{}); });
    Q.submit([](sycl::handler &CGH) { CGH.single_task<Kernel2Name>([]{}); });
  }
  
  const sycl::context Ctx = Q.get_context();
  const sycl::device Dev = Q.get_device();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});

  assert(KernelBundle.has_specialization_constant<SpecConst1>() == true);
  KernelBundle.set_specialization_constant<SpecConst2>(1.f);
  const auto SC = KernelBundle.get_specialization_constant<SpecConst1>();

  Q.submit([](sycl::handler &CGH) {
    CGH.set_specialization_constant<SpecConst3>(0.f);
    const auto SC = CGH.get_specialization_constant<SpecConst4>();
    CGH.single_task<class Kernel3Name>([]{});
  });

  return 0;
}
