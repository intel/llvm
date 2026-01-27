// REQUIRES: opencl-aot, ocloc, target-spir

// RUN: %if any-device-is-gpu %{ %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %s -o %t1.out %}
// RUN: %if gpu %{ %{run} %t1.out %}

// RUN: %if any-device-is-cpu %{ %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 %s -o %t2.out %}
// RUN: %if cpu %{ %{run} %t2.out %}
//
// Permute the order in which specialization constants are
// written into the input kernel_bundle to expose alignment
// issues when loading/storing the value of a composite specialization constant
// from the specialization constants buffer.

// Currently not supported on GPU because of the bug in IGC.
// UNSUPPORTED: gpu
// UNSUPPORTED-TRACKER: GSD-12237

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/specialization_id.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

struct Composite {
  float f;
  int i;
  char c;
  Composite() = delete;
  constexpr Composite(float fVal, int iVal, char cVal)
      : f(fVal), i(iVal), c(cVal) {}
  constexpr Composite(const Composite &) = default;
  bool operator==(const Composite &other) const {
    return f == other.f && i == other.i && c == other.c;
  }
};

class TestAlignment;

// Define several 1-byte spec constants and one Composite spec constant.
// We permute the order in which we call set_specialization_constant on the
// input kernel_bundle to exercise ordering effects.
constexpr sycl::specialization_id<char> char_spec0('a');
constexpr sycl::specialization_id<char> char_spec1('b');
constexpr sycl::specialization_id<char> char_spec2('c');
constexpr sycl::specialization_id<Composite> composite_spec(Composite{3.14f, 42,
                                                                      'X'});

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();
  sycl::context ctx = q.get_context();

  char char_out0 = 0, char_out1 = 0, char_out2 = 0;
  Composite comp_out{0.0f, 0, 0};

  char char_vals[3] = {'U', 'V', 'W'};
  Composite set_comp{2.71f, 99, 'Y'};

  // We will permute these 4 spec "items": indices 0..2 -> char specs,
  // index 3 -> composite spec.
  std::vector<int> items = {0, 1, 2, 3};

  // Record whether any permutation exposed a mismatch
  bool all_pass = true;
  int perm_index = 0;

  // Try all permutations (24)
  do {
    ++perm_index;
    {
      sycl::buffer<char, 1> b0(&char_out0, sycl::range<1>(1));
      sycl::buffer<char, 1> b1(&char_out1, sycl::range<1>(1));
      sycl::buffer<char, 1> b2(&char_out2, sycl::range<1>(1));
      sycl::buffer<Composite, 1> bc(&comp_out, sycl::range<1>(1));

      q.submit([&](sycl::handler &cgh) {
        // Apply set_specialization_constant in the order defined by this
        // permutation.
        for (int idx : items) {
          switch (idx) {
          case 0:
            cgh.set_specialization_constant<char_spec0>(char_vals[0]);
            break;
          case 1:
            cgh.set_specialization_constant<char_spec1>(char_vals[1]);
            break;
          case 2:
            cgh.set_specialization_constant<char_spec2>(char_vals[2]);
            break;
          case 3:
            cgh.set_specialization_constant<composite_spec>(set_comp);
            break;
          default:
            break;
          }
        }

        auto acc0 = b0.template get_access<sycl::access::mode::write>(cgh);
        auto acc1 = b1.template get_access<sycl::access::mode::write>(cgh);
        auto acc2 = b2.template get_access<sycl::access::mode::write>(cgh);
        auto accc = bc.template get_access<sycl::access::mode::write>(cgh);

        cgh.single_task<TestAlignment>([=](sycl::kernel_handler kh) {
          acc0[0] = kh.get_specialization_constant<char_spec0>();
          acc1[0] = kh.get_specialization_constant<char_spec1>();
          acc2[0] = kh.get_specialization_constant<char_spec2>();
          accc[0] = kh.get_specialization_constant<composite_spec>();
        });
      });
      q.wait_and_throw();
    }

    // Validate results
    bool pass = true;
    if (char_out0 != char_vals[0])
      pass = false;
    if (char_out1 != char_vals[1])
      pass = false;
    if (char_out2 != char_vals[2])
      pass = false;
    if (!(comp_out == set_comp))
      pass = false;

    if (!pass) {
      all_pass = false;
      std::cerr << "FAIL: permutation " << perm_index
                << " produced wrong values\n";
      std::cerr << " permutation order: ";
      for (int x : items)
        std::cerr << x << ' ';
      std::cerr << "\n  char_outs: " << char_out0 << ' ' << char_out1 << ' '
                << char_out2 << "\n";
      std::cerr << "  comp_out: f=" << comp_out.f << " i=" << comp_out.i
                << " c='" << comp_out.c << "'\n";
      break;
    }

  } while (std::next_permutation(items.begin(), items.end()));

  if (all_pass) {
    std::cout << "PASS: all permutations produced expected values\n";
    return 0;
  } else {
    std::cerr << "Some permutation failed - this likely indicates a bug"
                 "sensitive to ordering and placement of specialization"
                 "constants values in the specialization constants"
                 "buffer, like incorrect alignment assumptions when"
                 "loading/storing the values, missing padding etc.\n";
    return 1;
  }
}
