// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -fno-builtin -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: cuda || hip
#include "imf_utils.hpp"
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/math.hpp>

namespace s = sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

void run_vabs2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_vals[NUM] = {0,          1,          0x98324,    0xFFFFFFFF,
                              0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned output2_vals[NUM];
  unsigned output4_vals[NUM];
  unsigned ref2_vals[NUM] = {0,         1,          0x97CDC,    0x10001,
                             0x1225468, 0x12345678, 0x443466D9, 0x5F653E8A};

  unsigned ref4_vals[NUM] = {0,         1,          0x97D24,    0x1010101,
                             0x2225568, 0x12345678, 0x45346727, 0x60653f76};
  {
    s::buffer<unsigned, 1> output2_buf(&output2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output4_buf(&output4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output2_acc = output2_buf.get_access<sycl_write>(cgh);
          auto output4_acc = output4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vabs2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output2_acc[I[0]] =
                    sycl::ext::intel::math::vabs2(input_vals[I[0]]);
                output4_acc[I[0]] =
                    sycl::ext::intel::math::vabs4(input_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output2_vals[idx] != ref2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vabs2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output4_vals[idx] != ref4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vabs4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vabs2_4 test pass." << std::endl;
}

void run_vabsss2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_vals[NUM] = {0,          1,          0x98324,    0x07FF07FE,
                              0x80008000, 0x80808080, 0xBBCC9927, 0x80000001};
  unsigned output2_vals[NUM];
  unsigned output4_vals[NUM];
  unsigned ref2_vals[NUM] = {0,          1,          0x97CDC,    0x07FF07FE,
                             0x7FFF7FFF, 0x7F807F80, 0x443466D9, 0x7FFF0001};

  unsigned ref4_vals[NUM] = {0,          1,          0x97D24,    0x07010702,
                             0x7F007F00, 0x7F7F7F7F, 0x45346727, 0x7F000001};
  {
    s::buffer<unsigned, 1> output2_buf(&output2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output4_buf(&output4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output2_acc = output2_buf.get_access<sycl_write>(cgh);
          auto output4_acc = output4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vabsss2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output2_acc[I[0]] =
                    sycl::ext::intel::math::vabsss2(input_vals[I[0]]);
                output4_acc[I[0]] =
                    sycl::ext::intel::math::vabsss4(input_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output2_vals[idx] != ref2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vabsss2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output4_vals[idx] != ref4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vabsss4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vabsss2_4 test pass." << std::endl;
}

void run_vabsdiffsu2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          1,          0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          1,          0x8372833,  0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned output_s2_vals[NUM];
  unsigned output_s4_vals[NUM];
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned ref_s2_vals[NUM] = {0,          0,          0x82EA50F,  0x54006713,
                               0x56A77DFA, 0x556568F5, 0x21299E97, 0xD2E75091};
  unsigned ref_s4_vals[NUM] = {0,          0,          0x82EA50F,  0x54006713,
                               0x56597E06, 0x566569F5, 0x21299E69, 0xD319516F};
  unsigned ref_u2_vals[NUM] = {0,          0,          0x82E5AF1,  0x54006713,
                               0x56A78206, 0xAA9B970B, 0x21296169, 0x2D19AF6F};
  unsigned ref_u4_vals[NUM] = {0,          0,          0x82E5B0F,  0x54006713,
                               0x56A78206, 0xAA9B970B, 0x21296297, 0x2D19AF6F};

  {
    s::buffer<unsigned, 1> output_s2_buf(&output_s2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_s4_buf(&output_s4_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_s2_acc = output_s2_buf.get_access<sycl_write>(cgh);
          auto output_s4_acc = output_s4_buf.get_access<sycl_write>(cgh);
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vabsdiffsu2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_s2_acc[I[0]] = sycl::ext::intel::math::vabsdiffs2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_s4_acc[I[0]] = sycl::ext::intel::math::vabsdiffs4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u2_acc[I[0]] = sycl::ext::intel::math::vabsdiffu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] = sycl::ext::intel::math::vabsdiffu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s2_vals[idx] != ref_s2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vabsdiffs2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s4_vals[idx] != ref_s4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vabsdiffs4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vabsdiffu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vabsdiffu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vabsdiffsu_2_4 test pass." << std::endl;
}

void run_vadd_ss_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x6DDF66CC, 0x8372833,  0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned output_ss2_vals[NUM];
  unsigned output_ss4_vals[NUM];
  unsigned ref_u2_vals[NUM] = {0,          0xED5EE5CB, 0x840AB57,  0xABFE98EB,
                               0xA715D52A, 0xCF0343FB, 0x566FD0E5, 0x141DD37D};
  unsigned ref_u4_vals[NUM] = {0,          0xEC5EE4CB, 0x840AB57,  0xAAFE97EB,
                               0xA615D42A, 0xCE0343FB, 0x556FD0E5, 0x131DD37D};
  unsigned ref_ss2_vals[NUM] = {0,          0x7FFF7FFF, 0x840AB57,  0xABFE98EB,
                                0xA715D52A, 0xCF0343FB, 0x8000D0E5, 0x141DD37D};
  unsigned ref_ss4_vals[NUM] = {0,          0x7F5E7FCB, 0x840AB57,  0xAAFE97EB,
                                0xA615D480, 0xCE0343FB, 0x8080D0E5, 0x1380D37D};

  {
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_ss2_buf(&output_ss2_vals[0],
                                          s::range<1>(NUM));
    s::buffer<unsigned, 1> output_ss4_buf(&output_ss4_vals[0],
                                          s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          auto output_ss2_acc = output_ss2_buf.get_access<sycl_write>(cgh);
          auto output_ss4_acc = output_ss4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vadd2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_u2_acc[I[0]] = sycl::ext::intel::math::vadd2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] = sycl::ext::intel::math::vadd4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_ss2_acc[I[0]] = sycl::ext::intel::math::vaddss2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_ss4_acc[I[0]] = sycl::ext::intel::math::vaddss4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vadd2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vadd4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_ss2_vals[idx] != ref_ss2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vaddss2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_ss4_vals[idx] != ref_ss4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vaddss4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vadd_2_4 test pass." << std::endl;
}

void run_vadd_us_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x6DDF66CC, 0x8372833,  0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned ref_u2_vals[NUM] = {0,          0xED5EE5CB, 0x840AB57,  0xFFFFFFFF,
                               0xFFFFD52A, 0xCF03FFFF, 0xFFFFD0E5, 0xFFFFD37D};
  unsigned ref_u4_vals[NUM] = {0,          0xECFFE4FF, 0x840AB57,  0xFFFFFFFF,
                               0xFFFFD4FF, 0xCEFFFFFB, 0xFFFFD0E5, 0xFFFFD37D};

  {
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vaddus2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_u2_acc[I[0]] = sycl::ext::intel::math::vaddus2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] = sycl::ext::intel::math::vaddus4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vaddus2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vaddus4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vaddus_2_4 test pass." << std::endl;
}

void run_vhaddu_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x6DDF66CC, 0x8372833,  0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned ref_u2_vals[NUM] = {0,          0x76AF72E5, 0x42055AB,  0xD5FFCC75,
                               0xD38A6A95, 0x6781A1FD, 0xAB376872, 0x8A0E69BE};
  unsigned ref_u4_vals[NUM] = {0,          0x76AF72E5, 0x420552B,  0xD5FFCBF5,
                               0xD38A6A95, 0x6781A17D, 0xAAB76872, 0x898E693E};

  {
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vhaddu_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_u2_acc[I[0]] = sycl::ext::intel::math::vhaddu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] = sycl::ext::intel::math::vhaddu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vhaddu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vhaddu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vhaddu_2_4 test pass." << std::endl;
}

void run_vsub_ss_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x6DDF66CC, 0x8372833,  0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned output_ss2_vals[NUM];
  unsigned output_ss4_vals[NUM];
  unsigned ref_u2_vals[NUM] = {0,          0x11A01833, 0xF7D25AF1, 0x54006713,
                               0x56A78206, 0x556568F5, 0x21296169, 0x2D19AF6F};
  unsigned ref_u4_vals[NUM] = {0,          0x12A01833, 0xF8D25BF1, 0x54006713,
                               0x56A78206, 0x566569F5, 0x21296269, 0x2D19AF6F};
  unsigned ref_ss2_vals[NUM] = {0,          0x11A01833, 0xF7D28000, 0x54006713,
                                0x56A78206, 0x556568F5, 0x21298000, 0x8000AF6F};
  unsigned ref_ss4_vals[NUM] = {0,          0x127F1833, 0xF8D280F1, 0x54006713,
                                0x56A78206, 0x5665697F, 0x21298069, 0x8019AF6F};

  {
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_ss2_buf(&output_ss2_vals[0],
                                          s::range<1>(NUM));
    s::buffer<unsigned, 1> output_ss4_buf(&output_ss4_vals[0],
                                          s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          auto output_ss2_acc = output_ss2_buf.get_access<sycl_write>(cgh);
          auto output_ss4_acc = output_ss4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vsub2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_u2_acc[I[0]] = sycl::ext::intel::math::vsub2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] = sycl::ext::intel::math::vsub4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_ss2_acc[I[0]] = sycl::ext::intel::math::vsubss2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_ss4_acc[I[0]] = sycl::ext::intel::math::vsubss4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsub2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsub4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_ss2_vals[idx] != ref_ss2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsubss2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_ss4_vals[idx] != ref_ss4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsubss4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vsub_2_4 test pass." << std::endl;
}

void run_vsub_us_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x6DDF66CC, 0x8372833,  0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned ref_u2_vals[NUM] = {0,          0x11A01833, 0x5AF1,     0x54006713,
                               0x56A78206, 0x0,        0x21296169, 0x2D19AF6F};
  unsigned ref_u4_vals[NUM] = {0,          0x12001833, 0x5B00,     0x54006713,
                               0x56A78206, 0x0,        0x21296200, 0x2D19AF6F};

  {
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vsubus2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_u2_acc[I[0]] = sycl::ext::intel::math::vsubus2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] = sycl::ext::intel::math::vsubus4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsubus2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsubus4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vsubus_2_4 test pass." << std::endl;
}

void run_vavgs_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324017, 0xFFFFFFFF,
                                0xFEDEAB98, 0x829C5678, 0x80007FFF, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x7DDF56CC, 0x83728331, 0xABFF98EC,
                                0xA8372992, 0x8A6CED83, 0x800037BE, 0x73821207};
  unsigned output_s2_vals[NUM];
  unsigned output_s4_vals[NUM];
  unsigned ref_s2_vals[NUM] = {0,          0x7EAF6AE6, 0x8DD2E1A4, 0xD5FFCC75,
                               0xD38AEA95, 0x868421FE, 0x80005BDF, 0xA0FE9BE};
  unsigned ref_s4_vals[NUM] = {0,          0x7E2F6AE5, 0x8D52E124, 0xD5FFCBF5,
                               0xD30BEA95, 0x860422FD, 0x80005BDE, 0xA8EE93F};

  {
    s::buffer<unsigned, 1> output_s2_buf(&output_s2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_s4_buf(&output_s4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_s2_acc = output_s2_buf.get_access<sycl_write>(cgh);
          auto output_s4_acc = output_s4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vavgs_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_s2_acc[I[0]] = sycl::ext::intel::math::vavgs2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_s4_acc[I[0]] = sycl::ext::intel::math::vavgs4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s2_vals[idx] != ref_s2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vavgs2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s4_vals[idx] != ref_s4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vavgs4 failed! idx = " << idx
                << std::endl;
      std::cout << std::hex << output_s4_vals[idx] << "  " << ref_s4_vals[idx]
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vavgs_2_4 test pass." << std::endl;
}

void run_vavgu_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324017, 0xFFFFFFFF,
                                0xFEDEAB98, 0x829C5678, 0x80007FFF, 0xA09BC176};
  unsigned input_y_vals[NUM] = {1,          0x7DDF56CC, 0x83728331, 0xABFF98EC,
                                0xA8372992, 0x8A6CED83, 0x800037BE, 0x73821207};
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned ref_u2_vals[NUM] = {1,          0x7EAF6AE6, 0x8DD261A4, 0xD5FFCC76,
                               0xD38B6A95, 0x8684A1FE, 0x80005BDF, 0x8A0F69BF};
  unsigned ref_u4_vals[NUM] = {1,          0x7EAF6AE6, 0x8E526224, 0xD5FFCCF6,
                               0xD38B6A95, 0x8684A27E, 0x80005BDF, 0x8A8F6A3F};

  {
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vavgu_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_u2_acc[I[0]] = sycl::ext::intel::math::vavgu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] = sycl::ext::intel::math::vavgu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vavgu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vavgu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vavgu_2_4 test pass." << std::endl;
}

void run_vcmpgts_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324017, 0xFFFFFFFF,
                                0xFEDEAB98, 0x829C5678, 0x80007FFF, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x7DDF56CC, 0x98338331, 0xABFF98EC,
                                0xA8372992, 0x8A6CED83, 0x800037BE, 0x73821207};
  unsigned output_s2_vals[NUM];
  unsigned output_s4_vals[NUM];
  unsigned ref_s2_vals[NUM] = {0,          0xFFFFFFFF, 0x0000FFFF, 0xFFFFFFFF,
                               0xFFFF0000, 0x0000FFFF, 0x0000FFFF, 0};
  unsigned ref_s4_vals[NUM] = {0,          0xFFFFFFFF, 0x0000FF00, 0xFF00FFFF,
                               0xFF0000FF, 0x0000FFFF, 0x0000FFFF, 0x00FF00FF};

  {
    s::buffer<unsigned, 1> output_s2_buf(&output_s2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_s4_buf(&output_s4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_s2_acc = output_s2_buf.get_access<sycl_write>(cgh);
          auto output_s4_acc = output_s4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vcmpgts_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_s2_acc[I[0]] = sycl::ext::intel::math::vcmpgts2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_s4_acc[I[0]] = sycl::ext::intel::math::vcmpgts4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s2_vals[idx] != ref_s2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpgts2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s4_vals[idx] != ref_s4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpgts4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vcmpgts_2_4 test pass." << std::endl;
}

void run_vmaxmin_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          1,          0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          1,          0x83712833, 0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned max_output_s2_vals[NUM];
  unsigned max_output_s4_vals[NUM];
  unsigned max_output_u2_vals[NUM];
  unsigned max_output_u4_vals[NUM];
  unsigned min_output_s2_vals[NUM];
  unsigned min_output_s4_vals[NUM];
  unsigned min_output_u2_vals[NUM];
  unsigned min_output_u4_vals[NUM];
  unsigned max_ref_s2_vals[NUM] = {0,          1,          0x92833,
                                   0xFFFFFFFF, 0xFEDE2992, 0x12345678,
                                   0xBBCC37BE, 0x73821207};
  unsigned max_ref_s4_vals[NUM] = {0,          1,          0x712833,
                                   0xFFFFFFFF, 0xFE372998, 0x12345678,
                                   0xBBCC3727, 0x739B1276};
  unsigned max_ref_u2_vals[NUM] = {0,          1,          0x83718324,
                                   0xFFFFFFFF, 0xFEDEAB98, 0xBCCFED83,
                                   0xBBCC9927, 0xA09BC176};
  unsigned max_ref_u4_vals[NUM] = {0,          1,          0x83718333,
                                   0xFFFFFFFF, 0xFEDEAB98, 0xBCCFED83,
                                   0xBBCC99BE, 0xA09BC176};

  unsigned min_ref_s2_vals[NUM] = {0,          1,          0x83718324,
                                   0xABFF98EC, 0xA837Ab98, 0xBCCFED83,
                                   0x9AA39927, 0xA09BC176};
  unsigned min_ref_s4_vals[NUM] = {0,          1,          0x83098324,
                                   0xABFF98EC, 0xA8DEAB92, 0xBCCFED83,
                                   0x9AA399BE, 0xA082C107};
  unsigned min_ref_u2_vals[NUM] = {0,          1,          0x92833,
                                   0xABFF98EC, 0xA8372992, 0x12345678,
                                   0x9AA337BE, 0x73821207};
  unsigned min_ref_u4_vals[NUM] = {0,          1,          0x92824,
                                   0xABFF98EC, 0xA8372992, 0x12345678,
                                   0x9AA33727, 0x73821207};

  {
    s::buffer<unsigned, 1> max_output_s2_buf(&max_output_s2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> max_output_s4_buf(&max_output_s4_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> max_output_u2_buf(&max_output_u2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> max_output_u4_buf(&max_output_u4_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> min_output_s2_buf(&min_output_s2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> min_output_s4_buf(&min_output_s4_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> min_output_u2_buf(&min_output_u2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> min_output_u4_buf(&min_output_u4_vals[0],
                                             s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto max_output_s2_acc =
              max_output_s2_buf.get_access<sycl_write>(cgh);
          auto max_output_s4_acc =
              max_output_s4_buf.get_access<sycl_write>(cgh);
          auto max_output_u2_acc =
              max_output_u2_buf.get_access<sycl_write>(cgh);
          auto max_output_u4_acc =
              max_output_u4_buf.get_access<sycl_write>(cgh);
          auto min_output_s2_acc =
              min_output_s2_buf.get_access<sycl_write>(cgh);
          auto min_output_s4_acc =
              min_output_s4_buf.get_access<sycl_write>(cgh);
          auto min_output_u2_acc =
              min_output_u2_buf.get_access<sycl_write>(cgh);
          auto min_output_u4_acc =
              min_output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vmaxmin_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                max_output_s2_acc[I[0]] = sycl::ext::intel::math::vmaxs2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                max_output_s4_acc[I[0]] = sycl::ext::intel::math::vmaxs4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                max_output_u2_acc[I[0]] = sycl::ext::intel::math::vmaxu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                max_output_u4_acc[I[0]] = sycl::ext::intel::math::vmaxu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                min_output_s2_acc[I[0]] = sycl::ext::intel::math::vmins2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                min_output_s4_acc[I[0]] = sycl::ext::intel::math::vmins4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                min_output_u2_acc[I[0]] = sycl::ext::intel::math::vminu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                min_output_u4_acc[I[0]] = sycl::ext::intel::math::vminu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (max_output_s2_vals[idx] != max_ref_s2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vmaxs2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (max_output_s4_vals[idx] != max_ref_s4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vmaxs4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (max_output_u2_vals[idx] != max_ref_u2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vmaxu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (max_output_u4_vals[idx] != max_ref_u4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vmaxu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (min_output_s2_vals[idx] != min_ref_s2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vmins2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (min_output_s4_vals[idx] != min_ref_s4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vmins4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (min_output_u2_vals[idx] != min_ref_u2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vminu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (min_output_u4_vals[idx] != min_ref_u4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vminu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vmaxmin_2_4 test pass." << std::endl;
}

void run_vneg_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_vals[NUM] = {0,          1,          0x98324,    0xFFFFFFFF,
                              0x1345AB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned output2_vals[NUM];
  unsigned output4_vals[NUM];
  unsigned ref2_vals[NUM] = {0,          0xFFFF,     0xFFF77CDC, 0x10001,
                             0xECBB5468, 0xEDCCA988, 0x443466D9, 0x5F653E8A};

  unsigned ref4_vals[NUM] = {0,          0xFF,       0xF77DDC,   0x1010101,
                             0xEDBB5568, 0xEECCAA88, 0x453467D9, 0x60653F8A};
  {
    s::buffer<unsigned, 1> output2_buf(&output2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output4_buf(&output4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output2_acc = output2_buf.get_access<sycl_write>(cgh);
          auto output4_acc = output4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vneg_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output2_acc[I[0]] =
                    sycl::ext::intel::math::vneg2(input_vals[I[0]]);
                output4_acc[I[0]] =
                    sycl::ext::intel::math::vneg4(input_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output2_vals[idx] != ref2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vneg2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output4_vals[idx] != ref4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vneg4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vneg_2_4 test pass." << std::endl;
}

void run_vnegss_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 4;
  unsigned input_vals[NUM] = {1, 0x98324, 0x7FFF8000, 0x7F801127};
  unsigned output2_vals[NUM];
  unsigned output4_vals[NUM];
  unsigned ref2_vals[NUM] = {0xFFFF, 0xFFF77CDC, 0x80017FFF, 0x8080EED9};
  unsigned ref4_vals[NUM] = {0xFF, 0xF77DDC, 0x81017F00, 0x817FEFD9};
  {
    s::buffer<unsigned, 1> output2_buf(&output2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output4_buf(&output4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output2_acc = output2_buf.get_access<sycl_write>(cgh);
          auto output4_acc = output4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vnegss_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output2_acc[I[0]] =
                    sycl::ext::intel::math::vnegss2(input_vals[I[0]]);
                output4_acc[I[0]] =
                    sycl::ext::intel::math::vnegss4(input_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output2_vals[idx] != ref2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vnegss2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output4_vals[idx] != ref4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vnegss4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vnegss_2_4 test pass." << std::endl;
}

void run_vsad_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0xFFFF,     0x76498324, 0x80008000,
                                0x7F6F5F4F, 0xFFFFFFFF, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0x7F7F8000, 0x12458964, 0x34212833, 0x7FFF7FFF,
                                0x80818283, 0x0,        0x7FFF6883, 0x12345678};
  unsigned output_s2_vals[NUM];
  unsigned output_s4_vals[NUM];
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned ref_s2_vals[NUM] = {0xFF7F,  0x88E0, 0xE737,  0x1FFFE,
                               0x1DBBA, 0x2,    0x1938F, 0x1069B};
  unsigned ref_s4_vals[NUM] = {0x17E, 0x132, 0x11E, 0x200,
                               0x396, 0x4,   0x26A, 0x1A2};
  unsigned ref_u2_vals[NUM] = {0xFF7F, 0x88E0,  0x9D19, 0x2,
                               0x2446, 0x1FFFE, 0x6C71, 0xF965};
  unsigned ref_u4_vals[NUM] = {0x17E, 0x168, 0xD4, 0x200,
                               0x6A,  0x3FC, 0xFC, 0x162};

  {
    s::buffer<unsigned, 1> output_s2_buf(&output_s2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_s4_buf(&output_s4_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_s2_acc = output_s2_buf.get_access<sycl_write>(cgh);
          auto output_s4_acc = output_s4_buf.get_access<sycl_write>(cgh);
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vsad_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_s2_acc[I[0]] = sycl::ext::intel::math::vsads2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_s4_acc[I[0]] = sycl::ext::intel::math::vsads4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u2_acc[I[0]] = sycl::ext::intel::math::vsadu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] = sycl::ext::intel::math::vsadu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s2_vals[idx] != ref_s2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsads2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s4_vals[idx] != ref_s4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsads4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsadu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsadu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vsad_2_4 test pass." << std::endl;
}

void run_veqne_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 5;
  unsigned input_x_vals[NUM] = {0x1, 0xFEDE1122, 0x12345678, 0xFFFFFFFF,
                                0x98765432};
  unsigned input_y_vals[NUM] = {0x2, 0xFEDF1133, 0x12345679, 0xFFFFFFFF,
                                0x87654321};
  unsigned output_cmpeq2_vals[NUM];
  unsigned output_cmpeq4_vals[NUM];
  unsigned output_cmpne2_vals[NUM];
  unsigned output_cmpne4_vals[NUM];
  unsigned output_seteq2_vals[NUM];
  unsigned output_seteq4_vals[NUM];
  unsigned output_setne2_vals[NUM];
  unsigned output_setne4_vals[NUM];
  unsigned cmpeq2_ref_vals[NUM] = {0xFFFF0000, 0, 0xFFFF0000, 0xFFFFFFFF, 0};
  unsigned cmpeq4_ref_vals[NUM] = {0xFFFFFF00, 0xFF00FF00, 0xFFFFFF00,
                                   0xFFFFFFFF, 0};
  unsigned cmpne2_ref_vals[NUM] = {0xFFFF, 0xFFFFFFFF, 0xFFFF, 0, 0xFFFFFFFF};
  unsigned cmpne4_ref_vals[NUM] = {0xFF, 0xFF00FF, 0xFF, 0, 0xFFFFFFFF};

  unsigned seteq2_ref_vals[NUM] = {0x10000, 0, 0x10000, 0x10001, 0};
  unsigned seteq4_ref_vals[NUM] = {0x1010100, 0x1000100, 0x1010100, 0x1010101,
                                   0};
  unsigned setne2_ref_vals[NUM] = {1, 0x10001, 1, 0, 0x10001};
  unsigned setne4_ref_vals[NUM] = {1, 0x10001, 1, 0x0, 0x1010101};

  {
    s::buffer<unsigned, 1> output_cmpeq2_buf(&output_cmpeq2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpeq4_buf(&output_cmpeq4_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpne2_buf(&output_cmpne2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpne4_buf(&output_cmpne4_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_seteq2_buf(&output_seteq2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_seteq4_buf(&output_seteq4_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setne2_buf(&output_setne2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setne4_buf(&output_setne4_vals[0],
                                             s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_cmpeq2_acc =
              output_cmpeq2_buf.get_access<sycl_write>(cgh);
          auto output_cmpeq4_acc =
              output_cmpeq4_buf.get_access<sycl_write>(cgh);
          auto output_cmpne2_acc =
              output_cmpne2_buf.get_access<sycl_write>(cgh);
          auto output_cmpne4_acc =
              output_cmpne4_buf.get_access<sycl_write>(cgh);
          auto output_seteq2_acc =
              output_seteq2_buf.get_access<sycl_write>(cgh);
          auto output_seteq4_acc =
              output_seteq4_buf.get_access<sycl_write>(cgh);
          auto output_setne2_acc =
              output_setne2_buf.get_access<sycl_write>(cgh);
          auto output_setne4_acc =
              output_setne4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class veqne_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_cmpeq2_acc[I[0]] = sycl::ext::intel::math::vcmpeq2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpeq4_acc[I[0]] = sycl::ext::intel::math::vcmpeq4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpne2_acc[I[0]] = sycl::ext::intel::math::vcmpne2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpne4_acc[I[0]] = sycl::ext::intel::math::vcmpne4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_seteq2_acc[I[0]] = sycl::ext::intel::math::vseteq2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_seteq4_acc[I[0]] = sycl::ext::intel::math::vseteq4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setne2_acc[I[0]] = sycl::ext::intel::math::vsetne2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setne4_acc[I[0]] = sycl::ext::intel::math::vsetne4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpeq2_vals[idx] != cmpeq2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpeq2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpeq4_vals[idx] != cmpeq4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpeq4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpne2_vals[idx] != cmpne2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpne2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpne4_vals[idx] != cmpne4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpne4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_seteq2_vals[idx] != seteq2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vseteq2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_seteq4_vals[idx] != seteq4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vseteq4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setne2_vals[idx] != setne2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetne2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setne4_vals[idx] != setne4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetne4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::veqne_2_4 test pass." << std::endl;
}

void run_vgelt_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 5;
  unsigned input_x_vals[NUM] = {0x1, 0xFEDE1122, 0x12345678, 0xFFFFFFFF,
                                0x98765432};
  unsigned input_y_vals[NUM] = {0x2, 0xFEDF1133, 0x12345679, 0xFFFFFFFF,
                                0x87654321};
  unsigned output_cmpges2_vals[NUM];
  unsigned output_cmpges4_vals[NUM];
  unsigned output_cmplts2_vals[NUM];
  unsigned output_cmplts4_vals[NUM];
  unsigned output_cmpgeu2_vals[NUM];
  unsigned output_cmpgeu4_vals[NUM];
  unsigned output_cmpltu2_vals[NUM];
  unsigned output_cmpltu4_vals[NUM];
  unsigned output_setges2_vals[NUM];
  unsigned output_setges4_vals[NUM];
  unsigned output_setlts2_vals[NUM];
  unsigned output_setlts4_vals[NUM];
  unsigned output_setgeu2_vals[NUM];
  unsigned output_setgeu4_vals[NUM];
  unsigned output_setltu2_vals[NUM];
  unsigned output_setltu4_vals[NUM];
  unsigned cmpges2_ref_vals[NUM] = {0xFFFF0000, 0, 0xFFFF0000, 0xFFFFFFFF,
                                    0xFFFFFFFF};
  unsigned cmpges4_ref_vals[NUM] = {0xFFFFFF00, 0xFF00FF00, 0xFFFFFF00,
                                    0xFFFFFFFF, 0xFFFFFFFF};
  unsigned cmplts2_ref_vals[NUM] = {0xFFFF, 0xFFFFFFFF, 0xFFFF, 0, 0};
  unsigned cmplts4_ref_vals[NUM] = {0xFF, 0xFF00FF, 0xFF, 0, 0};

  unsigned cmpgeu2_ref_vals[NUM] = {0xFFFF0000, 0, 0xFFFF0000, 0xFFFFFFFF,
                                    0xFFFFFFFF};
  unsigned cmpgeu4_ref_vals[NUM] = {0xFFFFFF00, 0xFF00FF00, 0xFFFFFF00,
                                    0xFFFFFFFF, 0xFFFFFFFF};
  unsigned cmpltu2_ref_vals[NUM] = {0xFFFF, 0xFFFFFFFF, 0xFFFF, 0, 0};
  unsigned cmpltu4_ref_vals[NUM] = {0xFF, 0xFF00FF, 0xFF, 0, 0};
  unsigned setges2_ref_vals[NUM] = {0x10000, 0, 0x10000, 0x10001, 0x10001};
  unsigned setges4_ref_vals[NUM] = {0x1010100, 0x1000100, 0x1010100, 0x1010101,
                                    0x1010101};
  unsigned setlts2_ref_vals[NUM] = {1, 0x10001, 1, 0, 0};
  unsigned setlts4_ref_vals[NUM] = {1, 0x10001, 1, 0, 0};
  unsigned setgeu2_ref_vals[NUM] = {0x10000, 0, 0x10000, 0x10001, 0x10001};
  unsigned setgeu4_ref_vals[NUM] = {0x1010100, 0x1000100, 0x1010100, 0x1010101,
                                    0x1010101};
  unsigned setltu2_ref_vals[NUM] = {1, 0x10001, 1, 0, 0};
  unsigned setltu4_ref_vals[NUM] = {1, 0x10001, 1, 0, 0};

  {
    s::buffer<unsigned, 1> output_cmpges2_buf(&output_cmpges2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpges4_buf(&output_cmpges4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmplts2_buf(&output_cmplts2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmplts4_buf(&output_cmplts4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpgeu2_buf(&output_cmpgeu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpgeu4_buf(&output_cmpgeu4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpltu2_buf(&output_cmpltu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpltu4_buf(&output_cmpltu4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setges2_buf(&output_setges2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setges4_buf(&output_setges4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setlts2_buf(&output_setlts2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setlts4_buf(&output_setlts4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setgeu2_buf(&output_setgeu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setgeu4_buf(&output_setgeu4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setltu2_buf(&output_setltu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setltu4_buf(&output_setltu4_vals[0],
                                              s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_cmpges2_acc =
              output_cmpges2_buf.get_access<sycl_write>(cgh);
          auto output_cmpges4_acc =
              output_cmpges4_buf.get_access<sycl_write>(cgh);
          auto output_cmplts2_acc =
              output_cmplts2_buf.get_access<sycl_write>(cgh);
          auto output_cmplts4_acc =
              output_cmplts4_buf.get_access<sycl_write>(cgh);
          auto output_cmpgeu2_acc =
              output_cmpgeu2_buf.get_access<sycl_write>(cgh);
          auto output_cmpgeu4_acc =
              output_cmpgeu4_buf.get_access<sycl_write>(cgh);
          auto output_cmpltu2_acc =
              output_cmpltu2_buf.get_access<sycl_write>(cgh);
          auto output_cmpltu4_acc =
              output_cmpltu4_buf.get_access<sycl_write>(cgh);
          auto output_setges2_acc =
              output_setges2_buf.get_access<sycl_write>(cgh);
          auto output_setges4_acc =
              output_setges4_buf.get_access<sycl_write>(cgh);
          auto output_setlts2_acc =
              output_setlts2_buf.get_access<sycl_write>(cgh);
          auto output_setlts4_acc =
              output_setlts4_buf.get_access<sycl_write>(cgh);
          auto output_setgeu2_acc =
              output_setgeu2_buf.get_access<sycl_write>(cgh);
          auto output_setgeu4_acc =
              output_setgeu4_buf.get_access<sycl_write>(cgh);
          auto output_setltu2_acc =
              output_setltu2_buf.get_access<sycl_write>(cgh);
          auto output_setltu4_acc =
              output_setltu4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vgelt_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_cmpges2_acc[I[0]] = sycl::ext::intel::math::vcmpges2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpges4_acc[I[0]] = sycl::ext::intel::math::vcmpges4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmplts2_acc[I[0]] = sycl::ext::intel::math::vcmplts2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmplts4_acc[I[0]] = sycl::ext::intel::math::vcmplts4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpgeu2_acc[I[0]] = sycl::ext::intel::math::vcmpgeu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpgeu4_acc[I[0]] = sycl::ext::intel::math::vcmpgeu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpltu2_acc[I[0]] = sycl::ext::intel::math::vcmpltu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpltu4_acc[I[0]] = sycl::ext::intel::math::vcmpltu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setges2_acc[I[0]] = sycl::ext::intel::math::vsetges2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setges4_acc[I[0]] = sycl::ext::intel::math::vsetges4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setlts2_acc[I[0]] = sycl::ext::intel::math::vsetlts2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setlts4_acc[I[0]] = sycl::ext::intel::math::vsetlts4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setgeu2_acc[I[0]] = sycl::ext::intel::math::vsetgeu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setgeu4_acc[I[0]] = sycl::ext::intel::math::vsetgeu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setltu2_acc[I[0]] = sycl::ext::intel::math::vsetltu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setltu4_acc[I[0]] = sycl::ext::intel::math::vsetltu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpges2_vals[idx] != cmpges2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpges2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpges4_vals[idx] != cmpges4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpges4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmplts2_vals[idx] != cmplts2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmplts2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmplts4_vals[idx] != cmplts4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmplts4 failed! idx = " << idx
                << std::endl;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpgeu2_vals[idx] != cmpgeu2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpgeu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpgeu4_vals[idx] != cmpgeu4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpgeu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpltu2_vals[idx] != cmpltu2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpltu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpltu4_vals[idx] != cmpltu4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpltu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setges2_vals[idx] != setges2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetges2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setges4_vals[idx] != setges4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetges4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setlts2_vals[idx] != setlts2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetlts2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setlts4_vals[idx] != setlts4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetlts4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setgeu2_vals[idx] != setgeu2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetgeu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setgeu4_vals[idx] != setgeu4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetgeu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setltu2_vals[idx] != setltu2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetltu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setltu4_vals[idx] != setltu4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetltu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vgelt_2_4 test pass." << std::endl;
}

void run_vgtle_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 5;
  unsigned input_x_vals[NUM] = {0x1, 0xFEDE1122, 0x12345678, 0xFFFFFFFF,
                                0x98765432};
  unsigned input_y_vals[NUM] = {0x2, 0xFEDF1133, 0x12345679, 0xFFFFFFFF,
                                0x87654321};
  unsigned output_cmpgts2_vals[NUM];
  unsigned output_cmpgts4_vals[NUM];
  unsigned output_cmples2_vals[NUM];
  unsigned output_cmples4_vals[NUM];
  unsigned output_cmpgtu2_vals[NUM];
  unsigned output_cmpgtu4_vals[NUM];
  unsigned output_cmpleu2_vals[NUM];
  unsigned output_cmpleu4_vals[NUM];
  unsigned output_setgts2_vals[NUM];
  unsigned output_setgts4_vals[NUM];
  unsigned output_setles2_vals[NUM];
  unsigned output_setles4_vals[NUM];
  unsigned output_setgtu2_vals[NUM];
  unsigned output_setgtu4_vals[NUM];
  unsigned output_setleu2_vals[NUM];
  unsigned output_setleu4_vals[NUM];
  unsigned cmpgts2_ref_vals[NUM] = {0, 0, 0, 0, 0xFFFFFFFF};
  unsigned cmpgts4_ref_vals[NUM] = {0, 0, 0, 0, 0xFFFFFFFF};
  unsigned cmples2_ref_vals[NUM] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                    0xFFFFFFFF, 0};
  unsigned cmples4_ref_vals[NUM] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                    0xFFFFFFFF, 0};

  unsigned cmpgtu2_ref_vals[NUM] = {0, 0, 0, 0, 0xFFFFFFFF};
  unsigned cmpgtu4_ref_vals[NUM] = {0, 0, 0, 0, 0xFFFFFFFF};
  unsigned cmpleu2_ref_vals[NUM] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                    0xFFFFFFFF, 0};
  unsigned cmpleu4_ref_vals[NUM] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                    0xFFFFFFFF, 0};
  unsigned setgts2_ref_vals[NUM] = {0, 0, 0, 0, 0x10001};
  unsigned setgts4_ref_vals[NUM] = {0, 0, 0, 0, 0x1010101};
  unsigned setles2_ref_vals[NUM] = {0x10001, 0x10001, 0x10001, 0x10001, 0};
  unsigned setles4_ref_vals[NUM] = {0x1010101, 0x1010101, 0x1010101, 0x1010101,
                                    0};
  unsigned setgtu2_ref_vals[NUM] = {0, 0, 0, 0, 0x10001};
  unsigned setgtu4_ref_vals[NUM] = {0, 0, 0, 0, 0x1010101};
  unsigned setleu2_ref_vals[NUM] = {0x10001, 0x10001, 0x10001, 0x10001, 0};
  unsigned setleu4_ref_vals[NUM] = {0x1010101, 0x1010101, 0x1010101, 0x1010101,
                                    0};

  {
    s::buffer<unsigned, 1> output_cmpgts2_buf(&output_cmpgts2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpgts4_buf(&output_cmpgts4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmples2_buf(&output_cmples2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmples4_buf(&output_cmples4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpgtu2_buf(&output_cmpgtu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpgtu4_buf(&output_cmpgtu4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpleu2_buf(&output_cmpleu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpleu4_buf(&output_cmpleu4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setgts2_buf(&output_setgts2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setgts4_buf(&output_setgts4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setles2_buf(&output_setles2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setles4_buf(&output_setles4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setgtu2_buf(&output_setgtu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setgtu4_buf(&output_setgtu4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setleu2_buf(&output_setleu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setleu4_buf(&output_setleu4_vals[0],
                                              s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_cmpgts2_acc =
              output_cmpgts2_buf.get_access<sycl_write>(cgh);
          auto output_cmpgts4_acc =
              output_cmpgts4_buf.get_access<sycl_write>(cgh);
          auto output_cmples2_acc =
              output_cmples2_buf.get_access<sycl_write>(cgh);
          auto output_cmples4_acc =
              output_cmples4_buf.get_access<sycl_write>(cgh);
          auto output_cmpgtu2_acc =
              output_cmpgtu2_buf.get_access<sycl_write>(cgh);
          auto output_cmpgtu4_acc =
              output_cmpgtu4_buf.get_access<sycl_write>(cgh);
          auto output_cmpleu2_acc =
              output_cmpleu2_buf.get_access<sycl_write>(cgh);
          auto output_cmpleu4_acc =
              output_cmpleu4_buf.get_access<sycl_write>(cgh);
          auto output_setgts2_acc =
              output_setgts2_buf.get_access<sycl_write>(cgh);
          auto output_setgts4_acc =
              output_setgts4_buf.get_access<sycl_write>(cgh);
          auto output_setles2_acc =
              output_setles2_buf.get_access<sycl_write>(cgh);
          auto output_setles4_acc =
              output_setles4_buf.get_access<sycl_write>(cgh);
          auto output_setgtu2_acc =
              output_setgtu2_buf.get_access<sycl_write>(cgh);
          auto output_setgtu4_acc =
              output_setgtu4_buf.get_access<sycl_write>(cgh);
          auto output_setleu2_acc =
              output_setleu2_buf.get_access<sycl_write>(cgh);
          auto output_setleu4_acc =
              output_setleu4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vgtle_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_cmpgts2_acc[I[0]] = sycl::ext::intel::math::vcmpgts2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpgts4_acc[I[0]] = sycl::ext::intel::math::vcmpgts4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmples2_acc[I[0]] = sycl::ext::intel::math::vcmples2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmples4_acc[I[0]] = sycl::ext::intel::math::vcmples4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpgtu2_acc[I[0]] = sycl::ext::intel::math::vcmpgtu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpgtu4_acc[I[0]] = sycl::ext::intel::math::vcmpgtu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpleu2_acc[I[0]] = sycl::ext::intel::math::vcmpleu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpleu4_acc[I[0]] = sycl::ext::intel::math::vcmpleu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setgts2_acc[I[0]] = sycl::ext::intel::math::vsetgts2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setgts4_acc[I[0]] = sycl::ext::intel::math::vsetgts4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setles2_acc[I[0]] = sycl::ext::intel::math::vsetles2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setles4_acc[I[0]] = sycl::ext::intel::math::vsetles4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setgtu2_acc[I[0]] = sycl::ext::intel::math::vsetgtu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setgtu4_acc[I[0]] = sycl::ext::intel::math::vsetgtu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setleu2_acc[I[0]] = sycl::ext::intel::math::vsetleu2(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setleu4_acc[I[0]] = sycl::ext::intel::math::vsetleu4(
                    input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpgts2_vals[idx] != cmpgts2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpgts2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpgts4_vals[idx] != cmpgts4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpgts4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmples2_vals[idx] != cmples2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmples2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmples4_vals[idx] != cmples4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmples4 failed! idx = " << idx
                << std::endl;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpgtu2_vals[idx] != cmpgtu2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpgtu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpgtu4_vals[idx] != cmpgtu4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpgtu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpleu2_vals[idx] != cmpleu2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpleu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpleu4_vals[idx] != cmpleu4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vcmpleu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setgts2_vals[idx] != setgts2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetgts2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setgts4_vals[idx] != setgts4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetgts4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setles2_vals[idx] != setles2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetles2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setles4_vals[idx] != setles4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetles4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setgtu2_vals[idx] != setgtu2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetgtu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setgtu4_vals[idx] != setgtu4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetgtu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setleu2_vals[idx] != setleu2_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetleu2 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setleu4_vals[idx] != setleu4_ref_vals[idx]) {
      std::cout << "sycl::ext::intel::math::vsetleu4 failed! idx = " << idx
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "sycl::ext::intel::math::vgtle_2_4 test pass." << std::endl;
}

void run_viaddmax_s16x2_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0, 0xffd23211, 0x6b8b4567, 0x66334873, 0x2eb141f2, 0x71f32454, 0xb03e0c6};
  std::initializer_list<unsigned> input_vals2 = {
      0, 0x99233, 0x327b23c6, 0x74b0dc51, 0x41b71efb, 0x2ca88611, 0x189a769b,
  };
  std::initializer_list<unsigned> input_vals3 = {
      0, 0x81ffff77, 0x643c9869, 0x19495cff, 0x79e2a9e3, 0x836c40e, 0x54e49eb4};
  std::initializer_list<unsigned> ref_vals = {
      0, 0xffdbff77, 0x643c692d, 0x19495cff, 0x79e260ed, 0x836c40e, 0x54e45761};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::viaddmax_s16x2));
  std::cout << "sycl::ext::intel::math::viaddmax_s16x2 test pass." << std::endl;
}

void run_viaddmax_s16x2_relu_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x6b8b4567, 0x66334873, 0x2ae8944a, 0x46e87ccd, 0x2eb141f2,
      0x7545e146, 0x12200854, 0x1f16e9e8, 0x140e0f76, 0xded7263,  0x41a7c4c9,
      0x25e45d32, 0x3f2dba31, 0x62bbd95a, 0x333ab105, 0x2d1d5ae9, 0x8edbdab,
      0xb03e0c6,  0x71f32454, 0x2901d82};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x327b23c6, 0x74b0dc51, 0x625558ec, 0x3d1b58ba, 0x41b71efb,
      0x515f007c, 0x4db127f8, 0x1190cde7, 0x3352255a, 0x7fdcc233, 0x6b68079a,
      0x519b500d, 0x7c83e458, 0x436c6125, 0x721da317, 0x6763845e, 0x79838cb2,
      0x189a769b, 0x2ca88611, 0x3a95f874};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x643c9869, 0x19495cff, 0x238e1f29, 0x507ed7ab, 0x79e2a9e3,
      0x5bd062c2, 0x216231b,  0x66ef438d, 0x109cf92e, 0x1befd79f, 0x4e6afb66,
      0x431bd7b7, 0x257130a3, 0x628c895d, 0x2443a858, 0x75a2a8d4, 0x4353d0cd,
      0x54e49eb4, 0x836c40e,  0x8138641};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x643c692d, 0x19495cff, 0x238e1f29, 0x507e0000, 0x79e260ed,
      0x5bd062c2, 0x5fd1304c, 0x66ef438d, 0x476034d0, 0x1bef3496, 0x4e6a0000,
      0x777f0000, 0x257130a3, 0x628c3a7f, 0x2443541c, 0x75a20000, 0x43534a5d,
      0x54e45761, 0x8360000,  0x3d2515f6};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::viaddmax_s16x2_relu));
  std::cout << "sycl::ext::intel::math::viaddmax_s16x2_relu test pass."
            << std::endl;
}

void run_viaddmax_s32_test(s::queue &device_q) {
  std::initializer_list<int> input_vals1 = {
      0,          1804289383, 719885386,  -783368690, 304089172,   336465782,
      1101513929, 1059961393, -859484421, 149798315,  -1911759956, -511702305,
      -805750846, -939819582, 1374344043, 1780695788, 2053999932,  855636226,
      2040651434, 1330573317, -402724286};
  std::initializer_list<int> input_vals2 = {
      0,           -1681692777, -596516649, 2044897763, 35005211,   -278722862,
      1315634022,  628175011,   -608413784, 1129566413, 137806862,  -1937477084,
      -1100661313, -1998898814, 1477171087, -491705403, 1411549676, -1469348094,
      -317097467,  1687926652,  -1194953865};
  std::initializer_list<int> input_vals3 = {
      0,           -1957747793, -1025202362, 1365180540, -294702567,
      -2145174067, -1369133069, -1131176229, 1734575198, 412776091,
      -982906996,  -572660336,  -1141616124, 610515434,  945117276,
      -752392754,  943947739,   1036140795,  1376710097, 959997301,
      -364228444};
  std::initializer_list<int> ref_vals = {
      0,           122596606,  123368737,  1365180540, 339094383,  57742920,
      -1369133069, 1688136404, 1734575198, 1279364728, -982906996, 1845787907,
      -1141616124, 1356248900, 945117276,  1288990385, 943947739,  1036140795,
      1723553967,  959997301,  -364228444};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::viaddmax_s32));
  std::cout << "sycl::ext::intel::math::viaddmax_s32 test pass." << std::endl;
}

void run_viaddmax_s32_relu_test(s::queue &device_q) {
  std::initializer_list<int> input_vals1 = {
      0,          1804289383, 719885386,  -783368690, 304089172,   336465782,
      1101513929, 1059961393, -859484421, 149798315,  -1911759956, -511702305,
      -805750846, -939819582, 1374344043, 1780695788, 2053999932,  855636226,
      2040651434, 1330573317, -402724286};
  std::initializer_list<int> input_vals2 = {
      0,           -1681692777, -596516649, 2044897763, 35005211,   -278722862,
      1315634022,  628175011,   -608413784, 1129566413, 137806862,  -1937477084,
      -1100661313, -1998898814, 1477171087, -491705403, 1411549676, -1469348094,
      -317097467,  1687926652,  -1194953865};
  std::initializer_list<int> input_vals3 = {
      0,           -1957747793, -1025202362, 1365180540, -294702567,
      -2145174067, -1369133069, -1131176229, 1734575198, 412776091,
      -982906996,  -572660336,  -1141616124, 610515434,  945117276,
      -752392754,  943947739,   1036140795,  1376710097, 959997301,
      -364228444};
  std::initializer_list<int> ref_vals = {
      0,          122596606,  123368737,  1365180540, 339094383, 57742920,
      0,          1688136404, 1734575198, 1279364728, 0,         1845787907,
      0,          1356248900, 945117276,  1288990385, 943947739, 1036140795,
      1723553967, 959997301,  0};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::viaddmax_s32_relu));
  std::cout << "sycl::ext::intel::math::viaddmax_s32_relu test pass."
            << std::endl;
}

void run_viaddmax_u32_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x6b8b4567, 0x66334873, 0x2ae8944a, 0x46e87ccd, 0x2eb141f2,
      0x7545e146, 0x12200854, 0x1f16e9e8, 0x140e0f76, 0xded7263,  0x41a7c4c9,
      0x25e45d32, 0x3f2dba31, 0x62bbd95a, 0x333ab105, 0x2d1d5ae9, 0x8edbdab,
      0xb03e0c6,  0x71f32454, 0x2901d82};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x327b23c6, 0x74b0dc51, 0x625558ec, 0x3d1b58ba, 0x41b71efb,
      0x515f007c, 0x4db127f8, 0x1190cde7, 0x3352255a, 0x7fdcc233, 0x6b68079a,
      0x519b500d, 0x7c83e458, 0x436c6125, 0x721da317, 0x6763845e, 0x79838cb2,
      0x189a769b, 0x2ca88611, 0x3a95f874};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x643c9869, 0x19495cff, 0x238e1f29, 0x507ed7ab, 0x79e2a9e3,
      0x5bd062c2, 0x216231b,  0x66ef438d, 0x109cf92e, 0x1befd79f, 0x4e6afb66,
      0x431bd7b7, 0x257130a3, 0x628c895d, 0x2443a858, 0x75a2a8d4, 0x4353d0cd,
      0x54e49eb4, 0x836c40e,  0x8138641};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x9e06692d, 0xdae424c4, 0x8d3ded36, 0x8403d587, 0x79e2a9e3,
      0xc6a4e1c2, 0x5fd1304c, 0x66ef438d, 0x476034d0, 0x8dca3496, 0xad0fcc63,
      0x777fad3f, 0xbbb19e89, 0xa6283a7f, 0xa558541c, 0x9480df47, 0x82714a5d,
      0x54e49eb4, 0x9e9baa65, 0x3d2615f6};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::viaddmax_u32));
  std::cout << "sycl::ext::intel::math::viaddmax_u32 test pass." << std::endl;
}

void run_viaddmax_u16x2_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x6b8b4567, 0x66334873, 0x2ae8944a, 0x46e87ccd, 0x2eb141f2,
      0x7545e146, 0x12200854, 0x1f16e9e8, 0x140e0f76, 0xded7263,  0x41a7c4c9,
      0x25e45d32, 0x3f2dba31, 0x62bbd95a, 0x333ab105, 0x2d1d5ae9, 0x8edbdab,
      0xb03e0c6,  0x71f32454, 0x2901d82};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x327b23c6, 0x74b0dc51, 0x625558ec, 0x3d1b58ba, 0x41b71efb,
      0x515f007c, 0x4db127f8, 0x1190cde7, 0x3352255a, 0x7fdcc233, 0x6b68079a,
      0x519b500d, 0x7c83e458, 0x436c6125, 0x721da317, 0x6763845e, 0x79838cb2,
      0x189a769b, 0x2ca88611, 0x3a95f874};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x643c9869, 0x19495cff, 0x238e1f29, 0x507ed7ab, 0x79e2a9e3,
      0x5bd062c2, 0x216231b,  0x66ef438d, 0x109cf92e, 0x1befd79f, 0x4e6afb66,
      0x431bd7b7, 0x257130a3, 0x628c895d, 0x2443a858, 0x75a2a8d4, 0x4353d0cd,
      0x54e49eb4, 0x836c40e,  0x8138641};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x9e069869, 0xdae35cff, 0x8d3ded36, 0x8403d7ab, 0x79e2a9e3,
      0xc6a4e1c2, 0x5fd1304c, 0x66efb7cf, 0x4760f92e, 0x8dc9d79f, 0xad0ffb66,
      0x777fd7b7, 0xbbb09e89, 0xa627895d, 0xa557a858, 0x9480df47, 0x8270d0cd,
      0x54e49eb4, 0x9e9bc40e, 0x3d258641};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::viaddmax_u16x2));
  std::cout << "sycl::ext::intel::math::viaddmax_u16x2 test pass." << std::endl;
}

void run_viaddmin_s16x2_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x6b8b4567, 0x66334873, 0x2ae8944a, 0x46e87ccd, 0x2eb141f2,
      0x7545e146, 0x12200854, 0x1f16e9e8, 0x140e0f76, 0xded7263,  0x41a7c4c9,
      0x25e45d32, 0x3f2dba31, 0x62bbd95a, 0x333ab105, 0x2d1d5ae9, 0x8edbdab,
      0xb03e0c6,  0x71f32454, 0x2901d82};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x327b23c6, 0x74b0dc51, 0x625558ec, 0x3d1b58ba, 0x41b71efb,
      0x515f007c, 0x4db127f8, 0x1190cde7, 0x3352255a, 0x7fdcc233, 0x6b68079a,
      0x519b500d, 0x7c83e458, 0x436c6125, 0x721da317, 0x6763845e, 0x79838cb2,
      0x189a769b, 0x2ca88611, 0x3a95f874};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x643c9869, 0x19495cff, 0x238e1f29, 0x507ed7ab, 0x79e2a9e3,
      0x5bd062c2, 0x216231b,  0x66ef438d, 0x109cf92e, 0x1befd79f, 0x4e6afb66,
      0x431bd7b7, 0x257130a3, 0x628c895d, 0x2443a858, 0x75a2a8d4, 0x4353d0cd,
      0x54e49eb4, 0x836c40e,  0x8138641};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x9e069869, 0xdae324c4, 0x8d3ded36, 0x8403d587, 0x7068a9e3,
      0xc6a4e1c2, 0x216231b,  0x30a6b7cf, 0x109cf92e, 0x8dc9d79f, 0xad0fcc63,
      0x431bad3f, 0xbbb09e89, 0xa627895d, 0xa557a858, 0x9480a8d4, 0x8270d0cd,
      0x239d9eb4, 0x9e9baa65, 0x8138641};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::viaddmin_s16x2));
  std::cout << "sycl::ext::intel::math::viaddmin_s16x2 test pass." << std::endl;
}

void run_viaddmin_s16x2_relu_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x6b8b4567, 0x66334873, 0x2ae8944a, 0x46e87ccd, 0x2eb141f2,
      0x7545e146, 0x12200854, 0x1f16e9e8, 0x140e0f76, 0xded7263,  0x41a7c4c9,
      0x25e45d32, 0x3f2dba31, 0x62bbd95a, 0x333ab105, 0x2d1d5ae9, 0x8edbdab,
      0xb03e0c6,  0x71f32454, 0x2901d82};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x327b23c6, 0x74b0dc51, 0x625558ec, 0x3d1b58ba, 0x41b71efb,
      0x515f007c, 0x4db127f8, 0x1190cde7, 0x3352255a, 0x7fdcc233, 0x6b68079a,
      0x519b500d, 0x7c83e458, 0x436c6125, 0x721da317, 0x6763845e, 0x79838cb2,
      0x189a769b, 0x2ca88611, 0x3a95f874};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x643c9869, 0x19495cff, 0x238e1f29, 0x507ed7ab, 0x79e2a9e3,
      0x5bd062c2, 0x216231b,  0x66ef438d, 0x109cf92e, 0x1befd79f, 0x4e6afb66,
      0x431bd7b7, 0x257130a3, 0x628c895d, 0x2443a858, 0x75a2a8d4, 0x4353d0cd,
      0x54e49eb4, 0x836c40e,  0x8138641};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x0,       0x24c4,     0x0,        0x0, 0x70680000,
      0x0,        0x216231b, 0x30a60000, 0x109c0000, 0x0, 0x0,
      0x431b0000, 0x0,       0x0,        0x0,        0x0, 0x0,
      0x239d0000, 0x0,       0x8130000};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::viaddmin_s16x2_relu));
  std::cout << "sycl::ext::intel::math::viaddmin_s16x2_relu test pass."
            << std::endl;
}

void run_viaddmin_s32_test(s::queue &device_q) {
  std::initializer_list<int> input_vals1 = {
      0,          1804289383, 719885386,  -783368690, 304089172,   336465782,
      1101513929, 1059961393, -859484421, 149798315,  -1911759956, -511702305,
      -805750846, -939819582, 1374344043, 1780695788, 2053999932,  855636226,
      2040651434, 1330573317, -402724286};
  std::initializer_list<int> input_vals2 = {
      0,           -1681692777, -596516649, 2044897763, 35005211,   -278722862,
      1315634022,  628175011,   -608413784, 1129566413, 137806862,  -1937477084,
      -1100661313, -1998898814, 1477171087, -491705403, 1411549676, -1469348094,
      -317097467,  1687926652,  -1194953865};
  std::initializer_list<int> input_vals3 = {
      0,           -1957747793, -1025202362, 1365180540, -294702567,
      -2145174067, -1369133069, -1131176229, 1734575198, 412776091,
      -982906996,  -572660336,  -1141616124, 610515434,  945117276,
      -752392754,  943947739,   1036140795,  1376710097, 959997301,
      -364228444};
  std::initializer_list<int> ref_vals = {
      0,           -1957747793, -1025202362, 1261529073,  -294702567,
      -2145174067, -1877819345, -1131176229, -1467898205, 412776091,
      -1773953094, -572660336,  -1906412159, 610515434,   -1443452166,
      -752392754,  -829417688,  -613711868,  1376710097,  -1276467327,
      -1597678151};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::viaddmin_s32));
  std::cout << "sycl::ext::intel::math::viaddmin_s32 test pass." << std::endl;
}

void run_viaddmin_s32_relu_test(s::queue &device_q) {
  std::initializer_list<int> input_vals1 = {
      0,          1804289383, 719885386,  -783368690, 304089172,   336465782,
      1101513929, 1059961393, -859484421, 149798315,  -1911759956, -511702305,
      -805750846, -939819582, 1374344043, 1780695788, 2053999932,  855636226,
      2040651434, 1330573317, -402724286};
  std::initializer_list<int> input_vals2 = {
      0,           -1681692777, -596516649, 2044897763, 35005211,   -278722862,
      1315634022,  628175011,   -608413784, 1129566413, 137806862,  -1937477084,
      -1100661313, -1998898814, 1477171087, -491705403, 1411549676, -1469348094,
      -317097467,  1687926652,  -1194953865};
  std::initializer_list<int> input_vals3 = {
      0,           -1957747793, -1025202362, 1365180540, -294702567,
      -2145174067, -1369133069, -1131176229, 1734575198, 412776091,
      -982906996,  -572660336,  -1141616124, 610515434,  945117276,
      -752392754,  943947739,   1036140795,  1376710097, 959997301,
      -364228444};
  std::initializer_list<int> ref_vals = {
      0, 0, 0,         1261529073, 0, 0, 0, 0,          0, 412776091, 0,
      0, 0, 610515434, 0,          0, 0, 0, 1376710097, 0, 0,
  };
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::viaddmin_s32_relu));
  std::cout << "sycl::ext::intel::math::viaddmin_s32_relu test pass."
            << std::endl;
}

void run_viaddmin_u32_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x6b8b4567, 0x66334873, 0x2ae8944a, 0x46e87ccd, 0x2eb141f2,
      0x7545e146, 0x12200854, 0x1f16e9e8, 0x140e0f76, 0xded7263,  0x41a7c4c9,
      0x25e45d32, 0x3f2dba31, 0x62bbd95a, 0x333ab105, 0x2d1d5ae9, 0x8edbdab,
      0xb03e0c6,  0x71f32454, 0x2901d82};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x327b23c6, 0x74b0dc51, 0x625558ec, 0x3d1b58ba, 0x41b71efb,
      0x515f007c, 0x4db127f8, 0x1190cde7, 0x3352255a, 0x7fdcc233, 0x6b68079a,
      0x519b500d, 0x7c83e458, 0x436c6125, 0x721da317, 0x6763845e, 0x79838cb2,
      0x189a769b, 0x2ca88611, 0x3a95f874};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x643c9869, 0x19495cff, 0x238e1f29, 0x507ed7ab, 0x79e2a9e3,
      0x5bd062c2, 0x216231b,  0x66ef438d, 0x109cf92e, 0x1befd79f, 0x4e6afb66,
      0x431bd7b7, 0x257130a3, 0x628c895d, 0x2443a858, 0x75a2a8d4, 0x4353d0cd,
      0x54e49eb4, 0x836c40e,  0x8138641};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x643c9869, 0x19495cff, 0x238e1f29, 0x507ed7ab, 0x706860ed,
      0x5bd062c2, 0x216231b,  0x30a7b7cf, 0x109cf92e, 0x1befd79f, 0x4e6afb66,
      0x431bd7b7, 0x257130a3, 0x628c895d, 0x2443a858, 0x75a2a8d4, 0x4353d0cd,
      0x239e5761, 0x836c40e,  0x8138641};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::viaddmin_u32));
  std::cout << "sycl::ext::intel::math::viaddmin_u32 test pass." << std::endl;
}

void run_viaddmin_u16x2_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x6b8b4567, 0x66334873, 0x2ae8944a, 0x46e87ccd, 0x2eb141f2,
      0x7545e146, 0x12200854, 0x1f16e9e8, 0x140e0f76, 0xded7263,  0x41a7c4c9,
      0x25e45d32, 0x3f2dba31, 0x62bbd95a, 0x333ab105, 0x2d1d5ae9, 0x8edbdab,
      0xb03e0c6,  0x71f32454, 0x2901d82};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x327b23c6, 0x74b0dc51, 0x625558ec, 0x3d1b58ba, 0x41b71efb,
      0x515f007c, 0x4db127f8, 0x1190cde7, 0x3352255a, 0x7fdcc233, 0x6b68079a,
      0x519b500d, 0x7c83e458, 0x436c6125, 0x721da317, 0x6763845e, 0x79838cb2,
      0x189a769b, 0x2ca88611, 0x3a95f874};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x643c9869, 0x19495cff, 0x238e1f29, 0x507ed7ab, 0x79e2a9e3,
      0x5bd062c2, 0x216231b,  0x66ef438d, 0x109cf92e, 0x1befd79f, 0x4e6afb66,
      0x431bd7b7, 0x257130a3, 0x628c895d, 0x2443a858, 0x75a2a8d4, 0x4353d0cd,
      0x54e49eb4, 0x836c40e,  0x8138641};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x643c692d, 0x194924c4, 0x238e1f29, 0x507ed587, 0x706860ed,
      0x5bd062c2, 0x216231b,  0x30a6438d, 0x109c34d0, 0x1bef3496, 0x4e6acc63,
      0x431bad3f, 0x257130a3, 0x628c3a7f, 0x2443541c, 0x75a2a8d4, 0x43534a5d,
      0x239d5761, 0x836aa65,  0x81315f6};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::viaddmin_u16x2));
  std::cout << "sycl::ext::intel::math::viaddmin_u16x2 test pass." << std::endl;
}

void run_vibmax_s16x2_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x6b8b4567, 0x643c9869, 0x74b0dc51, 0x2ae8944a, 0x238e1f29,
      0x3d1b58ba, 0x2eb141f2, 0x79e2a9e3, 0x515f007c, 0x12200854, 0x216231b,
      0x1190cde7, 0x140e0f76, 0x109cf92e, 0x7fdcc233, 0x41a7c4c9, 0x4e6afb66,
      0x519b500d, 0x3f2dba31, 0x257130a3};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x327b23c6, 0x66334873, 0x19495cff, 0x625558ec, 0x46e87ccd,
      0x507ed7ab, 0x41b71efb, 0x7545e146, 0x5bd062c2, 0x4db127f8, 0x1f16e9e8,
      0x66ef438d, 0x3352255a, 0xded7263,  0x1befd79f, 0x6b68079a, 0x25e45d32,
      0x431bd7b7, 0x7c83e458, 0x62bbd95a};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x6b8b4567, 0x66334873, 0x74b05cff, 0x625558ec, 0x46e87ccd,
      0x507e58ba, 0x41b741f2, 0x79e2e146, 0x5bd062c2, 0x4db127f8, 0x1f16231b,
      0x66ef438d, 0x3352255a, 0x109c7263, 0x7fdcd79f, 0x6b68079a, 0x4e6a5d32,
      0x519b500d, 0x7c83e458, 0x62bb30a3};
  std::initializer_list<bool> ref_preds = {
      true,  true,  true,  true,  false, false, true,  false, false,
      false, false, false, false, true,  false, true,  true,  false,
      false, false, false, false, false, true,  false, false, false,
      false, true,  false, true,  false, false, false, true,  false,
      true,  true,  false, false, false, true};
  assert(ref_preds.size() == 2 * input_vals1.size());
  test2_with_pred<2>(device_q, input_vals1, input_vals2, ref_vals, ref_preds,
                     F2_PRED2(s::ext::intel::math::vibmax_s16x2));
  std::cout << "sycl::ext::intel::math::vibmax_s16x2 test pass." << std::endl;
}

void run_vibmax_s32_test(s::queue &device_q) {
  std::initializer_list<int> input_vals1 = {
      0,           1804289383, -1957747793, -596516649, -783368690, 1365180540,
      35005211,    336465782,  -2145174067, 1315634022, 1059961393, -1131176229,
      -608413784,  149798315,  412776091,   137806862,  -511702305, -572660336,
      -1100661313, -939819582, 610515434};
  std::initializer_list<int> input_vals2 = {
      0,          -1681692777, 719885386,  -1025202362, 2044897763,
      304089172,  -294702567,  -278722862, 1101513929,  -1369133069,
      628175011,  -859484421,  1734575198, 1129566413,  -1911759956,
      -982906996, -1937477084, -805750846, -1141616124, -1998898814,
      1374344043};
  std::initializer_list<int> ref_vals = {
      0,           1804289383, 719885386,  -596516649, 2044897763, 1365180540,
      35005211,    336465782,  1101513929, 1315634022, 1059961393, -859484421,
      1734575198,  1129566413, 412776091,  137806862,  -511702305, -572660336,
      -1100661313, -939819582, 1374344043};
  std::initializer_list<bool> ref_preds = {
      true,  true,  false, true, false, true, true, true, false, true, true,
      false, false, false, true, true,  true, true, true, true,  false};
  test2_with_pred<1>(device_q, input_vals1, input_vals2, ref_vals, ref_preds,
                     F2_PRED1(s::ext::intel::math::vibmax_s32));
  std::cout << "sycl::ext::intel::math::vibmax_s32 test pass." << std::endl;
}

void run_vibmax_u16x2_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x6b8b4567, 0x643c9869, 0x74b0dc51, 0x2ae8944a, 0x238e1f29,
      0x3d1b58ba, 0x2eb141f2, 0x79e2a9e3, 0x515f007c, 0x12200854, 0x216231b,
      0x1190cde7, 0x140e0f76, 0x109cf92e, 0x7fdcc233, 0x41a7c4c9, 0x4e6afb66,
      0x519b500d, 0x3f2dba31, 0x257130a3};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x327b23c6, 0x66334873, 0x19495cff, 0x625558ec, 0x46e87ccd,
      0x507ed7ab, 0x41b71efb, 0x7545e146, 0x5bd062c2, 0x4db127f8, 0x1f16e9e8,
      0x66ef438d, 0x3352255a, 0xded7263,  0x1befd79f, 0x6b68079a, 0x25e45d32,
      0x431bd7b7, 0x7c83e458, 0x62bbd95a};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x6b8b4567, 0x66339869, 0x74b0dc51, 0x6255944a, 0x46e87ccd,
      0x507ed7ab, 0x41b741f2, 0x79e2e146, 0x5bd062c2, 0x4db127f8, 0x1f16e9e8,
      0x66efcde7, 0x3352255a, 0x109cf92e, 0x7fdcd79f, 0x6b68c4c9, 0x4e6afb66,
      0x519bd7b7, 0x7c83e458, 0x62bbd95a};
  std::initializer_list<bool> ref_preds = {
      true,  true,  true,  true,  false, true,  true,  true, false,
      true,  false, false, false, false, false, true,  true, false,
      false, false, false, false, false, false, false, true, false,
      false, true,  true,  true,  false, false, true,  true, true,
      true,  false, false, false, false, false};
  assert(ref_preds.size() == 2 * input_vals1.size());
  test2_with_pred<2>(device_q, input_vals1, input_vals2, ref_vals, ref_preds,
                     F2_PRED2(s::ext::intel::math::vibmax_u16x2));
  std::cout << "sycl::ext::intel::math::vibmax_u16x2 test pass." << std::endl;
}

void run_vibmax_u32_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x6b8b4567, 0x643c9869, 0x74b0dc51, 0x2ae8944a, 0x238e1f29,
      0x3d1b58ba, 0x2eb141f2, 0x79e2a9e3, 0x515f007c, 0x12200854, 0x216231b,
      0x1190cde7, 0x140e0f76, 0x109cf92e, 0x7fdcc233, 0x41a7c4c9, 0x4e6afb66,
      0x519b500d, 0x3f2dba31, 0x257130a3};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x327b23c6, 0x66334873, 0x19495cff, 0x625558ec, 0x46e87ccd,
      0x507ed7ab, 0x41b71efb, 0x7545e146, 0x5bd062c2, 0x4db127f8, 0x1f16e9e8,
      0x66ef438d, 0x3352255a, 0xded7263,  0x1befd79f, 0x6b68079a, 0x25e45d32,
      0x431bd7b7, 0x7c83e458, 0x62bbd95a};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x6b8b4567, 0x66334873, 0x74b0dc51, 0x625558ec, 0x46e87ccd,
      0x507ed7ab, 0x41b71efb, 0x79e2a9e3, 0x5bd062c2, 0x4db127f8, 0x1f16e9e8,
      0x66ef438d, 0x3352255a, 0x109cf92e, 0x7fdcc233, 0x6b68079a, 0x4e6afb66,
      0x519b500d, 0x7c83e458, 0x62bbd95a};
  std::initializer_list<bool> ref_preds = {
      true,  true, false, true,  false, false, false,
      false, true, false, false, false, false, false,
      true,  true, false, true,  true,  false, false};
  test2_with_pred<1>(device_q, input_vals1, input_vals2, ref_vals, ref_preds,
                     F2_PRED1(s::ext::intel::math::vibmax_u32));
  std::cout << "sycl::ext::intel::math::vibmax_u32 test pass." << std::endl;
}

void run_vibmin_s16x2_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x3ae44994, 0x787aec15, 0x38f673b1, 0x1b50fa01, 0x1b7628fe,
      0x739f56bf, 0x1879b014, 0x1078d2b5, 0x375126c4, 0xa32741e,  0xdbdaa09,
      0x41f0eef3, 0x155f4b7c, 0x17d90dbf, 0x93498f3,  0x32df8360, 0x6c8b2bcf,
      0x1988c03c, 0x16761845, 0x7c66d55b};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x6ff2e5d4, 0x556fddb5, 0x6b1ba8d0, 0x56e535db, 0x6cc7a299,
      0x7f311340, 0xaef7e35,  0x7abb83ee, 0xeb6c876,  0x3bde1841, 0x482e927c,
      0x163a2ef4, 0x703bc2d9, 0x23e40bc6, 0x7c9845fa, 0x4418e287, 0x2b5a6f75,
      0x25819f80, 0x34d9ba3d, 0x31ec4143};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x3ae4e5d4, 0x556fddb5, 0x38f6a8d0, 0x1b50fa01, 0x1b76a299,
      0x739f1340, 0xaefb014,  0x107883ee, 0xeb6c876,  0xa321841,  0xdbd927c,
      0x163aeef3, 0x155fc2d9, 0x17d90bc6, 0x93498f3,  0x32df8360, 0x2b5a2bcf,
      0x19889f80, 0x1676ba3d, 0x31ecd55b};
  std::initializer_list<bool> ref_preds = {
      true,  true,  true,  false, false, false, true,  false, true,
      true,  true,  false, true,  false, false, true,  true,  false,
      false, false, true,  false, true,  false, false, true,  true,
      false, true,  false, true,  true,  true,  true,  false, true,
      true,  false, true,  false, false, true};
  assert(ref_preds.size() == 2 * input_vals1.size());
  test2_with_pred<2>(device_q, input_vals1, input_vals2, ref_vals, ref_preds,
                     F2_PRED2(s::ext::intel::math::vibmin_s16x2));
  std::cout << "sycl::ext::intel::math::vibmin_s16x2 test pass." << std::endl;
}

void run_vibmin_s32_test(s::queue &device_q) {
  std::initializer_list<int> input_vals1 = {
      0,          611047636,   672483410,  1754331344, -612962572, -2041446763,
      378469960,  -458562784,  1292245326, 1038406112, 1240407460, 920494138,
      2107191082, -1359579123, -185159361, 96649971,   1695288540, 1132217736,
      1156357266, -975142102,  -983330111};
  std::initializer_list<int> input_vals2 = {
      0,          1348551940, -910303841, -614341486,  1944071070,  1721305015,
      295605318,  -682512160, 799785470,  -1698158054, 1179333692,  1130914815,
      -610223694, 1150185827, -368424926, -331578504,  -1144704611, -149616880,
      1072396705, 1462824467, 788520085};
  std::initializer_list<int> ref_vals = {
      0,           611047636,   -910303841, -614341486,  -612962572,
      -2041446763, 295605318,   -682512160, 799785470,   -1698158054,
      1179333692,  920494138,   -610223694, -1359579123, -368424926,
      -331578504,  -1144704611, -149616880, 1072396705,  -975142102,
      -983330111};
  std::initializer_list<bool> ref_preds = {
      true,  true,  false, false, true,  true,  false,
      false, false, false, false, true,  false, true,
      false, false, false, false, false, true,  true};
  test2_with_pred<1>(device_q, input_vals1, input_vals2, ref_vals, ref_preds,
                     F2_PRED1(s::ext::intel::math::vibmin_s32));
  std::cout << "sycl::ext::intel::math::vibmin_s32 test pass." << std::endl;
}

void run_vibmin_u16x2_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x56638e,   0x69822658, 0x94ff48d,  0x51077110, 0x14a68a57,
      0x76dac91e, 0x304b4f55, 0xb886b59,  0x3353648d, 0x448a1247, 0x21d831a5,
      0xe8fd70b,  0x38a7aef4, 0x2bd7eec,  0x7b0f87e6, 0x221d608,  0xfd833f1,
      0x1d65e540, 0x75bec180, 0x7324ce39};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x5274c4a3, 0x21fff9cc, 0xa1ac520,  0x59fca5bb, 0x75f6d3d,
      0x1b13b867, 0x1890e709, 0x129b691d, 0x3d6e76c1, 0x5542b927, 0x7c5efe8c,
      0x5c51243e, 0x6beef979, 0x339cae8e, 0x3d636f4d, 0x7b65eb74, 0x6ba3fc60,
      0x1928287e, 0x6e6d5651, 0xa654bd7};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x56638e,   0x21ff2658, 0x94fc520,  0x51077110, 0x75f6d3d,
      0x1b13b867, 0x18904f55, 0xb88691d,  0x3353648d, 0x448a1247, 0x21d831a5,
      0xe8f243e,  0x38a7aef4, 0x2bd7eec,  0x3d636f4d, 0x221d608,  0xfd833f1,
      0x1928287e, 0x6e6d5651, 0xa654bd7};
  std::initializer_list<bool> ref_preds = {
      true,  true,  true,  true,  false, true,  true, false, true,
      true,  false, false, false, false, false, true, true,  false,
      true,  true,  true,  true,  true,  true,  true, false, true,
      true,  true,  true,  false, false, true,  true, true,  true,
      false, false, false, false, false, false};
  assert(ref_preds.size() == 2 * input_vals1.size());
  test2_with_pred<2>(device_q, input_vals1, input_vals2, ref_vals, ref_preds,
                     F2_PRED2(s::ext::intel::math::vibmin_u16x2));
  std::cout << "sycl::ext::intel::math::vibmin_u16x2 test pass." << std::endl;
}

void run_vibmin_u32_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x3ff0ddd1, 0x757f6f96, 0xe73c53b,  0x559221f0, 0x78bc6e5d,
      0x2d64c9da, 0x2fd1fd44, 0x2fc2a701, 0x48d528f7, 0x6272a12,  0x72c5ce60,
      0x90aa59f,  0x755e7b44, 0x1871aa1b, 0x3431e998, 0x74fba93d, 0x35f081f0,
      0x2b3214ff, 0x242873e9, 0x4dc12143};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x2578430b, 0x370f4d96, 0x39ad5b16, 0x95cda17,  0x3f40b5ed,
      0x28761919, 0x10ffe2c6, 0x62e0548c, 0xa6f340a,  0x7cf2423,  0x47e44311,
      0x43afbcfb, 0x6339465b, 0x4d44064e, 0x10783ee4, 0x7422c769, 0x6a7b18d3,
      0x4464472b, 0xc436ef,   0x1ce4e246};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x2578430b, 0x370f4d96, 0xe73c53b,  0x95cda17,  0x3f40b5ed,
      0x28761919, 0x10ffe2c6, 0x2fc2a701, 0xa6f340a,  0x6272a12,  0x47e44311,
      0x90aa59f,  0x6339465b, 0x1871aa1b, 0x10783ee4, 0x7422c769, 0x35f081f0,
      0x2b3214ff, 0xc436ef,   0x1ce4e246};
  std::initializer_list<bool> ref_preds = {
      true,  false, false, true, false, false, false, false, true,  false, true,
      false, true,  false, true, false, false, true,  true,  false, false};
  test2_with_pred<1>(device_q, input_vals1, input_vals2, ref_vals, ref_preds,
                     F2_PRED1(s::ext::intel::math::vibmin_u32));
  std::cout << "sycl::ext::intel::math::vibmin_u32 test pass." << std::endl;
}

void run_vimax3_s16x2_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x4b7a4937, 0x4d18ee29, 0x68efe422, 0x5532e212, 0x3c6db6de,
      0x62ad36ff, 0x2909b52,  0x4e6ad8fc, 0x677e94fa, 0x184b7238, 0x37ebfeca,
      0x39a572f7, 0x1a1a98d,  0x63170eab, 0x3b1ca699, 0x528ee5f1, 0x50c6447e,
      0x523b132d, 0x29e8bfdf, 0x680f689b};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x487d8b61, 0x3bfc5454, 0x282f3c8a, 0x293aeca1, 0x151b3b5c,
      0x7c63bdf2, 0x54fd85ef, 0x30de04d2, 0x43bc0cd1, 0x28666607, 0x73e0af3e,
      0x40f99d67, 0x29e98189, 0x7f1c639b, 0x3b8a1a79, 0x1e375178, 0x20c7ecca,
      0x6f32c5c7, 0x56b15ac1, 0x6efcccf9};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x1b9742d,  0x47fc3696, 0x6175651e, 0x580597ed, 0x17723f58,
      0x7e375e8d, 0x174ceae,  0x57adacb2, 0x3e26a8bc, 0x77e83b2f, 0x4065c690,
      0x7c621ae5, 0x2491576f, 0x4dcc4410, 0x62e77f6c, 0x5f4b3d5e, 0x3448c34d,
      0x6526c81f, 0x28e2d4f1, 0x51493af8};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x4b7a742d, 0x4d185454, 0x68ef651e, 0x5805eca1, 0x3c6d3f58,
      0x7e375e8d, 0x54fdceae, 0x57ad04d2, 0x677e0cd1, 0x77e87238, 0x73e0feca,
      0x7c6272f7, 0x29e9576f, 0x7f1c639b, 0x62e77f6c, 0x5f4b5178, 0x50c6447e,
      0x6f32132d, 0x56b15ac1, 0x6efc689b};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::vimax3_s16x2));
  std::cout << "sycl::ext::intel::math::vimax3_s16x2 test pass." << std::endl;
}

int main(int, char **) {
  s::queue device_queue(s::default_selector_v);
  std::cout << "Running on "
            << device_queue.get_device().get_info<s::info::device::name>()
            << "\n";
  run_vabs2_4_test(device_queue);
  run_vabsss2_4_test(device_queue);
  run_vabsdiffsu2_4_test(device_queue);
  run_vadd_ss_2_4_test(device_queue);
  run_vadd_us_2_4_test(device_queue);
  run_vsub_ss_2_4_test(device_queue);
  run_vsub_us_2_4_test(device_queue);
  run_vhaddu_2_4_test(device_queue);
  run_vavgu_2_4_test(device_queue);
  run_vcmpgts_2_4_test(device_queue);
  run_vmaxmin_2_4_test(device_queue);
  run_vneg_2_4_test(device_queue);
  run_vnegss_2_4_test(device_queue);
  run_vsad_2_4_test(device_queue);
  run_veqne_2_4_test(device_queue);
  run_vgelt_2_4_test(device_queue);
  run_vgtle_2_4_test(device_queue);
  run_vavgs_2_4_test(device_queue);
  run_viaddmax_s16x2_test(device_queue);
  run_viaddmax_s16x2_relu_test(device_queue);
  run_viaddmax_s32_test(device_queue);
  run_viaddmax_s32_relu_test(device_queue);
  run_viaddmax_u16x2_test(device_queue);
  run_viaddmax_u32_test(device_queue);
  run_viaddmin_s16x2_test(device_queue);
  run_viaddmin_s16x2_relu_test(device_queue);
  run_viaddmin_s32_test(device_queue);
  run_viaddmin_s32_relu_test(device_queue);
  run_viaddmin_u16x2_test(device_queue);
  run_viaddmin_u32_test(device_queue);
  run_vibmax_s16x2_test(device_queue);
  run_vibmax_s32_test(device_queue);
  run_vibmax_u16x2_test(device_queue);
  run_vibmax_u32_test(device_queue);
  run_vibmin_s16x2_test(device_queue);
  run_vibmin_s32_test(device_queue);
  run_vibmin_u16x2_test(device_queue);
  run_vibmin_u32_test(device_queue);
  run_vimax3_s16x2_test(device_queue);
  return 0;
}
