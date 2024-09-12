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

void run_vimin3_s16x2_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x4f57d411, 0x4c26207,  0x2ca8da9e, 0x61392f42, 0x34d89c37,
      0x1206ac6d, 0x728b19f9, 0x765a6295, 0x36dbf763, 0x739734f,  0x7c09cb1d,
      0x34f1eb03, 0x1d98cee4, 0x698a958c, 0x674b456a, 0x44698deb, 0x3ff288d7,
      0x5445ca7c, 0x2f0ae37,  0x6bb35255};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x18027349, 0x264c6286, 0x47875b8b, 0x4612c8ee, 0x4994a1b7,
      0x4d561240, 0x2c29666f, 0x482741c,  0x2f98613f, 0x10067cdf, 0x5f5e50f0,
      0x6420b2f8, 0x10c98d96, 0x7202bcd8, 0x26db590f, 0x38e2057d, 0x2b6d1f76,
      0x21c7820b, 0x58a3796f, 0x5fdcecbe};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x38e81fe5, 0x68a6e3e1, 0x4bf1c6a8, 0x7dc0afdd, 0x5d1e4881,
      0x7b88faec, 0x145341a4, 0x2eaae3bb, 0x68c2a41e, 0x69410d78, 0x14380c1,
      0x278fe348, 0x6f173ed3, 0x352a07c2, 0x7ebea979, 0x4c14bbba, 0x783e2229,
      0x7cc09645, 0x2c58f784, 0x3c5f7464};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x1802d411, 0x4c2e3e1,  0x2ca8c6a8, 0x4612afdd, 0x34d89c37,
      0x1206ac6d, 0x145319f9, 0x482e3bb,  0x2f98a41e, 0x7390d78,  0x14380c1,
      0x278fb2f8, 0x10c98d96, 0x352a958c, 0x26dba979, 0x38e28deb, 0x2b6d88d7,
      0x21c7820b, 0x2f0ae37,  0x3c5fecbe};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::vimin3_s16x2));
  std::cout << "sycl::ext::intel::math::vimin3_s16x2 test pass." << std::endl;
}

void run_vimax3_s16x2_relu_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x50cd63f6, 0x44826549, 0x415211f4, 0x363618a6, 0x13db1ca9,
      0x2303ee80, 0x36163f94, 0x5beffc38, 0x45f3ebc5, 0x537ab4dc, 0x367cb835,
      0x25f947a1, 0x1a7954af, 0x85be209,  0x447a2de0, 0xca3ea84,  0x674caa1a,
      0x2144a51b, 0x46fe916e, 0x10815a2e};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x2c5da1a6, 0x2bf3d9a2, 0x2e4d22cd, 0x385a093f, 0x52eaa576,
      0x27a45723, 0x3798af50, 0x796652ee, 0x4dd9269e, 0x61b59dc,  0x56e8bdd3,
      0x1b6b231c, 0x5cbd3510, 0x12f34db6, 0x26ce6a60, 0x49d258e0, 0x7fe89874,
      0x5bd894ad, 0x21cc8072, 0x7547354e};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x6f7c8f6b, 0x74800d0e, 0x6de28d5a, 0x3c1e4bd7, 0x4829bca3,
      0x5aa8bf96, 0x39f7fb01, 0x25b9ec52, 0x4982c8c0, 0x1998ceee, 0x45f67094,
      0x71ea4a36, 0x20376d03, 0x58917642, 0x2b7c1bb8, 0x532072dc, 0xab9222c,
      0x41f751b,  0x51f89bb9, 0x5813f596};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x6f7c63f6, 0x74806549, 0x6de222cd, 0x3c1e4bd7, 0x52ea1ca9,
      0x5aa85723, 0x39f73f94, 0x796652ee, 0x4dd9269e, 0x537a59dc, 0x56e87094,
      0x71ea4a36, 0x5cbd6d03, 0x58917642, 0x447a6a60, 0x532072dc, 0x7fe8222c,
      0x5bd8751b, 0x51f80000, 0x75475a2e};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::vimax3_s16x2_relu));
  std::cout << "sycl::ext::intel::math::vimax3_s16x2_relu test pass."
            << std::endl;
}

void run_vimin3_s16x2_relu_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x25879885, 0x7e2361a7, 0x3d5285d4, 0x6a2677ac, 0x39e67c81,
      0x75b12d6a, 0x774201f,  0x6aeeb696, 0x2b05223b, 0x2200d55,  0x3ccd66c6,
      0x79cec7d6, 0x76dce36e, 0x6198d27f, 0x625e9bf6, 0x614ab233, 0x6a78ee9c,
      0x367bab94, 0x241a258a, 0x458e3e1e};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x5f613e39, 0x539d1cea, 0x5162a414, 0x6ac2f697, 0x7e863916,
      0x30fb2b95, 0x7b6caa8f, 0x2f3c6b4,  0x65ea6316, 0x319aec18, 0x5722849e,
      0x5545e645, 0x12986c19, 0x7cbee3c6, 0x36a56047, 0x2c568db1, 0x33caadd0,
      0x1eb96466, 0x49be86a2, 0x4bde93f7};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x3d016110, 0x7d0e1b97, 0x6abbef11, 0xc5c977,   0x7eec163d,
      0x92e3c68,  0x4c02bcf8, 0x6d9e79f5, 0x21741894, 0x1dc235ef, 0x7d237428,
      0x50c09112, 0x22233526, 0xce62bbe,  0xb6c64d4,  0x3c679069, 0x37d43af8,
      0x3ac801ad, 0x20b264c3, 0x524d50db};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x25870000, 0x539d1b97, 0x3d520000, 0xc50000,   0x39e6163d,
      0x92e2b95,  0x7740000,  0x2f30000,  0x21741894, 0x2200000,  0x3ccd0000,
      0x50c00000, 0x12980000, 0xce60000,  0xb6c0000,  0x2c560000, 0x33ca0000,
      0x1eb90000, 0x20b20000, 0x458e0000};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::vimin3_s16x2_relu));
  std::cout << "sycl::ext::intel::math::vimin3_s16x2_relu test pass."
            << std::endl;
}

void run_vimax3_s32_test(s::queue &device_q) {
  std::initializer_list<int> input_vals1 = {
      0,          -742133555, 397347510,  1579311184, -1423647137, -2047856174,
      1481948582, 348229674,  1369364594, -84514390,  -416077873,  49823278,
      -102896595, 821257839,  1034147111, 2029050660, -1961347722, 1217494617,
      1728948925, 823267476,  760021269};
  std::initializer_list<int> input_vals2 = {
      0,         1165600424,  -646345851, 25373477,    -2104769074, 203141950,
      610965818, -1819670293, -560752773, -950013719,  -565469972,  1057801409,
      31204368,  -361123047,  -676398745, -2039651173, 522112306,   -1834882414,
      411946897, -1084882858, 194527069};
  std::initializer_list<int> input_vals3 = {
      0,           978625329,   1373987954, -1238867373, 1242359758,
      948840765,   -1435511372, -449045571, 519556104,   -731439951,
      -1721481747, -1556012040, -751333592, -1367952609, 1494266817,
      382080865,   -1092693239, 970673074,  1026457879,  -1897282173,
      -78402523};
  std::initializer_list<int> ref_vals = {
      0,          1165600424, 1373987954, 1579311184, 1242359758, 948840765,
      1481948582, 348229674,  1369364594, -84514390,  -416077873, 1057801409,
      31204368,   821257839,  1494266817, 2029050660, 522112306,  1217494617,
      1728948925, 823267476,  760021269};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::vimax3_s32));
  std::cout << "sycl::ext::intel::math::vimax3_s32 test pass." << std::endl;
}

void run_vimin3_s32_test(s::queue &device_q) {
  std::initializer_list<int> input_vals1 = {
      0,           -1471611625, 9999914,    659952013, -2092947813, 1147981719,
      -2008458679, 521151995,   -816042144, 425774287, -1338184873, -457830008,
      -582916353,  -1887045021, 1927359983, 878193563, 1557974931,  -1897745960,
      -555631778,  -1684504516, 1747637813};
  std::initializer_list<int> input_vals2 = {
      0,          463882823,  -2045679495, -86355736,  31464511,    1699586676,
      -772060921, -45032941,  1134855236,  1273829199, 517751151,   1277920908,
      1150746543, -965934964, 1452888395,  1210550422, -1053941213, -1929174740,
      685330798,  -340615400, -526853110};
  std::initializer_list<int> input_vals3 = {
      0,           728448149,   -1214062786, 2065571785,  -698142478,
      1791762714,  -2078896508, -651360028,  -1344486457, -1452323802,
      -1064949703, 1250782590,  -43798780,   -1762238805, -497768242,
      578757370,   -118994660,  -1329860307, -1943838290, 629936511,
      756591140};
  std::initializer_list<int> ref_vals = {
      0,           -1471611625, -2045679495, -86355736,   -2092947813,
      1147981719,  -2078896508, -651360028,  -1344486457, -1452323802,
      -1338184873, -457830008,  -582916353,  -1887045021, -497768242,
      578757370,   -1053941213, -1929174740, -1943838290, -1684504516,
      -526853110};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::vimin3_s32));
  std::cout << "sycl::ext::intel::math::vimin3_s32 test pass." << std::endl;
}

void run_vimax3_s32_relu_test(s::queue &device_q) {
  std::initializer_list<int> input_vals1 = {
      0,           52953643,   200934364,   -320079327, -1023087153,
      -2013529685, 1708900323, 1729046100,  2014458917, 204509490,
      -1775896788, 1510824901, -2067452966, -961263900, 1783827961,
      422885021,   -420041979, 17859616,    758599933,  -1550109771,
      -986339893};
  std::initializer_list<int> input_vals2 = {
      0,           -1870383071, -807524162,  -1672320404, 585857541,
      -1806386998, -46951753,   23435910,    -542100165,  684042841,
      1069715283,  1133089007,  -292269911,  -924750229,  -1979339789,
      1390421507,  -344460416,  -1305077947, 344922780,   387360071,
      1843454056};
  std::initializer_list<int> input_vals3 = {
      0,          -462724558,  994422761,   -1885847379, 1565938672,
      -5210040,   1626606917,  -156200495,  1089493027,  1920001488,
      -770052941, -1763269363, 1733918430,  992705724,   -1987582246,
      903311876,  1824429784,  -1687804331, 2004948649,  -1961680118,
      1729681795};
  std::initializer_list<int> ref_vals = {
      0,          52953643,   994422761,  0,          1565938672, 0,
      1708900323, 1729046100, 2014458917, 1920001488, 1069715283, 1510824901,
      1733918430, 992705724,  1783827961, 1390421507, 1824429784, 17859616,
      2004948649, 387360071,  1843454056};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::vimax3_s32_relu));
  std::cout << "sycl::ext::intel::math::vimax3_s32_relu test pass."
            << std::endl;
}

void run_vimin3_s32_relu_test(s::queue &device_q) {
  std::initializer_list<int> input_vals1 = {
      0,          331089896,   -1934710780, -1239279316, 154189716,
      245200284,  -425215500,  -1836736674, 1966240382,  1252655883,
      237991677,  -150087665,  -672923374,  -1479538282, -1627556526,
      1146627861, -1200652318, 159097447,   794292603,   274455501,
      1426313945};
  std::initializer_list<int> input_vals2 = {
      0,           -1600082444, -1939972842, 604719016,   2142389754,
      2000266666,  56310693,    610666988,   1270298077,  -126706733,
      1359054381,  -656942960,  28255806,    -277418562,  429743474,
      -1259686747, 1989293157,  -441513038,  -1660194415, 502272614,
      1182638704};
  std::initializer_list<int> input_vals3 = {
      0,           -410214321,  1732764580, -1902389385, 1413141516,
      -1630882989, -1939526504, 578683544,  -1072809422, -292371494,
      -152195455,  -455225251,  1773704145, 2015936629,  -221187206,
      549953580,   -906439604,  715469692,  -649877038,  1884213176,
      1241555985};
  std::initializer_list<int> ref_vals = {0, 0, 0, 0,         154189716, 0, 0, 0,
                                         0, 0, 0, 0,         0,         0, 0, 0,
                                         0, 0, 0, 274455501, 1182638704};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::vimin3_s32_relu));
  std::cout << "sycl::ext::intel::math::vimin3_s32_relu test pass."
            << std::endl;
}

void run_vimax3_u16x2_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x110dfc17, 0xd3d81c2,  0x26478328, 0x46ff8e92, 0x79276c08,
      0x24f4dc7f, 0x597dd09a, 0x42498472, 0xa737cc6,  0x68b68489, 0x47240e84,
      0x1109534,  0x7351608,  0x580424fb, 0x2b087207, 0x1e92fc69, 0x108e0114,
      0x6c181400, 0x2214e9f6, 0x32cad3b};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x5f10afd5, 0x4f92b2f5, 0x38a9152e, 0xb9234cb,  0x30286640,
      0x7810c4cc, 0x10041879, 0x68e5c049, 0x47280242, 0x41bedc6f, 0x52ccd887,
      0x600a5a49, 0x651dd71,  0x4d516c03, 0x4678d80c, 0x6b6db48b, 0x44eb8525,
      0x7350997,  0x11a8865d, 0x7a5f0ae6};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x39ec86b0, 0x62480d3,  0x50cf0ef3, 0x53044d0c, 0x738a8a61,
      0x71fb04ab, 0x5b8a12ec, 0x35fcd5f6, 0x6117c345, 0x4e2a0771, 0x2d3ab746,
      0x7ccd6a3b, 0x35767f69, 0x4108b435, 0x71311a75, 0x6941df41, 0x7945f7ba,
      0x622bb804, 0x2953ba46, 0x6b1296b5};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x5f10fc17, 0x4f92b2f5, 0x50cf8328, 0x53048e92, 0x79278a61,
      0x7810dc7f, 0x5b8ad09a, 0x68e5d5f6, 0x6117c345, 0x68b6dc6f, 0x52ccd887,
      0x7ccd9534, 0x3576dd71, 0x5804b435, 0x7131d80c, 0x6b6dfc69, 0x7945f7ba,
      0x6c18b804, 0x2953e9f6, 0x7a5fad3b};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::vimax3_u16x2));
  std::cout << "sycl::ext::intel::math::vimax3_u16x2 test pass." << std::endl;
}

void run_vimin3_u16x2_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x5af8e81d, 0x4884b910, 0x315549b8, 0x28025f8d, 0x13d70878,
      0x1623420e, 0x247e9333, 0x380e509c, 0x23b5da61, 0x750adc50, 0x7520b3fc,
      0x6373fbba, 0x3e9942a7, 0x44f32fb2, 0x5666e8be, 0x4706b850, 0x1ed7817c,
      0x56961023, 0x1b949968, 0xdd1b03d};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x7dd8fa99, 0x7672aa5c, 0x6a377414, 0xf8c6100,  0x479aff4,
      0x7e5bcc01, 0xede1402,  0x77aa1b63, 0x40509007, 0x26a43324, 0x19d1b41,
      0x4a21d451, 0x7b771e09, 0x23797d96, 0x3750860f, 0x4d73c81d, 0x71f25b51,
      0x2a00abed, 0x4db6864e, 0x42c1629f};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x6e5347be, 0x5b2546ec, 0x659ed0b,  0x1173b90c, 0x709fcf92,
      0x57d0c92b, 0x37be8ea7, 0x44fe8945, 0x723d16d4, 0x7669c5d,  0x53f96f6,
      0x7bb24152, 0x65e9b566, 0x75761666, 0x79efc65b, 0x784b925c, 0x729a65e,
      0x7ed3c1c1, 0x3f2451c8, 0x65c884ec};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x5af847be, 0x488446ec, 0x65949b8,  0xf8c5f8d,  0x4790878,
      0x1623420e, 0xede1402,  0x380e1b63, 0x23b516d4, 0x7663324,  0x19d1b41,
      0x4a214152, 0x3e991e09, 0x23791666, 0x3750860f, 0x4706925c, 0x7295b51,
      0x2a001023, 0x1b9451c8, 0xdd1629f};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::vimin3_u16x2));
  std::cout << "sycl::ext::intel::math::vimin3_u16x2 test pass." << std::endl;
}

void run_vimax3_u32_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x24daf71,  0x12b50bca, 0x5ef0aa1f, 0x7570ecf3, 0x23395eb0,
      0x733dbcde, 0x799cfb9a, 0x5b0d7e75, 0x510bd866, 0x4f3ee7d4, 0xdc1d4f5,
      0x1bb4c1ba, 0x4f69a93c, 0x3c2b6cc3, 0x56bc32be, 0x723e151b, 0xbf78926,
      0x72cd6e9e, 0x7d5a26c0, 0x6c98300a};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x2a402507, 0x55c7862d, 0x7773edc1, 0x6644fc32, 0x420a8be5,
      0x77a32610, 0x5807eab0, 0x646b2a6b, 0xd68dfb0,  0x6f52bda,  0x942db4b,
      0x1bf7e715, 0x7ae89134, 0x70597e28, 0x1392dcd8, 0x6d099b6,  0x6d9550,
      0x5b7b13c5, 0x2c86ec2c, 0x7bc5d400};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0xdf2ecc5,  0x33b4e782, 0x6cc1c387, 0x1a90c5fb, 0x1b81e25d,
      0x19b9740b, 0x66d5e577, 0xa8cb822,  0x6f3e0949, 0x1593ad9,  0x2b995fe0,
      0x160e60e,  0x78d4d3cf, 0x5f19d001, 0x21245be6, 0x18c781f6, 0x70cf6ca6,
      0x553a9711, 0x62a376c1, 0x6998a29b};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x2a402507, 0x55c7862d, 0x7773edc1, 0x7570ecf3, 0x420a8be5,
      0x77a32610, 0x799cfb9a, 0x646b2a6b, 0x6f3e0949, 0x4f3ee7d4, 0x2b995fe0,
      0x1bf7e715, 0x7ae89134, 0x70597e28, 0x56bc32be, 0x723e151b, 0x70cf6ca6,
      0x72cd6e9e, 0x7d5a26c0, 0x7bc5d400};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::vimax3_u32));
  std::cout << "sycl::ext::intel::math::vimax3_u32 test pass." << std::endl;
}

void run_vimin3_u32_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x5f5bd74e, 0x3bacaac,  0x7f29f996, 0x17668700, 0x720fff63,
      0x5691cd7,  0x1915f193, 0x3aa7969b, 0x1af2a45d, 0x722f08f8, 0x157dff1e,
      0x7821a2,   0x7301d8aa, 0x40355f5b, 0x15e2825e, 0x286c4d39, 0x7364949f,
      0x2e842313, 0x25a74faa, 0x47beb4b2};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x464848a4, 0x49f8bb57, 0x4b8a2c58, 0x316e9e1c, 0x7dfdd79d,
      0x493a4ab7, 0x12f2a1ec, 0xea96101,  0x7653250c, 0x727da11f, 0x51d9786d,
      0x55944319, 0x54be3cb0, 0x6c24c3b0, 0x5e34c313, 0x639ddfea, 0x7cb3d17d,
      0x375b6819, 0x524e0c76, 0x447d156e};
  std::initializer_list<unsigned> input_vals3 = {
      0,          0x6afa2284, 0x7289b708, 0x4d3386b1, 0x55ad2302, 0x1289cadb,
      0x4af84766, 0x3b1f8e73, 0x77232c97, 0x22176508, 0x2c3ec6a2, 0x72870f46,
      0x3c7fca9d, 0x809f6f6,  0x39789512, 0x37766caf, 0xb0b766,   0x13a35953,
      0x224cba54, 0x189fdf60, 0xb1d807f};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x464848a4, 0x3bacaac,  0x4b8a2c58, 0x17668700, 0x1289cadb,
      0x5691cd7,  0x12f2a1ec, 0xea96101,  0x1af2a45d, 0x2c3ec6a2, 0x157dff1e,
      0x7821a2,   0x809f6f6,  0x39789512, 0x15e2825e, 0xb0b766,   0x13a35953,
      0x224cba54, 0x189fdf60, 0xb1d807f};
  test3(device_q, input_vals1, input_vals2, input_vals3, ref_vals,
        F3(s::ext::intel::math::vimin3_u32));
  std::cout << "sycl::ext::intel::math::vimin3_u32 test pass." << std::endl;
}

void run_vimax_s16x2_relu_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x2fdd6e6c, 0x804273d,  0x10207569, 0x2bcc963b, 0x21c39b78,
      0x518c73ab, 0x1f3dfb98, 0x42099538, 0x127a0bf6, 0x63067572, 0x3875e92,
      0x9b3b43d,  0x2c8d81fa, 0x6f95643b, 0x67d6a7b2, 0x323c0a53, 0x1543890e,
      0x362ca763, 0x663e7f6a, 0x20ad83fc};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x42080de5, 0x1e789144, 0x2bfe4dd9, 0x7b498585, 0x32772164,
      0x379920ef, 0x28696e6b, 0x3d473296, 0x674851a9, 0x70c9eb5d, 0x3666dbc9,
      0x1cdb74ab, 0x196c6187, 0x3973b4a6, 0x533b7b28, 0x17b4161f, 0x3a403191,
      0x2563fe77, 0x61f93d9e, 0x8021ae2};
  std::initializer_list<unsigned> ref_vals = {
      0,          0x42086e6c, 0x1e78273d, 0x2bfe7569, 0x7b490000, 0x32772164,
      0x518c73ab, 0x28696e6b, 0x42093296, 0x674851a9, 0x70c97572, 0x36665e92,
      0x1cdb74ab, 0x2c8d6187, 0x6f95643b, 0x67d67b28, 0x323c161f, 0x3a403191,
      0x362c0000, 0x663e7f6a, 0x20ad1ae2};
  test2(device_q, input_vals1, input_vals2, ref_vals,
        F2(s::ext::intel::math::vimax_s16x2_relu));
  std::cout << "sycl::ext::intel::math::vimax_s16x2_relu test pass."
            << std::endl;
}

void run_vimax_s32_relu_test(s::queue &device_q) {
  std::initializer_list<int> input_vals1 = {
      0,          1277406999,  -476474684,  -145388661,  -648450855,
      291909782,  -237801967,  -1237016565, -458966603,  -1799115114,
      1959993912, 750728013,   2145678471,  -1423699023, 737796213,
      -888868372, -1238208610, 1657497616,  -1794814919, 804794291,
      321992532};
  std::initializer_list<int> input_vals2 = {
      0,          -2040996039, 2062183443, 69521125,    1023929017,
      739228103,  66526667,    25667654,   -1905602722, 2045294117,
      -681206888, -1121749453, 684470920,  671269546,   -863200718,
      1480089535, 1759687147,  1113608030, -1830528485, -1785005260,
      -1193744019};
  std::initializer_list<int> ref_vals = {
      0,          1277406999, 2062183443, 69521125,   1023929017, 739228103,
      66526667,   25667654,   0,          2045294117, 1959993912, 750728013,
      2145678471, 671269546,  737796213,  1480089535, 1759687147, 1657497616,
      0,          804794291,  321992532};
  test2(device_q, input_vals1, input_vals2, ref_vals,
        F2(s::ext::intel::math::vimax_s32_relu));
  std::cout << "sycl::ext::intel::math::vimax_s32_relu test pass." << std::endl;
}

void run_vimin_s16x2_relu_test(s::queue &device_q) {
  std::initializer_list<unsigned> input_vals1 = {
      0,          0x66598536, 0x79a720c7, 0x230c1039, 0x7aea6c13, 0x28cc46cd,
      0x424cee5,  0x17cb4d27, 0x3d07eab9, 0x8a1a296,  0x2c0fb9c7, 0xea36317,
      0x3f71dcb,  0x1781e987, 0x187a846,  0x4669f625, 0x719530a2, 0x677ef563,
      0x79e8b625, 0x7a2bea6,  0x289a6468};
  std::initializer_list<unsigned> input_vals2 = {
      0,          0x4c23af17, 0x4d253ac9, 0x1c666d3c, 0x1e0f5ecc, 0x8aa7475,
      0x126d845a, 0x26a69327, 0x13834eb2, 0x11663096, 0x4bfb59a0, 0xe2c91ef,
      0x13855d9a, 0x49bb5ff5, 0x5dadf7fa, 0x1b5b464b, 0x2cc37b5b, 0x6b3c516a,
      0xa8b059c,  0x74d32238, 0x306f0573};
  std::initializer_list<unsigned> ref_vals = {
      0,         0x4c230000, 0x4d2520c7, 0x1c661039, 0x1e0f5ecc, 0x8aa46cd,
      0x4240000, 0x17cb0000, 0x13830000, 0x8a10000,  0x2c0f0000, 0xe2c0000,
      0x3f71dcb, 0x17810000, 0x1870000,  0x1b5b0000, 0x2cc330a2, 0x677e0000,
      0xa8b0000, 0x7a20000,  0x289a0573};
  test2(device_q, input_vals1, input_vals2, ref_vals,
        F2(s::ext::intel::math::vimin_s16x2_relu));
  std::cout << "sycl::ext::intel::math::vimin_s16x2_relu test pass."
            << std::endl;
}

void run_vimin_s32_relu_test(s::queue &device_q) {
  std::initializer_list<int> input_vals1 = {
      0,           -513972138, 164967209,  -1552700357, -1415372018, 1466604230,
      -2114923001, -777173506, 1024337784, -1530912123, 1541681187,  1082027099,
      1111207099,  -903971399, 952543777,  -292091500,  -413825753,  -777022092,
      443228777,   1213244154, -1095725924};
  std::initializer_list<int> input_vals2 = {
      0,          -1406387233, 1045875120,  -726054484, 557993502,  -1412263759,
      -586527793, -1218389818, -124524890,  -495806067, -355972615, -553213597,
      1639191288, 366015984,   -1221185330, 289300863,  281216025,  -87256161,
      660030557,  1604018284,  -2085520141};
  std::initializer_list<int> ref_vals = {
      0, 0,          164967209, 0, 0, 0, 0, 0,         0,          0, 0,
      0, 1111207099, 0,         0, 0, 0, 0, 443228777, 1213244154, 0};
  test2(device_q, input_vals1, input_vals2, ref_vals,
        F2(s::ext::intel::math::vimin_s32_relu));
  std::cout << "sycl::ext::intel::math::vimin_s32_relu test pass." << std::endl;
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
  run_vimax3_s16x2_relu_test(device_queue);
  run_vimin3_s16x2_test(device_queue);
  run_vimin3_s16x2_relu_test(device_queue);
  run_vimax3_s32_test(device_queue);
  run_vimax3_s32_relu_test(device_queue);
  run_vimin3_s32_test(device_queue);
  run_vimin3_s32_relu_test(device_queue);
  run_vimax3_u16x2_test(device_queue);
  run_vimax3_u32_test(device_queue);
  run_vimin3_u16x2_test(device_queue);
  run_vimin3_u32_test(device_queue);
  run_vimax_s16x2_relu_test(device_queue);
  run_vimax_s32_relu_test(device_queue);
  run_vimin_s16x2_relu_test(device_queue);
  run_vimin_s32_relu_test(device_queue);
  return 0;
}
