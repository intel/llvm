/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCLcompat API
 *
 *  math_byte_dot_product.cpp
 *
 *  Description:
 *    Dp4a and Dp2a tests
 **************************************************************************/

// ===----------- math_dp2a_dp4a.cpp ------------ -*- C++ -* --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

// RUN: %{build} %if any-device-is-cuda %{ -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_61 %} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <syclcompat.hpp>

template <typename T, size_t N> constexpr size_t array_size(T (&)[N]) {
  return N;
}

// TODO(syclcompat-lib-reviewers): Improve the tests to ensure that the
// intrinsics are actually used and the implementation is not defaulting to the
// library implementation in CUDA devices.

template <typename T1, typename T2> struct TestCaseStorage {
  T1 a;
  T2 b;
  syclcompat::dot_product_acc_t<T1, T2> c;
  syclcompat::dot_product_acc_t<T1, T2> d;
};

enum TestType { dp2a_lo, dp2a_hi, dp4a };

template <TestType, typename T1, typename T2> struct TestCase;

template <> struct TestCase<dp2a_lo, int32_t, int32_t> {
  static constexpr TestCaseStorage<int32_t, int32_t> data[] = {
      {930681129, 370772529, 2010968336, 2009507875},
      {-182801821, 2018321974, -1344607006, -1345896544},
      {-1405866995, -56456331, 2028627921, 2032457214},
      {-2067420235, 667032387, -1549633870, -1551931432},
      {150264517, 1499579728, 1168148523, 1167250815},
      {-1488693248, 590983308, -1132841811, -1133637779},
      {1952352829, 1541328881, -867130079, -868137584},
      {1402917188, -396551268, 682657336, 684431698},
      {1060076168, 2095822351, 266994190, 266267760},
      {-597525506, 329411575, -760256038, -761517342},
  };
};

template <> struct TestCase<dp2a_lo, int32_t, uint32_t> {
  static constexpr TestCaseStorage<int32_t, uint32_t> data[] = {
      {-1784870143, 3550903701, 929114859, 926130217},
      {-906522442, 2115573780, -1285980330, -1286882122},
      {1391650851, 4107608479, 273580150, 273309541},
      {-1501013502, 3932674350, -905231285, -909141521},
      {-304683280, 2104603303, -790552087, -792451259},
      {-1341822015, 615507964, -1323598253, -1321376558},
      {351927836, 264881689, -495668280, -494617318},
      {-151229742, 3617293176, 628248961, 631228133},
      {302881625, 4164956791, -1904446304, -1907281527},
      {2037447091, 4048192261, -200189002, -196124539},
  };
};

template <> struct TestCase<dp2a_lo, uint32_t, int32_t> {
  static constexpr TestCaseStorage<uint32_t, int32_t> data[] = {
      {3526794897, 1440743042, 370074542, 364852196},
      {262513653, 298144108, 1265851732, 1270709221},
      {1130955292, -963349034, -2078791855, -2076795466},
      {2514054142, -1350622828, 257209474, 255489619},
      {2734618833, -2039216829, 1170234974, 1174711303},
      {2679502652, -552107997, 1516795981, 1513777921},
      {2178722429, 1706794257, -1207356382, -1209905573},
      {2938336684, 1853682464, 1478700448, 1479081561},
      {4131007422, 88852262, 949301283, 946133869},
      {1426380125, 1310424908, 2110346787, 2117262011},
  };
};

template <> struct TestCase<dp2a_lo, uint32_t, uint32_t> {
  static constexpr TestCaseStorage<uint32_t, uint32_t> data[] = {
      {261879580, 462533001, 1244651601, 1254025336},
      {3613440709, 39532914, 3612331201, 3620924635},
      {2613678921, 3074075559, 2197617435, 2210733821},
      {3858700825, 2932114399, 651043516, 660246528},
      {3641490311, 1203902590, 1264123439, 1271505857},
      {620567, 198432492, 1750593890, 1757851164},
      {1924357490, 2672674441, 363874491, 372965679},
      {575741870, 365675828, 4077327301, 4079479666},
      {779333090, 1461441270, 3936527378, 3949974932},
      {3047663397, 3117692984, 3095767416, 3100767768},
  };
};

template <> struct TestCase<dp2a_hi, int32_t, int32_t> {
  static constexpr TestCaseStorage<int32_t, int32_t> data[] = {
      {2033148131, 1987852344, 1836738289, 1843474575},
      {1854766635, -847369228, 570647947, 573274270},
      {1221789280, -1504599082, 2039564501, 2038018823},
      {1815893957, 522593320, -1194398972, -1192686202},
      {-942058619, -1694947839, -1791401709, -1790085126},
      {1261876252, -722935661, -401441440, -401822344},
      {-1276948036, -2045446196, 883626458, 886422108},
      {-1043904041, 1660095151, 924853314, 923046533},
      {1873342481, -183952166, 1422494064, 1422142929},
      {1548579097, 388816020, 1306723060, 1308540459},
  };
};
template <> struct TestCase<dp2a_hi, int32_t, uint32_t> {
  static constexpr TestCaseStorage<int32_t, uint32_t> data[] = {
      {925779231, 2297216285, -2134129287, -2128032131},
      {1226362493, 592978070, 1394319934, 1393859454},
      {-820606485, 3315032306, -1946036979, -1953068392},
      {865550467, 2594266420, 684086152, 688778945},
      {2042373655, 2279820469, 330650825, 337071442},
      {-803475029, 3557524416, 570180628, 567540937},
      {-1920282536, 4207418946, -179074286, -188839786},
      {-1611807508, 2012850000, -45410323, -52103004},
      {-209217908, 3249694139, -1047805020, -1053226557},
      {-938134420, 4023147013, -1637223186, -1637906791},
  };
};

template <> struct TestCase<dp2a_hi, uint32_t, int32_t> {
  static constexpr TestCaseStorage<uint32_t, int32_t> data[] = {
      {1465064346, -987065627, 511196861, 510174688},
      {423752047, -2037616892, 1367127780, 1359169438},
      {1732089906, 1660637927, 835046327, 837441559},
      {3240032526, -687279473, 314878829, 313361935},
      {2028889232, 453690876, -1579929106, -1578835800},
      {636106821, 1932111966, -1143803023, -1142096228},
      {1744753942, 2120462197, 543738507, 552493329},
      {1952094085, 75134480, -1870017090, -1865165688},
      {1238028676, -368589994, 400410492, 400370364},
      {1678354325, 1520837888, 900538674, 898982394},
  };
};

template <> struct TestCase<dp2a_hi, uint32_t, uint32_t> {
  static constexpr TestCaseStorage<uint32_t, uint32_t> data[] = {
      {3407045239, 1034879260, 1566081712, 1573664144},
      {1019854071, 319089899, 2048645673, 2049134832},
      {3484748932, 23066577, 2279969923, 2280327476},
      {772761490, 593919853, 110217101, 113334214},
      {3040024654, 3302072533, 3503588845, 3513981095},
      {247428909, 1708258743, 3414468907, 3421226563},
      {3214691207, 2264421274, 2096321799, 2107689847},
      {1978412244, 3523914401, 3482699206, 3489153446},
      {845968593, 3600665955, 3398632658, 3406090055},
      {2655885278, 642147090, 953440990, 957702400},
  };
};

template <> struct TestCase<dp4a, int32_t, int32_t> {
  static constexpr TestCaseStorage<int32_t, int32_t> data[] = {
      {-1190208646, 231822748, 1361188354, 1361171428},
      {-1897923580, -1660380472, -882257438, -882246232},
      {-579619596, 1428550082, -686850248, -686847084},
      {1276672648, 1193117464, 963222686, 963211136},
      {-1511270552, 346453515, 539470060, 539466436},
      {-1731107400, 30416897, 1116161329, 1116166641},
      {314175584, 917356905, 1924209306, 1924227259},
      {601261287, 461003584, -332185426, -332202489},
      {451422378, 1069445579, 2077503598, 2077515898},
      {1601425114, -1009494442, -12279717, -12298140},
  };
};

template <> struct TestCase<dp4a, int32_t, uint32_t> {
  static constexpr TestCaseStorage<int32_t, uint32_t> data[] = {
      {851192907, 4159889898, -1560201465, -1560178121},
      {-383662874, 94554831, -1699007777, -1699020048},
      {319925525, 3224159406, -1636209897, -1636218115},
      {390273202, 3538403320, 1599902512, 1599908059},
      {-2133436013, 2204709798, -745513793, -745548526},
      {-1365042624, 302260610, 1683641121, 1683648451},
      {839091651, 3945553885, 18130274, 18116990},
      {-92392216, 2135215000, -886668361, -886653647},
      {-968453153, 2050948958, 1992996892, 1992963259},
      {-234768205, 3930595068, -2067724845, -2067749613},
  };
};

template <> struct TestCase<dp4a, uint32_t, int32_t> {
  static constexpr TestCaseStorage<uint32_t, int32_t> data[] = {
      {908604347, 1279608234, -1450969803, -1450975502},
      {1784598592, 892171050, -824564831, -824528375},
      {3414325281, 110856089, 1344013863, 1343984032},
      {3589641407, 1110466407, 269001016, 269060567},
      {3064317481, -1629226109, -733249792, -733278528},
      {3599941523, 2112627078, 1626729914, 1626742113},
      {1503610658, 885664480, 1900050896, 1900048832},
      {2314829379, -2127096242, 1568300547, 1568304841},
      {2817858008, -384307221, 307309401, 307306234},
      {1408389703, 1080046077, -535563057, -535530708},
  };
};

template <> struct TestCase<dp4a, uint32_t, uint32_t> {
  static constexpr TestCaseStorage<uint32_t, uint32_t> data[] = {
      {3065883002, 1618319527, 3160878852, 3160964499},
      {750408200, 2617984089, 2072985277, 2073000475},
      {1703570544, 1174656448, 1981665359, 1981717351},
      {2526801072, 968400189, 821887370, 821972228},
      {4033238565, 2506370972, 1177018849, 1177074623},
      {2340922922, 2952738658, 316397016, 316469012},
      {2559339202, 800262553, 1317311402, 1317374242},
      {991496487, 2323953615, 2007618737, 2007639899},
      {3918465905, 1041229499, 2826819834, 2826860086},
      {4028147698, 2068172524, 482675182, 482797872}};
};

template <TestType Type, typename T1, typename T2> bool test() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  using Case = TestCase<Type, T1, T2>;
  using CaseElement =
      std::remove_cv_t<std::remove_extent_t<decltype(Case::data)>>;
  using ResultT = syclcompat::dot_product_acc_t<T1, T2>;
  constexpr size_t N = array_size(Case::data);
  std::vector<ResultT> result(N);
  std::vector<CaseElement> cases(std::begin(Case::data), std::end(Case::data));
  sycl::buffer<CaseElement, 1> buffer(cases.data(), N);
  sycl::buffer<ResultT, 1> result_buffer(result.data(), N);
  sycl::queue q;
  q.submit([&](sycl::handler &handler) {
    sycl::accessor src(buffer, handler, sycl::read_only);
    sycl::accessor res(result_buffer, handler, sycl::write_only);
    handler.parallel_for(N, [=](sycl::id<1> i) {
      if constexpr (Type == dp2a_lo)
        res[i] = syclcompat::dp2a_lo<T1, T2>(src[i].a, src[i].b, src[i].c);
      else if constexpr (Type == dp2a_hi)
        res[i] = syclcompat::dp2a_hi<T1, T2>(src[i].a, src[i].b, src[i].c);
      else
        res[i] = syclcompat::dp4a<T1, T2>(src[i].a, src[i].b, src[i].c);
    });
  });

  q.wait_and_throw();

  int failed = 0;
  sycl::host_accessor src(buffer, sycl::read_only);
  sycl::host_accessor res(result_buffer, sycl::read_only);

  for (int i = 0; i < N; ++i) {
    if (src[i].d != res[i]) {
      failed++;
      std::cout << "  [a = " << src[i].a << ", b = " << src[i].b
                << ", c = " << src[i].c << "] failed, expect " << src[i].d
                << " but got " << res[i] << std::endl;
    }
  }

  if (failed) {
    std::cout << "  Total: " << N << std::endl;
    std::cout << "  Success: " << N - failed << std::endl;
    std::cout << "  Failed: " << failed << std::endl;
  }

  return !failed;
}

int main() {
  bool passed = true;
  passed = test<dp2a_lo, int32_t, int32_t>() && passed;
  passed = test<dp2a_lo, int32_t, uint32_t>() && passed;
  passed = test<dp2a_lo, uint32_t, int32_t>() && passed;
  passed = test<dp2a_lo, uint32_t, uint32_t>() && passed;

  passed = test<dp2a_hi, int32_t, int32_t>() && passed;
  passed = test<dp2a_hi, int32_t, uint32_t>() && passed;
  passed = test<dp2a_hi, uint32_t, int32_t>() && passed;
  passed = test<dp2a_hi, uint32_t, uint32_t>() && passed;

  passed = test<dp4a, int32_t, int32_t>() && passed;
  passed = test<dp4a, int32_t, uint32_t>() && passed;
  passed = test<dp4a, uint32_t, int32_t>() && passed;
  passed = test<dp4a, uint32_t, uint32_t>() && passed;

  assert(passed);
  return 0;
}
