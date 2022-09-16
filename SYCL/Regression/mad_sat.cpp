// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUNx: %GPU_RUN_PLACEHOLDER %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

int main() {
  sycl::queue testQueue;

  {
    const sycl::longlong3 verification3(
        9223372036854775807LL, 9223372036854775807LL, -9223372036854775808LL);

    sycl::longlong3 inputData_0(1152105081885725616LL, 8383539663869980295LL,
                                -3013159033463244495LL);
    sycl::longlong3 inputData_1(9169239286331099647LL, 8545168655265359544LL,
                                69290337040907021LL);
    sycl::longlong3 inputData_2(-5670250901301018333LL, 216462155376518854LL,
                                -7910909987096217335LL);

    sycl::buffer<sycl::longlong3, 1> buffer(1);

    testQueue.submit([&](sycl::handler &h) {
      auto resultPtr = buffer.template get_access<sycl::access::mode::write>(h);
      h.single_task<class k3>([=]() {
        resultPtr[0] = sycl::mad_sat(inputData_0, inputData_1, inputData_2);
      });
    });
    const auto HostAccessor = buffer.get_access<sycl::access::mode::read>();
    for (int i = 0; i < 3; i++)
      assert((HostAccessor[0][i] == verification3[i]) && "Incorrect result");
  }
  {
    const sycl::longlong4 verification4(
        9223372036854775807LL, 9223372036854775807LL, -9223372036854775808LL,
        9223372036854775807LL);

    sycl::longlong4 inputData_0(-4713774672458165250LL, 7161321293740295698LL,
                                -7560042360032818022LL, 1712118953348815386LL);
    sycl::longlong4 inputData_1(-5256951628950351348LL, 3094294642897896981LL,
                                4183324171724765944LL, 1726930751531248453LL);
    sycl::longlong4 inputData_2(-614349234816759997LL, -7793620271163345724LL,
                                5480991826433743823LL, -3977840325478979484LL);

    sycl::buffer<sycl::longlong4, 1> buffer(1);

    testQueue.submit([&](sycl::handler &h) {
      auto resultPtr = buffer.template get_access<sycl::access::mode::write>(h);
      h.single_task<class k4>([=]() {
        resultPtr[0] = sycl::mad_sat(inputData_0, inputData_1, inputData_2);
      });
    });
    const auto HostAccessor = buffer.get_access<sycl::access::mode::read>();
    for (int i = 0; i < 4; i++)
      assert((HostAccessor[0][i] == verification4[i]) && "Incorrect result");
  }
  {
    const sycl::longlong8 verification8(
        9223372036854775807LL, 9223372036854775807LL, -9223372036854775808LL,
        -9223372036854775808LL, 9223372036854775807LL, 9223372036854775807LL,
        -9223372036854775808LL, -9223372036854775808LL);

    sycl::longlong8 inputData_0(3002837817109371705LL, -6132505093056073745LL,
                                -2677806413031023542LL, -3906932152445696896LL,
                                -5966911996430888011LL, 487233493241732294LL,
                                8234534527416862935LL, 8302379558520488989LL);
    sycl::longlong8 inputData_1(3895748400226584336LL, -3171989754828069475LL,
                                6135091761884568657LL, 3449810579449494485LL,
                                -5153085649597103327LL, 2036067225828737775LL,
                                -2456339276147680058LL, -2321401317481120691LL);
    sycl::longlong8 inputData_2(5847800471474896191LL, 6421268696360310080LL,
                                426131359031594004LL, 3388848179800138438LL,
                                9095634920776267157LL, 3909069092545608647LL,
                                -6551917618131929798LL, -5283018165188606431LL);

    sycl::buffer<sycl::longlong8, 1> buffer(1);

    testQueue.submit([&](sycl::handler &h) {
      auto resultPtr = buffer.template get_access<sycl::access::mode::write>(h);
      h.single_task<class k8>([=]() {
        resultPtr[0] = sycl::mad_sat(inputData_0, inputData_1, inputData_2);
      });
    });
    const auto HostAccessor = buffer.get_access<sycl::access::mode::read>();
    for (int i = 0; i < 8; i++)
      assert((HostAccessor[0][i] == verification8[i]) && "Incorrect result");
  }
  {
    const sycl::longlong16 verification16(
        -9223372036854775808LL, 9223372036854775807LL, 9223372036854775807LL,
        -9223372036854775808LL, 9223372036854775807LL, 9223372036854775807LL,
        9223372036854775807LL, 9223372036854775807LL, 9223372036854775807LL,
        9223372036854775807LL, -9223372036854775808LL, 9223372036854775807LL,
        -9223372036854775808LL, 9223372036854775807LL, -9223372036854775808LL,
        -9223372036854775808LL);

    sycl::longlong16 inputData_0(
        4711072418277000515LL, -8205098172692021203LL, -7385016145788992368LL,
        5953521028589173909LL, -5219240995491769312LL, 8710496141913755416LL,
        -6685846261491268433LL, 4193173269411595542LL, -8540195959022520771LL,
        -4715465363106336895LL, -1020086937442724783LL, 4496316677230042947LL,
        1321442475247578017LL, -7374746170855359764LL, -3206370806055241163LL,
        -2175226063524462053LL);
    sycl::longlong16 inputData_1(
        -9126728881985856159LL, -8235441378758843293LL, -3529617622861997052LL,
        -4696495345590499183LL, -2446014787831249326LL, 3966377959819902357LL,
        -8707315735766590681LL, 4940281453308003965LL, -4008494233289413829LL,
        -1007875458987895243LL, 8007184939842565626LL, 7006363475270750393LL,
        -3126435375497361798LL, -2666957213164527889LL, 3425215156535282625LL,
        5057359883753713949LL);
    sycl::longlong16 inputData_2(
        -5792361016316836568LL, 1155364222481085809LL, 7552404711758320408LL,
        -9123476257323872288LL, -924920183965907175LL, 1921314238201973170LL,
        3462681782260196063LL, 7822120358287768333LL, -3130033938219713817LL,
        -3165995450630991604LL, -7647706888277832178LL, -8427901934971949821LL,
        4207763935319579681LL, 1564279736903158695LL, 3722632463806041635LL,
        939009161285897285LL);

    sycl::buffer<sycl::longlong16, 1> buffer(1);

    testQueue.submit([&](sycl::handler &h) {
      auto resultPtr = buffer.template get_access<sycl::access::mode::write>(h);

      h.single_task<class k16>([=]() {
        resultPtr[0] = sycl::mad_sat(inputData_0, inputData_1, inputData_2);
      });
    });
    const auto HostAccessor = buffer.get_access<sycl::access::mode::read>();
    for (int i = 0; i < 16; i++)
      assert((HostAccessor[0][i] == verification16[i]) && "Incorrect result");
  }
}
