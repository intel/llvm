//==------------------------- WeakObject.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

#include <gtest/gtest.h>

template <typename SyclObjT> struct WeakObjectCheckExpired {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj{Obj};
    sycl::ext::oneapi::weak_object<SyclObjT> NullWeakObj;

    EXPECT_FALSE(WeakObj.expired());
    EXPECT_TRUE(NullWeakObj.expired());
  }
};

template <typename SyclObjT> struct WeakObjectCheckTryLock {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj{Obj};
    sycl::ext::oneapi::weak_object<SyclObjT> NullWeakObj;

    std::optional<SyclObjT> TLObj = WeakObj.try_lock();
    std::optional<SyclObjT> TLNull = NullWeakObj.try_lock();

    EXPECT_TRUE(TLObj.has_value());
    EXPECT_FALSE(TLNull.has_value());

    EXPECT_TRUE(TLObj.value() == Obj);
  }
};

template <typename SyclObjT> struct WeakObjectCheckLock {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj{Obj};
    sycl::ext::oneapi::weak_object<SyclObjT> NullWeakObj;

    SyclObjT LObj = WeakObj.lock();
    EXPECT_TRUE(LObj == Obj);

    try {
      SyclObjT LNull = NullWeakObj.lock();
      FAIL() << "Locking empty weak object did not throw.";
    } catch (sycl::exception &E) {
      EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::invalid))
          << "Unexpected thrown error code.";
    }
  }
};

template <typename SyclObjT> struct WeakObjectCheckOwnerBefore {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj{Obj};
    sycl::ext::oneapi::weak_object<SyclObjT> NullWeakObj;

    EXPECT_TRUE((WeakObj.owner_before(NullWeakObj) &&
                 !NullWeakObj.owner_before(WeakObj)) ||
                (NullWeakObj.owner_before(WeakObj) &&
                 !WeakObj.owner_before(NullWeakObj)));

    EXPECT_FALSE(WeakObj.owner_before(Obj));
    EXPECT_FALSE(Obj.ext_oneapi_owner_before(WeakObj));

    EXPECT_FALSE(Obj.ext_oneapi_owner_before(Obj));
  }
};

template <typename SyclObjT> struct WeakObjectCheckOwnerLess {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj{Obj};
    sycl::ext::oneapi::weak_object<SyclObjT> NullWeakObj;
    sycl::ext::oneapi::owner_less<SyclObjT> Comparator;

    EXPECT_TRUE((Comparator(WeakObj, NullWeakObj) &&
                 !Comparator(NullWeakObj, WeakObj)) ||
                (Comparator(NullWeakObj, WeakObj) &&
                 !Comparator(WeakObj, NullWeakObj)));

    EXPECT_FALSE(Comparator(WeakObj, Obj));
    EXPECT_FALSE(Comparator(Obj, WeakObj));
  }
};

template <typename SyclObjT> struct WeakObjectCheckReset {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj{Obj};
    sycl::ext::oneapi::weak_object<SyclObjT> NullWeakObj;

    WeakObj.reset();
    EXPECT_TRUE(WeakObj.expired());
    EXPECT_FALSE(WeakObj.owner_before(NullWeakObj));
    EXPECT_FALSE(NullWeakObj.owner_before(WeakObj));

    std::optional<SyclObjT> TLObj = WeakObj.try_lock();
    EXPECT_FALSE(TLObj.has_value());

    try {
      SyclObjT LObj = WeakObj.lock();
      FAIL() << "Locking reset weak object did not throw.";
    } catch (sycl::exception &E) {
      EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::invalid))
          << "Unexpected thrown error code.";
    }
  }
};

template <typename SyclObjT> struct WeakObjectCheckOwnerLessMulti {
  void operator()(SyclObjT Obj1, SyclObjT Obj2) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj1{Obj1};
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj2{Obj2};
    sycl::ext::oneapi::owner_less<SyclObjT> Comparator;

    EXPECT_TRUE(
        (Comparator(WeakObj1, WeakObj2) && !Comparator(WeakObj2, WeakObj1)) ||
        (Comparator(WeakObj2, WeakObj1) && !Comparator(WeakObj1, WeakObj2)));

    EXPECT_FALSE(Comparator(WeakObj1, Obj1));
    EXPECT_FALSE(Comparator(Obj1, WeakObj1));

    EXPECT_FALSE(Comparator(WeakObj2, Obj2));
    EXPECT_FALSE(Comparator(Obj2, WeakObj2));
  }
};

template <typename SyclObjT> struct WeakObjectCheckOwnerBeforeMulti {
  void operator()(SyclObjT Obj1, SyclObjT Obj2) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj1{Obj1};
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj2{Obj2};

    EXPECT_TRUE(
        (WeakObj1.owner_before(WeakObj2) && !WeakObj2.owner_before(WeakObj1)) ||
        (WeakObj2.owner_before(WeakObj1) && !WeakObj1.owner_before(WeakObj2)));

    EXPECT_FALSE(WeakObj1.owner_before(Obj1));
    EXPECT_FALSE(Obj1.ext_oneapi_owner_before(WeakObj1));

    EXPECT_FALSE(WeakObj2.owner_before(Obj2));
    EXPECT_FALSE(Obj2.ext_oneapi_owner_before(WeakObj2));

    EXPECT_TRUE((Obj1.ext_oneapi_owner_before(Obj2) &&
                 !Obj2.ext_oneapi_owner_before(Obj1)) ||
                (Obj2.ext_oneapi_owner_before(Obj1) &&
                 !Obj1.ext_oneapi_owner_before(Obj2)));
  }
};

template <typename SyclObjT> struct WeakObjectCheckOwnerLessMap {
  void operator()(SyclObjT Obj1, SyclObjT Obj2) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj1{Obj1};
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj2{Obj2};

    std::map<sycl::ext::oneapi::weak_object<SyclObjT>, int,
             sycl::ext::oneapi::owner_less<SyclObjT>>
        Map;
    Map[WeakObj1] = 1;
    Map[WeakObj2] = 2;

    EXPECT_EQ(Map.size(), (size_t)2);
    EXPECT_EQ(Map[WeakObj1], 1);
    EXPECT_EQ(Map[WeakObj2], 2);
    EXPECT_EQ(Map[Obj1], 1);
    EXPECT_EQ(Map[Obj2], 2);

    Map[WeakObj1] = 2;
    Map[WeakObj2] = 3;

    EXPECT_EQ(Map.size(), (size_t)2);
    EXPECT_EQ(Map[WeakObj1], 2);
    EXPECT_EQ(Map[WeakObj2], 3);
    EXPECT_EQ(Map[Obj1], 2);
    EXPECT_EQ(Map[Obj2], 3);

    Map[Obj1] = 5;
    Map[Obj2] = 6;

    EXPECT_EQ(Map.size(), (size_t)2);
    EXPECT_EQ(Map[WeakObj1], 5);
    EXPECT_EQ(Map[WeakObj2], 6);
    EXPECT_EQ(Map[Obj1], 5);
    EXPECT_EQ(Map[Obj2], 6);

    Map[sycl::ext::oneapi::weak_object<SyclObjT>{Obj1}] = 10;
    Map[sycl::ext::oneapi::weak_object<SyclObjT>{Obj2}] = 13;

    EXPECT_EQ(Map.size(), (size_t)2);
    EXPECT_EQ(Map[WeakObj1], 10);
    EXPECT_EQ(Map[WeakObj2], 13);
    EXPECT_EQ(Map[Obj1], 10);
    EXPECT_EQ(Map[Obj2], 13);
  }
};

template <template <typename> typename CallableT>
void runTest(sycl::unittest::PiMock &Mock) {
  sycl::platform Plt = Mock.getPlatform();
  sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Q{Dev};
  sycl::event E;
  sycl::kernel_id KId = sycl::get_kernel_id<TestKernel<>>();
  sycl::kernel_bundle KB =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  sycl::buffer<int, 1> Buf1D{1};
  sycl::buffer<int, 2> Buf2D{sycl::range<2>{1, 2}};
  sycl::buffer<int, 3> Buf3D{sycl::range<3>{1, 2, 3}};
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
  CallableT<decltype(PAcc1D)>()(PAcc1D);
  CallableT<decltype(PAcc2D)>()(PAcc2D);
  CallableT<decltype(PAcc3D)>()(PAcc3D);
  CallableT<decltype(HAcc1D)>()(HAcc1D);
  CallableT<decltype(HAcc2D)>()(HAcc2D);
  CallableT<decltype(HAcc3D)>()(HAcc3D);

  Q.submit([&](sycl::handler &CGH) {
    sycl::accessor DAcc1D{Buf1D, CGH, sycl::read_only};
    sycl::accessor DAcc2D{Buf2D, CGH, sycl::read_only};
    sycl::accessor DAcc3D{Buf3D, CGH, sycl::read_only};
    sycl::local_accessor<int, 1> LAcc1D{1, CGH};
    sycl::local_accessor<int, 2> LAcc2D{sycl::range<2>{1, 2}, CGH};
    sycl::local_accessor<int, 3> LAcc3D{sycl::range<3>{1, 2, 3}, CGH};

    CallableT<decltype(DAcc1D)>()(DAcc1D);
    CallableT<decltype(DAcc2D)>()(DAcc2D);
    CallableT<decltype(DAcc3D)>()(DAcc3D);
    CallableT<decltype(LAcc1D)>()(LAcc1D);
    CallableT<decltype(LAcc2D)>()(LAcc2D);
    CallableT<decltype(LAcc3D)>()(LAcc3D);
  });
}

template <template <typename> typename CallableT>
void runTestMulti(sycl::unittest::PiMock &Mock) {
  sycl::platform Plt = Mock.getPlatform();
  sycl::device Dev = Plt.get_devices()[0];

  sycl::context Ctx1{Dev};
  sycl::context Ctx2{Dev};
  sycl::queue Q1{Dev};
  sycl::queue Q2{Dev};
  sycl::event E1;
  sycl::event E2;
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

  CallableT<decltype(Ctx1)>()(Ctx1, Ctx2);
  CallableT<decltype(Q1)>()(Q1, Q2);
  CallableT<decltype(E1)>()(E1, E2);
  CallableT<decltype(KB1)>()(KB1, KB2);
  CallableT<decltype(Buf1D1)>()(Buf1D1, Buf1D2);
  CallableT<decltype(Buf2D1)>()(Buf2D1, Buf2D2);
  CallableT<decltype(Buf3D1)>()(Buf3D1, Buf3D2);
  CallableT<decltype(PAcc1D1)>()(PAcc1D1, PAcc1D2);
  CallableT<decltype(PAcc2D1)>()(PAcc2D1, PAcc2D2);
  CallableT<decltype(PAcc3D1)>()(PAcc3D1, PAcc3D2);
  CallableT<decltype(HAcc1D1)>()(HAcc1D1, HAcc1D2);
  CallableT<decltype(HAcc2D1)>()(HAcc2D1, HAcc2D2);
  CallableT<decltype(HAcc3D1)>()(HAcc3D1, HAcc3D2);

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

    CallableT<decltype(DAcc1D1)>()(DAcc1D1, DAcc1D2);
    CallableT<decltype(DAcc2D1)>()(DAcc2D1, DAcc2D2);
    CallableT<decltype(DAcc3D1)>()(DAcc3D1, DAcc3D2);
    CallableT<decltype(LAcc1D1)>()(LAcc1D1, LAcc1D2);
    CallableT<decltype(LAcc2D1)>()(LAcc2D1, LAcc2D2);
    CallableT<decltype(LAcc3D1)>()(LAcc3D1, LAcc3D2);
  });
}

TEST(WeakObjectTest, WeakObjectExpired) {
  sycl::unittest::PiMock Mock;
  runTest<WeakObjectCheckExpired>(Mock);
}

TEST(WeakObjectTest, WeakObjectTryLock) {
  sycl::unittest::PiMock Mock;
  runTest<WeakObjectCheckTryLock>(Mock);
}

TEST(WeakObjectTest, WeakObjectLock) {
  sycl::unittest::PiMock Mock;
  runTest<WeakObjectCheckLock>(Mock);
}

TEST(WeakObjectTest, WeakObjectOwnerBefore) {
  sycl::unittest::PiMock Mock;
  runTest<WeakObjectCheckOwnerBefore>(Mock);
}

TEST(WeakObjectTest, WeakObjectOwnerLess) {
  sycl::unittest::PiMock Mock;
  runTest<WeakObjectCheckOwnerLess>(Mock);
}

TEST(WeakObjectTest, WeakObjectReset) {
  sycl::unittest::PiMock Mock;
  runTest<WeakObjectCheckReset>(Mock);
}

TEST(WeakObjectTest, WeakObjectOwnerLessMulti) {
  sycl::unittest::PiMock Mock;
  runTestMulti<WeakObjectCheckOwnerLessMulti>(Mock);
}

TEST(WeakObjectTest, WeakObjectOwnerBeforeMulti) {
  sycl::unittest::PiMock Mock;
  runTestMulti<WeakObjectCheckOwnerBeforeMulti>(Mock);
}

TEST(WeakObjectTest, WeakObjectOwnerLessMap) {
  sycl::unittest::PiMock Mock;
  runTestMulti<WeakObjectCheckOwnerLessMap>(Mock);
}
