// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test checks handling of parallel_for() accepting nd_range and
// two or more reductions.

#include "reduction_utils.hpp"

using namespace sycl;

template <typename... Ts> class KNameGroup;
template <typename T, bool B> class KName;

constexpr access::mode RW = access::mode::read_write;
constexpr access::mode DW = access::mode::discard_write;

template <typename RangeT>
void printNVarsTestLabel(const RangeT &Range, bool ToCERR = false) {
  std::ostream &OS = ToCERR ? std::cerr : std::cout;
  OS << (ToCERR ? "Error" : "Start") << ", Range=" << Range;
  if (!ToCERR)
    OS << std::endl;
}

template <typename T, bool IsUsm, usm::alloc AllocType> auto InitOut(queue &Q) {
  if constexpr (IsUsm)
    return malloc<T>(1, Q, AllocType);
  else
    return buffer<T, 1>{1};
}

template <class T, class BinaryOperation, bool IsUSM,
          usm::alloc AllocType = usm::alloc::unknown,
          class PropListTy = property_list>
struct Red {
  Red(queue &Q, size_t NWorkItems, T IdentityVal, T InitVal,
      BinaryOperation BOp, PropListTy PropList = {})
      : Q(Q), NWorkItems(NWorkItems), IdentityVal(IdentityVal),
        InitVal(InitVal), BOp(BOp), PropList(PropList), InBuf(NWorkItems),
        Mem(InitOut<T, IsUSM, AllocType>(Q)) {}

  ~Red() {
    if constexpr (IsUSM) {
      free(Mem, Q);
    }
  }

  void init() {
    initInputData(InBuf, CorrectOut, IdentityVal, BOp, NWorkItems);
    if (!PropList.template has_property<
            property::reduction::initialize_to_identity>())
      CorrectOut = BOp(CorrectOut, InitVal);

    if constexpr (IsUSM) {
      if constexpr (AllocType == usm::alloc::device) {
        Q.single_task([InitVal = this->InitVal, Mem = this->Mem]() {
           *Mem = InitVal;
         }).wait();
      } else {
        *Mem = InitVal;
      }
    } else {
      host_accessor Acc(Mem, sycl::write_only);
      Acc[0] = InitVal;
    }
  }

  auto createRed(handler &CGH) {
    if constexpr (IsUSM)
      return reduction(Mem, IdentityVal, BOp, PropList);
    else
      return reduction(Mem, CGH, IdentityVal, BOp, PropList);
  }

  int checkResult(nd_range<1> NDR) {
    auto Out = [this]() {
      if constexpr (IsUSM) {
        if constexpr (AllocType == usm::alloc::device) {
          buffer<T, 1> B(1);
          Q.submit([&](handler &CGH) {
            accessor A(B, CGH, sycl::write_only);
            CGH.copy(Mem, A);
          });
          return host_accessor{B}[0];
        } else {
          return *Mem;
        }
      } else {
        return host_accessor{Mem}[0];
      }
    }();
    return checkResults(Q, BOp, NDR, Out, CorrectOut);
  }

  queue &Q;
  size_t NWorkItems;
  T IdentityVal;
  T InitVal;
  BinaryOperation BOp;
  PropListTy PropList;
  buffer<T, 1> InBuf;
  T CorrectOut;
  std::conditional_t<IsUSM, T *, buffer<T, 1>> Mem;
};

template <bool IsUSM, usm::alloc AllocType = usm::alloc::unknown>
struct RedFactory {
  template <class T, class BinaryOperation, class PropListTy = property_list>
  auto get(queue &Q, size_t NWorkItems, T IdentityVal, T InitVal,
           BinaryOperation BOp, PropListTy PropList = {}) {
    return Red<T, BinaryOperation, IsUSM, AllocType, PropListTy>(
        Q, NWorkItems, IdentityVal, InitVal, BOp, PropList);
  }
};

template <class Name, class... RedTys>
int test(queue &Q, size_t NWorkItems, size_t WGSize, RedTys... Reds) {
  auto NDR = nd_range<1>{range<1>(NWorkItems), range<1>{WGSize}};
  printNVarsTestLabel<>(NDR);

  (Reds.init(), ...);

  Q.submit([&](handler &CGH) {
     auto InAcc = sycl::detail::make_tuple(
         accessor(Reds.InBuf, CGH, sycl::read_only)...);
     auto SyclReds = std::forward_as_tuple(Reds.createRed(CGH)...);
     std::apply(
         [&](auto... SyclReds) {
           CGH.parallel_for<Name>(
               NDR, SyclReds..., [=](nd_item<1> NDIt, auto &...Reducers) {
                 static_assert(sizeof...(Reducers) == 4 ||
                               sizeof...(Reducers) == 2);
                 // No C++20, so don't have explicit template param lists in
                 // lambda and can't unfold std::integer_sequence to write
                 // generic code here.
                 auto ReducersTuple = std::forward_as_tuple(Reducers...);
                 size_t I = NDIt.get_global_id(0);

                 std::get<0>(ReducersTuple).combine(std::get<0>(InAcc)[I]);
                 std::get<1>(ReducersTuple).combine(std::get<1>(InAcc)[I]);
                 if constexpr (sizeof...(Reds) == 4) {
                   std::get<2>(ReducersTuple).combine(std::get<2>(InAcc)[I]);
                   std::get<3>(ReducersTuple).combine(std::get<3>(InAcc)[I]);
                 }

                 return;
               });
         },
         SyclReds);
   }).wait();

  int NumErrors = (0 + ... + Reds.checkResult(NDR));
  return NumErrors;
}

int main() {
  queue Q;
  auto Dev = Q.get_device();
  printDeviceInfo(Q);

  constexpr bool UseBuf = false;
  constexpr bool UseUSM = true;
  int Error = 0;

  size_t GSize = 16;
  size_t WGSize = 16;
  if (Dev.get_info<info::device::usm_shared_allocations>())
    Error += test<class Case1>(
        Q, GSize, WGSize,
        RedFactory<UseBuf>{}.get<float>(Q, GSize, 0, 1000, std::plus<>{},
                                        init_to_identity()),
        RedFactory<UseBuf>{}.get<int>(Q, GSize, 0, 2000, std::plus<>{}),
        RedFactory<UseBuf>{}.get<short>(Q, GSize, 0, 4000, std::bit_or<>{}),
        RedFactory<UseUSM, usm::alloc::shared>{}.get<int>(Q, GSize, 0, 8000,
                                                          std::bit_xor<>{}));

  GSize = 5 * (256 + 1);
  WGSize = 5;
  auto Add = [](auto x, auto y) { return (x + y); };
  if (Dev.get_info<info::device::usm_device_allocations>())
    Error += test<class Case2>(
        Q, GSize, WGSize,
        RedFactory<UseBuf>{}.get<float>(Q, GSize, 0, 1000, std::plus<>{}),
        RedFactory<UseBuf>{}.get<int>(Q, GSize, 0, 2000, std::plus<>{}),
        RedFactory<UseBuf>{}.get<short>(Q, GSize, 0, 4000, Add,
                                        init_to_identity()),
        RedFactory<UseUSM, usm::alloc::device>{}.get<int>(
            Q, GSize, 0, 8000, std::plus<>{}, init_to_identity()));

  // Use buffers only to verify same mangled kernel name isn't used twice inside
  // implementation.
  Error += test<class Case3>(
      Q, GSize, WGSize,
      RedFactory<UseBuf>{}.get<float>(Q, GSize, 0, 1000, std::plus<>{}),
      RedFactory<UseBuf>{}.get<int>(Q, GSize, 0, 2000, std::plus<>{}));

  printFinalStatus(Error);
  return Error;
}
