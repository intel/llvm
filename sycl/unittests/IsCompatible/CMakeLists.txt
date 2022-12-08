add_sycl_unittest(IsCompatibleTests OBJECT
  #TODO: for some reason the "IsCompatible" suite fails if we place it in the
  #binary with the "SYCL2020/SpecializationConstant". Move this to SYCL2020 when
  #the failure is investigated.
  #https://github.com/intel/llvm/issues/7685
  IsCompatible.cpp
)

