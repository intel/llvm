#include <helpers/PiMock.hpp>

#include <vector>

// Helper holder for all the data we want to capture from mocked APIs
struct LinkingCapturesHolder {
  unsigned NumOfPiProgramCreateCalls = 0;
  unsigned NumOfPiProgramLinkCalls = 0;
  unsigned ProgramUsedToCreateKernel = 0;
  std::vector<unsigned> LinkedPrograms;

  bool LinkedProgramsContains(std::initializer_list<unsigned> Programs) {
    return std::all_of(Programs.begin(), Programs.end(), [this](unsigned Prg) {
      return std::any_of(
          LinkedPrograms.begin(), LinkedPrograms.end(),
          [Prg](unsigned LinkedPrg) { return LinkedPrg == Prg; });
    });
  }

  void clear() {
    NumOfPiProgramCreateCalls = 0;
    NumOfPiProgramLinkCalls = 0;
    ProgramUsedToCreateKernel = 0;
    LinkedPrograms.clear();
  }
};

static LinkingCapturesHolder CapturedLinkingData;

static pi_result redefined_piProgramCreate(pi_context, const void *il,
                                           size_t length, pi_program *res) {
  auto *Magic = reinterpret_cast<const unsigned char *>(il);
  *res = createDummyHandle<pi_program>(sizeof(unsigned));
  reinterpret_cast<DummyHandlePtrT>(*res)->setDataAs<unsigned>(*Magic);
  ++CapturedLinkingData.NumOfPiProgramCreateCalls;
  return PI_SUCCESS;
}

static pi_result
redefined_piProgramLink(pi_context context, pi_uint32 num_devices,
                        const pi_device *device_list, const char *options,
                        pi_uint32 num_input_programs,
                        const pi_program *input_programs,
                        void (*pfn_notify)(pi_program program, void *user_data),
                        void *user_data, pi_program *ret_program) {
  unsigned ResProgram = 1;
  for (pi_uint32 I = 0; I < num_input_programs; ++I) {
    auto Val = reinterpret_cast<DummyHandlePtrT>(input_programs[I])
                   ->getDataAs<unsigned>();
    ResProgram *= Val;
    CapturedLinkingData.LinkedPrograms.push_back(Val);
  }

  ++CapturedLinkingData.NumOfPiProgramLinkCalls;

  *ret_program = createDummyHandle<pi_program>(sizeof(unsigned));
  reinterpret_cast<DummyHandlePtrT>(*ret_program)
      ->setDataAs<unsigned>(ResProgram);
  return PI_SUCCESS;
}

static pi_result redefined_piKernelCreate(pi_program program,
                                          const char *kernel_name,
                                          pi_kernel *ret_kernel) {
  CapturedLinkingData.ProgramUsedToCreateKernel =
      reinterpret_cast<DummyHandlePtrT>(program)->getDataAs<unsigned>();
  *ret_kernel = createDummyHandle<pi_kernel>();
  return PI_SUCCESS;
}

static sycl::unittest::PiMock setupRuntimeLinkingMock() {
  sycl::unittest::PiMock Mock;

  Mock.redefine<sycl::detail::PiApiKind::piProgramCreate>(
      redefined_piProgramCreate);
  Mock.redefine<sycl::detail::PiApiKind::piProgramLink>(
      redefined_piProgramLink);
  Mock.redefine<sycl::detail::PiApiKind::piKernelCreate>(
      redefined_piKernelCreate);

  return Mock;
}
