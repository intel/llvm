#include <helpers/UrMock.hpp>

#include <vector>

// Helper holder for all the data we want to capture from mocked APIs
struct LinkingCapturesHolder {
  unsigned NumOfUrProgramCreateCalls = 0;
  unsigned NumOfUrProgramCreateWithBinaryCalls = 0;
  unsigned NumOfUrProgramLinkCalls = 0;
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
    NumOfUrProgramCreateCalls = 0;
    NumOfUrProgramLinkCalls = 0;
    ProgramUsedToCreateKernel = 0;
    LinkedPrograms.clear();
  }
};

static LinkingCapturesHolder CapturedLinkingData;

static ur_result_t redefined_urProgramCreateWithIL(void *pParams) {
  auto Params = *static_cast<ur_program_create_with_il_params_t *>(pParams);
  auto *Magic = reinterpret_cast<const unsigned char *>(*Params.ppIL);
  ur_program_handle_t *res = *Params.pphProgram;
  *res = mock::createDummyHandle<ur_program_handle_t>(sizeof(unsigned));
  reinterpret_cast<mock::dummy_handle_t>(*res)->setDataAs<unsigned>(*Magic);
  ++CapturedLinkingData.NumOfUrProgramCreateCalls;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefined_urProgramCreateWithBinary(void *pParams) {
  auto Params = *static_cast<ur_program_create_with_binary_params_t *>(pParams);
  auto *Magic = reinterpret_cast<const unsigned char *>(*Params.pppBinaries[0]);
  ur_program_handle_t *res = *Params.pphProgram;
  *res = mock::createDummyHandle<ur_program_handle_t>(sizeof(unsigned));
  reinterpret_cast<mock::dummy_handle_t>(*res)->setDataAs<unsigned>(*Magic);
  ++CapturedLinkingData.NumOfUrProgramCreateWithBinaryCalls;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefined_urProgramLinkExp(void *pParams) {
  auto Params = *static_cast<ur_program_link_exp_params_t *>(pParams);
  unsigned ResProgram = 1;
  auto Programs = *Params.pphPrograms;
  for (uint32_t I = 0; I < *Params.pcount; ++I) {
    auto Val = reinterpret_cast<mock::dummy_handle_t>(Programs[I])
                   ->getDataAs<unsigned>();
    ResProgram *= Val;
    CapturedLinkingData.LinkedPrograms.push_back(Val);
  }

  ++CapturedLinkingData.NumOfUrProgramLinkCalls;

  ur_program_handle_t *ret_program = *Params.pphProgram;
  *ret_program = mock::createDummyHandle<ur_program_handle_t>(sizeof(unsigned));
  reinterpret_cast<mock::dummy_handle_t>(*ret_program)
      ->setDataAs<unsigned>(ResProgram);
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefined_urKernelCreate(void *pParams) {
  auto Params = *static_cast<ur_kernel_create_params_t *>(pParams);
  CapturedLinkingData.ProgramUsedToCreateKernel =
      reinterpret_cast<mock::dummy_handle_t>(*Params.phProgram)
          ->getDataAs<unsigned>();
  **Params.pphKernel = mock::createDummyHandle<ur_kernel_handle_t>();
  return UR_RESULT_SUCCESS;
}

static void setupRuntimeLinkingMock() {
  mock::getCallbacks().set_replace_callback("urProgramCreateWithIL",
                                            redefined_urProgramCreateWithIL);
  mock::getCallbacks().set_replace_callback(
      "urProgramCreateWithBinary", redefined_urProgramCreateWithBinary);
  mock::getCallbacks().set_replace_callback("urProgramLinkExp",
                                            redefined_urProgramLinkExp);
  mock::getCallbacks().set_replace_callback("urKernelCreate",
                                            redefined_urKernelCreate);
}
