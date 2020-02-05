#include <CL/sycl.hpp>

// A fake command class used for testing
class FakeCommand : public cl::sycl::detail::Command {
public:
  FakeCommand(cl::sycl::detail::QueueImplPtr Queue,
              cl::sycl::detail::Requirement Req)
      : Command{cl::sycl::detail::Command::ALLOCA, Queue},
        MRequirement{std::move(Req)} {}

  void printDot(std::ostream &Stream) const override {}

  const cl::sycl::detail::Requirement *getRequirement() const final {
    return &MRequirement;
  };

  cl_int enqueueImp() override { return MRetVal; }

  cl_int MRetVal = CL_SUCCESS;

protected:
  cl::sycl::detail::Requirement MRequirement;
};
