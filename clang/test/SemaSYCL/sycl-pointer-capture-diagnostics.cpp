// RUN: %clang_cc1  -fsycl -triple spir64 -fsycl-is-device -verify -fsyntax-only  %s
//
// Pointer variables captured by kernel lambda are checked.
// Ensure those diagnostics are working correctly.

namespace std {
class type_info;
typedef __typeof__(sizeof(int)) size_t;
} // namespace std

inline namespace cl {
namespace sycl {

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc(); //#call_kernelFunc
}

typedef int device;
typedef int context;
typedef double queue;

//-- Mock USM memory allocation functions.
namespace usm {
enum class alloc { host,
                   device,
                   shared,
                   unknown };
} // namespace usm

void *malloc(std::size_t sz, const device &dev, const context &ctxt, cl::sycl::usm::alloc kind);
void *malloc(std::size_t sz, const queue &q, cl::sycl::usm::alloc kind);
void *malloc_device(std::size_t sz, const device &dev, const context &ctxt);
void *malloc_device(std::size_t sz, const queue &q);
void *malloc_shared(std::size_t sz, const device &dev, const context &ctxt);
void *malloc_shared(std::size_t sz, const queue &q);
void *malloc_host(std::size_t sz, const context &ctxt);
void *malloc_host(std::size_t sz, const queue &q);
void *aligned_alloc(std::size_t alignment, std::size_t sz, const device &dev, const context &ctxt, cl::sycl::usm::alloc kind);
void *aligned_alloc(std::size_t alignment, std::size_t sz, const queue &q, cl::sycl::usm::alloc kind);
void *aligned_alloc_device(std::size_t alignment, std::size_t sz, const device &dev, const context &ctxt);
void *aligned_alloc_device(std::size_t alignment, std::size_t sz, const queue &q);
void *aligned_alloc_shared(std::size_t alignment, std::size_t sz, const device &dev, const context &ctxt);
void *aligned_alloc_shared(std::size_t alignment, std::size_t sz, const queue &q);
void *aligned_alloc_host(std::size_t alignment, std::size_t sz, const context &ctxt);
void *aligned_alloc_host(std::size_t alignment, std::size_t sz, const queue &q);

//template form
template <typename T>
T *malloc_shared(std::size_t Count, const device &Dev, const context &Ctxt) {
  return static_cast<T *>(malloc_shared(Count * sizeof(T), Dev, Ctxt));
}

} // namespace sycl
} // namespace cl

void *malloc(std::size_t sz);
void *calloc(std::size_t num, std::size_t sz);
//-- End Mocks

// User functions that might allocate memory in some way unknown to us.
float *unknownFunc();

void allocateUSMByHandle(float **pointerHandle, std::size_t sz, sycl::device &dev, sycl::context &ctxt) {
  float *mem = sycl::malloc_shared<float>(sz, dev, ctxt);
  *pointerHandle = mem;
}

struct Mesh {
  float a;
};

float calledFromLambda(float *first) {
  return first[0];
}

int main(int argc, char **argv) {

  int device = 0, context = 0;
  double queue = 0;

  //-- Declarations

  //-- various bad pointers
  float stackFloat = 20.0;
  float *stackFloatP = &stackFloat; //#decl_stackFloatP

  float *frenemy = stackFloatP; //#decl_frenemy
  frenemy++;

  // expected-warning@+1 {{cast to 'float *' from smaller integer type 'int'}}
  float *fromParam = (float *)(argc); //#decl_fromParam

  // std::string is already caught by 'non-trivially copy constructible' check.
  // so we only worry about literal strings.
  auto stringLiteral = "omgwtf"; //#decl_stringLiteral

  float *apocryphal = unknownFunc(); //#decl_apocryphal
  float *usmByHandle;
  allocateUSMByHandle(&usmByHandle, 10, device, context);

  //-- structs
  Mesh stackMesh;
  stackMesh.a = 31.0;
  Mesh *stackMeshP = &stackMesh; //#decl_stackMeshP
  Mesh *stackMeshP2;
  Mesh *mallocMeshP = static_cast<Mesh *>(malloc(sizeof(Mesh)));                            //#decl_mallocMeshP
  Mesh *mallocMeshP2 = static_cast<Mesh *>(malloc(sizeof(Mesh)));                           //#decl_mallocMeshP2
  Mesh *usmMeshP = static_cast<Mesh *>(sycl::malloc_shared(sizeof(Mesh), device, context)); //#decl_usmMeshP
  Mesh *usmMeshP2 = sycl::malloc_shared<Mesh>(1, device, context);                          //#decl_usmMeshP2

  //-- malloc
  float *mallocFloatP = static_cast<float *>(malloc(sizeof(float) * 2));  //#decl_mallocFloatP
  float *mallocFloatP2 = static_cast<float *>(malloc(sizeof(float) * 2)); //#decl_mallocFloatP2
  float *callocFloatP = static_cast<float *>(calloc(2, sizeof(float)));   //#decl_callocFloatP
  float *callocFloatP2 = static_cast<float *>(calloc(2, sizeof(float)));  //#decl_callocFloatP2

  //usm
  float *usmSharedP = static_cast<float *>(sycl::malloc_shared(sizeof(float), device, context));
  float *usmSharedP2 = static_cast<float *>(sycl::malloc_shared(sizeof(float), queue));
  float *usmSharedP3 = static_cast<float *>(sycl::malloc(sizeof(float), device, context, cl::sycl::usm::alloc::shared));
  float *usmSharedP4 = static_cast<float *>(sycl::malloc(sizeof(float), queue, cl::sycl::usm::alloc::shared));
  float *usmSharedP5 = sycl::malloc_shared<float>(1, device, context);

  float *usmShAlignP = static_cast<float *>(sycl::aligned_alloc_shared(1, sizeof(float), device, context));
  float *usmShAlignP2 = static_cast<float *>(sycl::aligned_alloc_shared(1, sizeof(float), queue));
  float *usmShAlignP3 = static_cast<float *>(sycl::aligned_alloc(1, sizeof(float), device, context, cl::sycl::usm::alloc::shared));
  float *usmShAlignP4 = static_cast<float *>(sycl::aligned_alloc(1, sizeof(float), queue, cl::sycl::usm::alloc::shared));

  float *usmHostP = static_cast<float *>(sycl::malloc_host(sizeof(float), context));
  float *usmHostP2 = static_cast<float *>(sycl::malloc_host(sizeof(float), queue));
  float *usmHostP3 = static_cast<float *>(sycl::malloc(sizeof(float), device, context, cl::sycl::usm::alloc::host));
  float *usmHostP4 = static_cast<float *>(sycl::malloc(sizeof(float), queue, cl::sycl::usm::alloc::host));

  float *usmHoAlignP = static_cast<float *>(sycl::aligned_alloc_host(1, sizeof(float), context));
  float *usmHoAlignP2 = static_cast<float *>(sycl::aligned_alloc_host(1, sizeof(float), queue));
  float *usmHoAlignP3 = static_cast<float *>(sycl::aligned_alloc(1, sizeof(float), device, context, cl::sycl::usm::alloc::host));
  float *usmHoAlignP4 = static_cast<float *>(sycl::aligned_alloc(1, sizeof(float), queue, cl::sycl::usm::alloc::host));

  float *usmDeviceP = static_cast<float *>(sycl::malloc_device(sizeof(float), device, context));
  float *usmDeviceP2 = static_cast<float *>(sycl::malloc_device(sizeof(float), queue));
  float *usmDeviceP3 = static_cast<float *>(sycl::malloc(sizeof(float), device, context, cl::sycl::usm::alloc::device));
  float *usmDeviceP4 = static_cast<float *>(sycl::malloc(sizeof(float), queue, cl::sycl::usm::alloc::device));

  cl::sycl::kernel_single_task<class AName>([=]() {
    // --- Captures

    //-- various bad pointers

    // expected-note@#call_kernelFunc {{called by 'kernel_single_task<AName, (lambda}}
    // expected-note@#decl_mallocFloatP {{Declared here.}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    calledFromLambda(mallocFloatP);

    // expected-note@#decl_stackFloatP {{Declared here.}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    stackFloatP[0] = 30.0;

    // expected-note@#decl_frenemy {{Declared here.}}
    // expected-note@+1 {{Unknown memory reference in SYCL device kernel. Be sure memory was allocated with USM (malloc_shared, etc).}}
    frenemy[0] = 40.0;

    // expected-note@#decl_fromParam {{Declared here.}}
    // expected-note@+1 {{Unknown memory reference in SYCL device kernel. Be sure memory was allocated with USM (malloc_shared, etc).}}
    fromParam[0] = 70.0;

    // expected-note@#decl_stringLiteral {{Declared here.}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    char x = stringLiteral[0];

    // expected-note@#decl_apocryphal {{Declared here.}}
    // expected-note@+1 {{Unknown memory reference in SYCL device kernel. Be sure memory was allocated with USM (malloc_shared, etc).}}
    apocryphal[0] = 88.0;
    // expected-note@+1 {{Unknown memory reference in SYCL device kernel. Be sure memory was allocated with USM (malloc_shared, etc).}}
    usmByHandle[0] = 89.0; // Unfortunately, while this might actually be USM, we cannot determine that, and issue a note reminding the user.

    //-- structs

    // expected-note@#decl_stackMeshP {{Declared here.}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    float smpa = stackMeshP->a;

    // expected-note@+1 {{Unknown memory reference in SYCL device kernel. Be sure memory was allocated with USM (malloc_shared, etc).}}
    stackMeshP2->a = 34.0;

    // expected-note@#decl_mallocMeshP {{Declared here.}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    float mmpa = mallocMeshP->a;

    // expected-note@#decl_mallocMeshP2 {{Declared here.}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    mallocMeshP2->a = 45.0;

    //-- malloc

    // expected-note@#decl_mallocFloatP2 {{Declared here.}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    mallocFloatP2[0] = 80;

    // expected-note@#decl_callocFloatP {{Declared here.}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    callocFloatP[0] = 80;

    // expected-note@#decl_callocFloatP2 {{Declared here.}}
    // expected-error@+1 {{Illegal memory reference in SYCL device kernel. Use USM (malloc_shared, etc) instead.}}
    float someValue = *callocFloatP2;

    // --- Only the first capture of a pointer emits anything. So these accesses will NOT emit redundant diagnostics.
    calledFromLambda(mallocFloatP);
    stackFloatP[0] = 31.0;
    frenemy[0] = 41.0;
    fromParam[0] = 71.0;
    char y = stringLiteral[0];
    apocryphal[0] = 88.1;
    float smpa2 = stackMeshP->a;
    stackMeshP2->a = 34.2;
    float mmpa2 = mallocMeshP->a;
    mallocMeshP2->a = 45.2;
    mallocFloatP2[0] = 81;
    callocFloatP[0] = 81;
    float someOtherValue = *callocFloatP2;

    // --- These captures all use USM, and should pass without any notes or errors.
    calledFromLambda(usmSharedP);
    float umpa = usmMeshP->a;
    usmMeshP2->a = 61.0;
    usmSharedP[0] = 1;
    usmSharedP2[0] = 1;
    usmSharedP3[0] = 1;
    usmSharedP4[0] = 1;
    usmSharedP5[0] = 1;
    usmShAlignP[0] = 1;
    usmShAlignP2[0] = 1;
    usmShAlignP3[0] = 1;
    usmShAlignP4[0] = 1;
    usmHostP[0] = 1;
    usmHostP2[0] = 1;
    usmHostP3[0] = 1;
    usmHostP4[0] = 1;
    usmHoAlignP[0] = 1;
    usmHoAlignP2[0] = 1;
    usmHoAlignP3[0] = 1;
    usmHoAlignP4[0] = 1;
    usmDeviceP[0] = 1;
    usmDeviceP2[0] = 1;
    usmDeviceP3[0] = 1;
    usmDeviceP4[0] = 1;
  });

  auto noProblemLambda = [=]() {
    // --- Outside a SYCL context no errors are emitted.
    calledFromLambda(mallocFloatP);
    calledFromLambda(usmSharedP);
    stackFloatP[0] = 30.0;
    frenemy[0] = 40.0;
    fromParam[0] = 70.0;
    char x = stringLiteral[0];
    apocryphal[0] = 89.0;
    usmByHandle[0] = 90.0;
    float smpa = stackMeshP->a;
    stackMeshP2->a = 34.0;
    float mmpa = mallocMeshP->a;
    mallocMeshP2->a = 45.0;
    mallocFloatP2[0] = 80;
    callocFloatP[0] = 80;
    float someValue = *callocFloatP2;
  };
  noProblemLambda();

  return 0;
}