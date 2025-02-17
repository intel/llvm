#include <iostream>
#include <string>

static int Check(int *Data, int Expected, size_t Index,
                 const std::string &TestName) {
  if (Data[Index] == Expected)
    return 0;
  std::cout << "Failed " << TestName << " at index " << Index << " : "
            << Data[Index] << " != " << Expected << std::endl;
  return 1;
}

static int Check(int &Data, int Expected, const std::string &TestName) {
  if (Data == Expected)
    return 0;
  std::cout << "Failed " << TestName << " : " << Data << " != " << Expected
            << std::endl;
  return 1;
}
