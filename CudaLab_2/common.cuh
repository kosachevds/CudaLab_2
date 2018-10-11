#ifndef COMMON_CUH
#define COMMON_CUH

#include <vector>

#define MAX_BLOCK_SIZE 1024

bool goodMiltiplication(unsigned* a, unsigned* b, unsigned* result, size_t size);

void writeVector(std::vector<float> const& values, std::ostream& out);

#endif // COMMON_CUH
