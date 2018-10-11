#include "common.cuh"
#include <iostream>

extern void task1();
extern void task2();
extern void task3();

int main()
{
    //task1();
    //task2();
    task3();
}

bool goodMiltiplication(unsigned* a, unsigned* b, unsigned* result, size_t size)
{
    for (auto i = 0u; i < size; ++i) {
        if (a[i] * b[i] != result[i]) {
            return false;
        }
    }
    return true;
}

void writeVector(std::vector<float> const& values, std::ostream& out)
{
    for (auto const& item: values) {
        out << item << std::endl;
    }
}
