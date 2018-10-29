#include "common.h"
#include <iostream>
#include <ctime>

int main()
{
    srand(time(nullptr));
    //Task1();
    //Task2();
    Task3();
}

bool GoodMiltiplication(std::vector<unsigned> const& a, std::vector<unsigned> const& b, std::vector<unsigned> const& result)
{
    for (auto i = 0u; i < a.size(); ++i) {
        if (a[i] * b[i] != result[i]) {
            return false;
        }
    }
    return true;
}
