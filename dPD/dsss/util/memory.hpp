#pragma once

#include <sys/resource.h>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>

namespace dsss {

long get_max_mem_kb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

long get_max_mem_bytes() { return 1000 * get_max_mem_kb(); }

void print_vector(auto& vec, std::string sep = " ") {
    bool is_first = true;
    for (auto x: vec) {
        if (!is_first) {
            std::cout << sep;
        }
        is_first = false;
        std::cout << std::fixed << std::setprecision(3) << x;
    }
    std::cout << std::endl;
}


} // namespace dsss