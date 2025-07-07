// This file has been added by Manuel Haag, Matthias Schimek, 2025
#pragma once

#include <iostream>
#include <mxx/comm.hpp>

void print_vector(auto& vec, std::string sep = " ") {
  bool is_first = true;
  for (auto x : vec) {
    if (!is_first) {
      std::cerr << sep;
    }
    is_first = false;
    std::cerr << std::fixed << std::setprecision(3) << x;
  }
  std::cerr << std::endl;
}

void print_on_root(const mxx::comm& comm, std::string msg) {
  comm.barrier();
  if (comm.rank() == 0) {
    std::cerr << msg << "\n";
  }
}
template <typename T>
void print_on_pe(const mxx::comm& comm, T x) {
  std::cerr << "PE " << comm.rank() << x << "\n";
}

template <typename T>
void print_concatenated(const mxx::comm& comm, T x, std::string msg = "") {
  // comm.barrier();
  std::vector<T> v = mxx::allgather(x, comm);
  if (comm.rank() == 0) {
    std::cerr << msg << "\n";
    print_vector(v, " ");
  }
}
