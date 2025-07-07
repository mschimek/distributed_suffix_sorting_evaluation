// This file has been added by Manuel Haag, Matthias Schimek, 2025
#pragma once

#include <sys/resource.h>

#include <mxx/collective.hpp>
#include <string>
#include <vector>

#include "print.hpp"

namespace memory {

long get_max_mem_kb() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss;
}

long get_max_mem_bytes() { return 1000 * get_max_mem_kb(); }

std::vector<long> get_mem_bytes_all(const mxx::comm& comm) {
  long mem_pe = get_max_mem_bytes();
  return mxx::allgather(mem_pe, comm);
}

void report_memory(const mxx::comm& comm, std::string name = "") {
  name = "DEBUG_PE_MEMORY_" + name;
  std::replace(name.begin(), name.end(), ' ', '_');

  auto mem = get_mem_bytes_all(comm);
  if (comm.rank() == 0) {
    std::cerr << name << "=";
    print_vector(mem, ",");
  }
}

}  // namespace memory
