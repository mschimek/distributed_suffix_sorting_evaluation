// This file has been added by Manuel Haag, Matthias Schimek, 2025
#pragma once

#include <cstdint>
#include <iostream>
#include <mxx/comm.hpp>
#include <vector>

#include "distribute.hpp"
#include "kamping/communicator.hpp"
#include "kamping/collectives/barrier.hpp"
#include "reduce.hpp"
#include "shift.hpp"
#include "sorting_wrapper.hpp"
#include "zip.hpp"

namespace dsss {

// adapated from:
// https://github.com/kurpicz/dsss/blob/master/dsss/suffix_sorting/sa_check.hpp
// Roman Dementiev, Juha Kärkkäinen, Jens Mehnert, and Peter Sanders. 2008.
// Better external memory suffix array construction.
template <typename IndexType, typename CharType>
bool check_suffixarray(std::vector<IndexType>& sa, std::vector<CharType>& text,
                       kamping::Communicator<>& comm,
                       const mxx::comm& mxx_comm) {
  using namespace kamping;

  auto& sorting_wrapper = mpi::get_sorting_instance(mxx_comm);

  bool is_correct = true;

  size_t local_size_sa = sa.size();
  size_t local_size_text = text.size();
  size_t global_size_sa = mpi_util::all_reduce_sum(local_size_sa, comm);
  size_t global_size_text = mpi_util::all_reduce_sum(local_size_text, comm);

  if (global_size_text != global_size_sa) {
    if (comm.rank() == 0) {
      std::cerr << "SA and text size don't match: " << global_size_sa
                << " != " << global_size_text << "\n";
    }
    return false;
  }

  struct sa_tuple {
    IndexType rank;
    IndexType sa;
  };

  struct rank_triple {
    IndexType rank1;
    IndexType rank2;
    CharType chr;

    bool operator<(const rank_triple& other) const {
      return std::tie(chr, rank2) < std::tie(other.chr, other.rank2);
    }

    bool operator<=(const rank_triple& other) const {
      return std::tie(chr, rank2) <= std::tie(other.chr, other.rank2);
    }
  };

  // index sa with 1, ..., n
  auto index_function = [](uint64_t idx, IndexType sa_idx) {
    return sa_tuple{1 + IndexType(idx), sa_idx};
  };
  std::vector<sa_tuple> sa_tuples =
      mpi_util::zip_with_index<IndexType, sa_tuple>(sa, index_function, comm);

  sorting_wrapper.sort_with_ams(sa_tuples, [](const sa_tuple& a, const sa_tuple& b) {
    return a.sa < b.sa;
  });

  sa_tuples = mpi_util::distribute_data(sa_tuples, comm);
  text = mpi_util::distribute_data(text, comm);
  comm.barrier();

  size_t local_size = sa_tuples.size();
  size_t offset = mpi_util::ex_prefix_sum(local_size, comm);

  bool is_permutation = true;
  for (size_t i = 0; i < local_size; ++i) {
    is_permutation &= (sa_tuples[i].sa == IndexType(i + offset));
  }

  is_correct = mpi_util::all_reduce_and(is_permutation, comm);
  if (!is_correct) {
    if (comm.rank() == 0) {
      std::cerr << "no permutation \n";
    }
    return false;
  }

  sa_tuple tuple_to_right = mpi_util::shift_left(sa_tuples.front(), comm);

  if (comm.rank() + 1 < comm.size()) {
    sa_tuples.emplace_back(tuple_to_right);
  } else {
    sa_tuples.emplace_back(sa_tuple{0, 0});
  }

  std::vector<rank_triple> rts;
  for (size_t i = 0; i < local_size; ++i) {
    rts.emplace_back(
        rank_triple{sa_tuples[i].rank, sa_tuples[i + 1].rank, text[i]});
  }

  sorting_wrapper.sort_with_ams(rts, [](const rank_triple& a, const rank_triple& b) {
    return a.rank1 < b.rank1;
  });

  local_size = rts.size();
  bool is_sorted = true;
  for (size_t i = 0; i < local_size - 1; ++i) {
    is_sorted &= (rts[i] <= rts[i + 1]);
  }

  auto smaller_triple = mpi_util::shift_right(rts.back(), comm);
  auto larger_triple = mpi_util::shift_left(rts.front(), comm);

  if (comm.rank() > 0) {
    is_sorted &= (smaller_triple < rts.front());
  }
  if (comm.rank() + 1 < comm.size()) {
    is_sorted &= (rts.back() < larger_triple);
  }

  is_correct = mpi_util::all_reduce_and(is_sorted, comm);

  if (!is_correct) {
    if (comm.rank() == 0) {
      std::cerr << "not sorted \n";
    }
  }
  return is_correct;
}

}  // namespace dsss
