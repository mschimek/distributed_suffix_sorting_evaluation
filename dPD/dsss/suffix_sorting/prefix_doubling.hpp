/*******************************************************************************
 * suffix_sorting/prefix_doubling.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <vector>

#include <tlx/math.hpp>

#include <kamping/communicator.hpp>

#include "mpi/allreduce.hpp"
#include "mpi/environment.hpp"
#include "mpi/scan.hpp"
#include "mpi/shift.hpp"
#include "mpi/sort.hpp"
#include "suffix_sorting/data_structs.hpp"
#include "util/string.hpp"

namespace dsss::suffix_sorting {




static constexpr bool debug = false;

template <typename T>
void free_memory(T&& to_drop) {
    std::remove_reference_t<T>(std::move(to_drop));
}

template <typename IndexType>
inline auto pack_alphabet(dsss::distributed_string& distributed_raw_string,
                          size_t& iteration) {
  using IRR = index_rank_rank<IndexType>;

  dsss::mpi::environment env;

  std::vector<dsss::char_type>& local_str = distributed_raw_string.string;
  std::vector<size_t> char_histogram(256, 0);
  for (const auto c : local_str) {
    ++char_histogram[c];
  }
  char_histogram = dsss::mpi::allreduce_sum(char_histogram, env);
  std::vector<size_t> char_map(256, 0);
  size_t new_alphabet_size = 1;
  for (size_t i = 0; i < 256; ++i) {
    if (char_histogram[i] != 0) {
      char_map[i] = new_alphabet_size++;
    }
  }
  size_t bits_per_char = tlx::integer_log2_ceil(new_alphabet_size);
  size_t k_fitting = (8 * sizeof(IndexType)) / bits_per_char;
  iteration = tlx::integer_log2_floor(k_fitting) + 1;

  if constexpr (debug) {
    std::vector<size_t> global_histogram =
        dsss::mpi::allreduce_sum(char_histogram, env);

    if (env.rank() == 0) {
      std::cout << "Text-Histogram:" << std::endl;
      for (size_t i = 0; i < 256; ++i) {
        if (size_t occ = char_histogram[i]; occ > 0) {
          std::cout << i << ": " << occ << " ("
                    << (100 / double(local_str.size())) * occ << "%)"
                    << std::endl;
        }
      }
      std::cout << "New alphabet size = " << new_alphabet_size << std::endl
                << "Requiring " << bits_per_char << " bits per character."
                << std::endl
                << "Packing " << k_fitting << " characters in one rank."
                << std::endl
                << "Starting at iteration " << iteration << "." << std::endl;
    }
  }

  size_t local_size = local_str.size();
  std::vector<dsss::char_type> right_chars =
      dsss::mpi::shift_left(local_str.data(), 2 * k_fitting, env);
  if (env.rank() + 1 < env.size()) {
    std::move(right_chars.begin(), right_chars.end(),
              std::back_inserter(local_str));
  } else {
    for (size_t i = 0; i < 2 * k_fitting; ++i) {
      local_str.emplace_back(0);
    }
  }

  size_t index = dsss::mpi::ex_prefix_sum(local_size, env);
  std::vector<IRR> result;
  result.reserve(local_size);
  for (size_t i = 0; i < local_size; ++i) {
    IndexType rank1 = IndexType(char_map[local_str[i]]);
    IndexType rank2 = IndexType(char_map[local_str[i + k_fitting]]);
    for (size_t j = 1; j < k_fitting; ++j) {
      rank1 = (rank1 << bits_per_char) | char_map[local_str[i + j]];
      rank2 = (rank2 << bits_per_char) | char_map[local_str[i + k_fitting + j]];
    }
    result.emplace_back(index++, rank1, rank2);
  }
  return result;
}

template <typename IndexType>
std::vector<IndexType> prefix_doubling(
    dsss::distributed_string&& distributed_raw_string) {
  using IR = index_rank<IndexType>;
  using IRR = index_rank_rank<IndexType>;

  dsss::mpi::environment env;
  kamping::Communicator comm;

  size_t offset = 0;
  size_t local_size = 0;
  size_t iteration = 0;
  std::vector<IRR> irrs =
      pack_alphabet<IndexType>(distributed_raw_string, iteration);
  std::vector<IR> irs;
  while (true) {
    // Sort based on two ranks
    dsss::mpi::sort(
        irrs,
        [](const IRR& a, const IRR& b) {
          return std::tie(a.rank1, a.rank2) < std::tie(b.rank1, b.rank2);
        },
        env);

    // Compute new ranks
    local_size = irrs.size();
    offset = dsss::mpi::ex_prefix_sum(local_size);

    irs.clear();
    irs.reserve(local_size);

    size_t cur_rank = offset;
    irs.emplace_back(irrs[0].index, cur_rank);
    for (size_t i = 1; i < local_size; ++i) {
      if (irrs[i - 1] != irrs[i]) {
        cur_rank = offset + i;
      }
      irs.emplace_back(irrs[i].index, cur_rank);
    }

    bool all_distinct = true;
    for (size_t i = 1; i < irs.size(); ++i) {
      all_distinct &= (irs[i].rank != irs[i - 1].rank);
      if (!all_distinct) {
        break;
      }
    }
    const bool finished = dsss::mpi::allreduce_and(all_distinct, env);
    if (finished) {
      break;
    }

    dsss::mpi::sort(
        irs,
        [iteration](const IR& a, const IR& b) {
          IndexType mod_mask = (size_t(1) << iteration) - 1;
          IndexType div_mask = ~mod_mask;

          if ((a.index & mod_mask) == (b.index & mod_mask)) {
            return (a.index & div_mask) < (b.index & div_mask);
          } else {
            return (a.index & mod_mask) < (b.index & mod_mask);
          }
        },
        env);

    local_size = irs.size();
    offset = dsss::mpi::ex_prefix_sum(local_size);

    IR rightmost_ir = dsss::mpi::shift_left(irs.front(), env);
    if (env.rank() + 1 < env.size()) {
      irs.emplace_back(rightmost_ir);
    } else {
      irs.emplace_back(0, 0);
    }

    irrs.clear();
    irrs.reserve(local_size);
    const size_t index_distance = size_t(1) << iteration;
    for (size_t i = 0; i < local_size; ++i) {
      IndexType second_rank = {0};
      if (DSSS_LIKELY(irs[i].index + index_distance == irs[i + 1].index)) {
        second_rank = irs[i + 1].rank;
      }
      irrs.emplace_back(irs[i].index, irs[i].rank, second_rank);
    }
    if constexpr (debug) {
      if (env.rank() == 0) {
        std::cout << "Finished iteration " << iteration << std::endl;
      }
    }
    ++iteration;
  }

  std::vector<IndexType> result;
  result.reserve(irs.size());
  std::transform(irs.begin(), irs.end(), std::back_inserter(result),
                 [](const IR& ir) { return ir.index; });
  return result;
}

template <typename IndexType>
struct prev_occur_count_t {
  IndexType last_r1;
  IndexType last_r2;
  size_t count_r1;  // TODO CHANGE HERE!!!
  size_t count_r2;  // TODO CHANGE HERE!!!
  bool is_empty() const { return (count_r1 == 0) && (count_r2 == 0); }
  bool same_ranks(IndexType r1, IndexType r2) const {
    if (is_empty()) {
      return false;
    }
    return r1 == last_r1 && r2 == last_r2;
  }
  bool same_first_rank(IndexType r1) const {
    if (is_empty()) {
      return false;
    }
    return r1 == last_r1;
  }

  std::string to_string() const {
    std::stringstream str;
    str << "[" << last_r1 << ", " << last_r2 << "] = (c1:" << count_r1
        << ", c2:" << count_r2;
    return str.str();
  }
};

template <typename IndexType>
struct rank_info_t {
  bool is_empty;
  IndexType rank;
  bool is_equal(IndexType r) const {
    if (is_empty) {
      return false;
    } else {
      return r == rank;
    }
  }
};

template <typename IndexType>
rank_info_t<IndexType> compute_right_nonempty_rank(
    rank_info_t<IndexType> rank_info,
    dsss::mpi::environment env = dsss::mpi::environment()) {
  auto const rank_infos = dsss::mpi::allgather(rank_info, env);
  for (std::int32_t i = env.rank() + 1; i < env.size(); ++i) {
    if (!rank_infos[i].is_empty) {
      return rank_infos[i];
    }
  }
  return rank_info_t<IndexType>{true, IndexType(0)};
}

template <typename IndexType, bool return_isa = false>
std::vector<IndexType> doubling_discarding(
    std::vector<index_rank_state<IndexType>>& irss, size_t iteration,
    dsss::mpi::environment env = dsss::mpi::environment()) {
  using namespace kamping;
  Communicator comm;
  using IR = index_rank<IndexType>;
  using IRS = index_rank_state<IndexType>;
  using IRR = index_rank_rank<IndexType>;

  std::vector<IRR> irrs;
  std::vector<IR> fully_discarded;
  while (iteration) {
    auto start_time = MPI_Wtime();
    if constexpr (debug) {
      auto const sum = comm.allreduce_single(
          send_buf(irss.size() + fully_discarded.size()), op(std::plus<>{}));
      if (comm.is_root()) {
        std::cout << "\t\titer: " << iteration << " " << sum << "\n\n\n"
                  << std::endl;
      }
    }

    if constexpr (debug) {
      env.barrier();
      if (env.rank() == 0) {
        std::cout << "Start Sorting Mod/Div in iteration " << iteration
                  << std::endl;
      }
      env.barrier();
    }

    auto comp_irss = [iteration](const IRS& a, const IRS& b) {
      IndexType mod_mask = (size_t(1) << iteration) - 1;
      IndexType div_mask = ~mod_mask;

      if ((a.index & mod_mask) == (b.index & mod_mask)) {
        return (a.index & div_mask) < (b.index & div_mask);
      } else {
        return (a.index & mod_mask) < (b.index & mod_mask);
      }
    };
    dsss::mpi::sort_ns(irss, comp_irss, env);

    if constexpr (debug) {
      env.barrier();
      if (env.rank() == 0) {
        std::cout << "Sorting Mod/Div in iteration " << iteration << std::endl;
      }
      env.barrier();
    }

    size_t local_size = irss.size();
    struct optional_irs {
      bool is_empty;
      IRS irs;
    } DSSS_ATTRIBUTE_PACKED;

    optional_irs o_irs = {local_size == 0, (local_size == 0)
                                               ? IRS{0, 0, rank_state::UNIQUE}
                                               : irss.front()};
    std::vector<optional_irs> rec_data = dsss::mpi::allgather(o_irs, env);
    IRS rightmost_irs = {0, 0, rank_state::UNIQUE};
    {
      int32_t rank = env.rank() + 1;
      while (rank < env.size() && rec_data[rank].is_empty) {
        ++rank;
      }
      if (rank < env.size()) {
        rightmost_irs = rec_data[rank].irs;
      }
    }

    if (env.rank() + 1 < env.size()) {
      irss.emplace_back(rightmost_irs);
    } else {
      irss.emplace_back(0, 0, rank_state::UNIQUE);
    }

    if constexpr (debug) {
      if (env.rank() == 0) {
        std::cout << "print rightmost" << std::endl;
        for (const auto& elem : rec_data) {
          std::cout << "is empty: " << elem.is_empty << " irs: " << elem.irs
                    << std::endl;
        }
      }
    }
    rec_data.clear();
    rec_data.shrink_to_fit();
    free_memory(std::move(rec_data));

    std::vector<IRS> unique;
    size_t prev_non_unique = 2;
    const size_t index_distance = size_t(1) << iteration;
    for (size_t i = 0; i < local_size; ++i) {
      if (irss[i].state == rank_state::UNIQUE) {
        if (prev_non_unique > 1) {
          unique.emplace_back(irss[i]);
        } else {
          fully_discarded.emplace_back(irss[i].index, irss[i].rank);
        }
        prev_non_unique = 0;
      } else {
        IndexType second_rank = {0};
        if (DSSS_LIKELY(irss[i].index + index_distance == irss[i + 1].index)) {
          second_rank = irss[i + 1].rank;
        }
        irrs.emplace_back(irss[i].index, irss[i].rank, second_rank);
        ++prev_non_unique;
      }
    }

    irss.clear();
    irss.shrink_to_fit();
    free_memory(std::move(irss));

    const auto irrs_comp = [](const IRR& a, const IRR& b) {
      return std::tie(a.rank1, a.rank2, a.index) <
             std::tie(b.rank1, b.rank2, b.index);
    };
    const auto irrs_comp_ = [](const IRR& a, const IRR& b) {
      return std::tie(a.rank1, a.rank2) < std::tie(b.rank1, b.rank2);
    };
    dsss::mpi::sort_ns(irrs, irrs_comp, env);
    local_size = irrs.size();
    irss.reserve(local_size);

    struct prev_occur {
      IndexType last_first_rank;
      size_t count;
    };

    prev_occur_count_t<IndexType> pocc;
    if (irrs.empty()) {
      pocc.last_r1 = IndexType(0);
      pocc.last_r2 = IndexType(0);
      pocc.count_r1 = 0;
      pocc.count_r2 = 0;
    } else {
      pocc.last_r1 = irrs.back().rank1;
      pocc.last_r2 = irrs.back().rank2;
      pocc.count_r1 = 0;
      pocc.count_r2 = 0;
      for (std::int64_t i = local_size - 1;
           i >= 0 && irrs[i].rank1 == pocc.last_r1; --i) {
        ++pocc.count_r1;
        if (irrs[i].rank2 == pocc.last_r2) {
          ++pocc.count_r2;
        }
      }
    }
    std::vector<prev_occur_count_t<IndexType>> poccs =
        dsss::mpi::allgather(pocc, env);
    if constexpr (debug) {
      if (env.rank() == 0) {
        for (const auto& pocc : poccs) {
          std::cout << pocc.to_string() << std::endl;
        }
      }
    }
    size_t offset_r1 = 0;
    size_t offset_r2 = 0;
    std::size_t cur_rank = 0;
    bool is_left_rank_same = false;
    if (local_size > 0) {
      const auto first_rank1 = irrs[0].rank1;
      const auto first_rank2 = irrs[0].rank2;
      for (int32_t i = env.rank() - 1;
           i >= 0 && (poccs[i].is_empty() || first_rank1 == poccs[i].last_r1);
           --i) {
        if (poccs[i].is_empty()) {
          continue;
        }
        if (first_rank1 == poccs[i].last_r1) {
          offset_r1 += poccs[i].count_r1;
          if (first_rank2 == poccs[i].last_r2) {
            offset_r2 += poccs[i].count_r2;
          }
        }
      }
      prev_occur_count_t<IndexType> const left_eligible = [&]() {
        for (int32_t i = env.rank() - 1; i >= 0; --i) {
          if (!poccs[i].is_empty()) {
            return poccs[i];
          }
        }
        return prev_occur_count_t<IndexType>{0, 0, 0, 0};
      }();

      cur_rank = first_rank1;
      is_left_rank_same = left_eligible.same_ranks(first_rank1, first_rank2);
      if (is_left_rank_same) {
        cur_rank += (offset_r1 - offset_r2);
      } else if (left_eligible.same_first_rank(first_rank1)) {
        cur_rank += offset_r1;
      }
      offset_r1++;
      irss.emplace_back(irrs[0].index, cur_rank,
                        rank_state::NONE);  // check rank_state later
    }
    KASSERT(offset_r2 <= offset_r1);
    for (size_t i = 1; i < local_size; ++i) {
      if (irrs[i].rank1 == irrs[i - 1].rank1) {
        if (irrs[i].rank2 != irrs[i - 1].rank2) {
          cur_rank = irrs[i].rank1 + offset_r1;
        }
        ++offset_r1;
      } else {
        offset_r1 = 1;
        cur_rank = irrs[i].rank1;
      }
      irss.emplace_back(irrs[i].index, cur_rank, rank_state::NONE);
    }
    poccs.clear();
    poccs.shrink_to_fit();
    free_memory(std::move(poccs));

    rank_info_t<IndexType> rank_info;
    if (local_size > 0) {
      rank_info.is_empty = false;
      rank_info.rank = irss[0].rank;
    } else {
      rank_info.is_empty = true;
      rank_info.rank = IndexType(0);
    }
    const rank_info_t<IndexType> right_new_rank =
        compute_right_nonempty_rank(rank_info);

    irrs.clear();
    irrs.shrink_to_fit();
    free_memory(std::move(irrs));

    bool all_unique_local = true;
    if (local_size == 1) {
      if (!is_left_rank_same && !right_new_rank.is_equal(irss[0].rank)) {
        irss[0].state = rank_state::UNIQUE;
      } else {
        all_unique_local = false;
      }
    }
    if (local_size > 1) {
      if (irss[0].rank == irss[1].rank || is_left_rank_same) {
        irss[0].state = rank_state::NONE;
        all_unique_local = false;
      } else {
        irss[0].state = rank_state::UNIQUE;
      }
      for (size_t i = 1; i + 1 < local_size; ++i) {
        if (irss[i].rank == irss[i - 1].rank ||
            irss[i].rank == irss[i + 1].rank) {
          irss[i].state = rank_state::NONE;
          all_unique_local = false;
        } else {
          irss[i].state = rank_state::UNIQUE;
        }
      }
      if (irss[local_size - 1].rank == irss[local_size - 2].rank ||
          right_new_rank.is_equal(irss[local_size - 1].rank)) {
        irss[local_size - 1].state = rank_state::NONE;
        all_unique_local = false;
      } else {
        irss[local_size - 1].state = rank_state::UNIQUE;
      }
    }

    if constexpr (debug) {
      env.barrier();
      if (env.rank() == 0) {
        std::cout << "Generating new ranks in iteration " << iteration
                  << std::endl;
      }
      env.barrier();
    }

    std::move(unique.begin(), unique.end(), std::back_inserter(irss));
    unique.clear();
    unique.shrink_to_fit();
    free_memory(std::move(unique));

    bool all_unique_global = dsss::mpi::allreduce_and(all_unique_local, env);
    if (all_unique_global) {
      break;
    }

    ++iteration;
    auto end_time = MPI_Wtime();

    if constexpr (debug) {
      if (env.rank() == 0) {
        std::cout << "Finished iteration " << iteration << " (with discarding)"
                  << " in " << end_time - start_time << " seconds" << std::endl;
      }
    }
    env.barrier();
  }
  if constexpr (debug) {
    auto const sum = comm.allreduce_single(
        send_buf(irss.size() + fully_discarded.size()), op(std::plus<>{}));
    if (comm.is_root()) {
      std::cout << "\t\tend iter: " << iteration << " " << sum << "\n\n\n"
                << std::endl;
    }
  }
  fully_discarded.reserve(fully_discarded.size() + irss.size());
  std::transform(irss.begin(), irss.end(), std::back_inserter(fully_discarded),
                 [](const IRS& irs) { return IR{irs.index, irs.rank}; });

  if constexpr (return_isa) {
    dsss::mpi::sort_ns(
        fully_discarded,
        [](const IR& a, const IR& b) { return a.index < b.index; }, env);
  } else {
    dsss::mpi::sort_ns(
        fully_discarded,
        [](const IR& a, const IR& b) { return a.rank < b.rank; }, env);
  }
  irss.clear();
  irss.shrink_to_fit();
  free_memory(std::move(irss));

  std::vector<IndexType> result;
  result.reserve(fully_discarded.size());

  std::transform(fully_discarded.begin(), fully_discarded.end(),
                 std::back_inserter(result), [](const IR& ir) {
                   if constexpr (return_isa) {
                     return ir.rank;
                   } else {
                     return ir.index;
                   }
                 });

  return result;
}

template <typename IndexType, bool return_isa = false>
std::vector<IndexType> prefix_doubling_discarding(
    dsss::distributed_string&& distributed_raw_string) {
  using IR = index_rank<IndexType>;
  using IRS = index_rank_state<IndexType>;
  using IRR = index_rank_rank<IndexType>;

  dsss::mpi::environment env;

  size_t offset = 0;
  size_t local_size = 0;
  size_t iteration = 0;
  std::vector<IRR> irrs =
      pack_alphabet<IndexType>(distributed_raw_string, iteration);

  std::vector<IR> irs;
  // Start one round of prefix doubling
  auto irrs_comp1 = [](const IRR& a, const IRR& b) {
    return std::tie(a.rank1, a.rank2) < std::tie(b.rank1, b.rank2);
  };
  dsss::mpi::sort_ns(irrs, irrs_comp1, env);

  // Compute new ranks
  local_size = irrs.size();
  offset = dsss::mpi::ex_prefix_sum(local_size);

  {
    // new
    prev_occur_count_t<IndexType> pocc;
    if (irrs.empty()) {
      pocc.last_r1 = IndexType(0);
      pocc.last_r2 = IndexType(0);
      pocc.count_r1 = 0;
      pocc.count_r2 = 0;
    } else {
      pocc.last_r1 = irrs.back().rank1;
      pocc.last_r2 = irrs.back().rank2;
      pocc.count_r1 = 0;
      pocc.count_r2 = 0;
      for (std::int64_t i = local_size - 1;
           i >= 0 && irrs[i].rank1 == pocc.last_r1; --i) {
        ++pocc.count_r1;
        if (irrs[i].rank2 == pocc.last_r2) {
          ++pocc.count_r2;
        }
      }
    }
    std::vector<prev_occur_count_t<IndexType>> poccs =
        dsss::mpi::allgather(pocc, env);
    if constexpr (debug) {
      if (env.rank() == 0) {
        for (const auto& pocc : poccs) {
          std::cout << pocc.to_string() << std::endl;
        }
      }
    }
    size_t offset_r1 = 0;
    size_t offset_r2 = 0;
    std::size_t cur_rank = 0;
    bool is_left_rank_same = false;

    irs.clear();
    irs.reserve(local_size);
    if (local_size > 0) {
      const auto first_rank1 = irrs[0].rank1;
      const auto first_rank2 = irrs[0].rank2;
      for (int32_t i = env.rank() - 1;
           i >= 0 && (poccs[i].is_empty() || first_rank1 == poccs[i].last_r1);
           --i) {
        if (poccs[i].is_empty()) {
          continue;
        }
        if (first_rank1 == poccs[i].last_r1) {
          offset_r1 += poccs[i].count_r1;
          if (first_rank2 == poccs[i].last_r2) {
            offset_r2 += poccs[i].count_r2;
          }
        }
      }
      prev_occur_count_t<IndexType> const left_eligible = [&]() {
        for (int32_t i = env.rank() - 1; i >= 0; --i) {
          if (!poccs[i].is_empty()) {
            return poccs[i];
          }
        }
        return prev_occur_count_t<IndexType>{0, 0, 0, 0};
      }();

      cur_rank = offset;
      is_left_rank_same = left_eligible.same_ranks(first_rank1, first_rank2);
      if (is_left_rank_same) {
        cur_rank -= offset_r2;
      }
      irs.emplace_back(irrs[0].index, cur_rank);  // check rank_state later
    }

    if (local_size > 0) {
      for (size_t i = 1; i < local_size; ++i) {
        if (irrs[i - 1] != irrs[i]) {
          cur_rank = offset + i;
        }
        irs.emplace_back(irrs[i].index, cur_rank);
      }
    }
  }
  offset = dsss::mpi::ex_prefix_sum(local_size);
  bool all_distinct = true;
  for (size_t i = 1; i < irs.size(); ++i) {
    all_distinct &= (irs[i].rank != irs[i - 1].rank);
    if (!all_distinct) {
      break;
    }
  }
  const bool finished = dsss::mpi::allreduce_and(all_distinct, env);
  if (finished) {
    // check neighboring ranks as they might overlap
    rank_info_t<IndexType> first_rank{true, 0};
    if (!irs.empty()) {
      first_rank.is_empty = false;
      first_rank.rank = irs.front().rank;
    }
    const auto right_rank_info = compute_right_nonempty_rank(first_rank, env);

    bool really_finished = true;
    if (!irs.empty()) {
      really_finished = !right_rank_info.is_equal(irs.back().rank);
    }
    if (dsss::mpi::allreduce_and(really_finished, env)) {
      std::vector<IndexType> result;
      result.reserve(irs.size());
      std::transform(irs.begin(), irs.end(), std::back_inserter(result),
                     [](const IR& ir) { return ir.index; });
      return result;
    }
  }

  auto irs_comp = [iteration](const IR& a, const IR& b) {
    IndexType mod_mask = (size_t(1) << iteration) - 1;
    IndexType div_mask = ~mod_mask;

    if ((a.index & mod_mask) == (b.index & mod_mask)) {
      return (a.index & div_mask) < (b.index & div_mask);
    } else {
      return (a.index & mod_mask) < (b.index & mod_mask);
    }
  };
  dsss::mpi::sort_ns(irs, irs_comp, env);

  local_size = irs.size();
  offset = dsss::mpi::ex_prefix_sum(local_size, env);

  struct optional_ir {
    bool is_empty;
    IR ir;
  } DSSS_ATTRIBUTE_PACKED;

  optional_ir o_ir = {local_size == 0,
                      (local_size == 0) ? IR{0, 0} : irs.front()};
  std::vector<optional_ir> rec_data = dsss::mpi::allgather(o_ir, env);
  IR rightmost_irs = {0, 0};
  {
    int32_t rank = env.rank() + 1;
    while (rank < env.size() && rec_data[rank].is_empty) {
      ++rank;
    }
    if (rank < env.size()) {
      rightmost_irs = rec_data[rank].ir;
    }
  }

  if (env.rank() + 1 < env.size()) {
    irs.emplace_back(rightmost_irs);
  } else {
    irs.emplace_back(0, 0);
  }

  irrs.clear();
  irrs.reserve(local_size);
  const size_t index_distance = size_t(1) << iteration;
  for (size_t i = 0; i < local_size; ++i) {
    IndexType second_rank = {0};
    if (DSSS_LIKELY(irs[i].index + index_distance == irs[i + 1].index)) {
      second_rank = irs[i + 1].rank;
    }
    irrs.emplace_back(irs[i].index, irs[i].rank, second_rank);
  }
  rec_data.clear();
  rec_data.shrink_to_fit();
  free_memory(std::move(rec_data));

  irs.clear();
  irs.shrink_to_fit();
  free_memory(std::move(irs));

  auto irrs_comp = [](const IRR& a, const IRR& b) {
    return std::tie(a.rank1, a.rank2) < std::tie(b.rank1, b.rank2);
  };
  dsss::mpi::sort_ns(irrs, irrs_comp, env);
  local_size = irrs.size();
  std::vector<IRS> irss;
  irss.reserve(local_size);
  {
    // new
    prev_occur_count_t<IndexType> pocc;
    if (irrs.empty()) {
      pocc.last_r1 = IndexType(0);
      pocc.last_r2 = IndexType(0);
      pocc.count_r1 = 0;
      pocc.count_r2 = 0;
    } else {
      pocc.last_r1 = irrs.back().rank1;
      pocc.last_r2 = irrs.back().rank2;
      pocc.count_r1 = 0;
      pocc.count_r2 = 0;
      for (std::int64_t i = local_size - 1;
           i >= 0 && irrs[i].rank1 == pocc.last_r1; --i) {
        ++pocc.count_r1;
        if (irrs[i].rank2 == pocc.last_r2) {
          ++pocc.count_r2;
        }
      }
    }
    std::vector<prev_occur_count_t<IndexType>> poccs =
        dsss::mpi::allgather(pocc, env);
    if constexpr (debug) {
      if (env.rank() == 0) {
        for (const auto& pocc : poccs) {
          std::cout << pocc.to_string() << std::endl;
        }
      }
    }
    size_t offset_r1 = 0;
    size_t offset_r2 = 0;
    std::size_t cur_rank = 0;
    bool is_left_rank_same = false;
    if (local_size > 0) {
      const auto first_rank1 = irrs[0].rank1;
      const auto first_rank2 = irrs[0].rank2;
      for (int32_t i = env.rank() - 1;
           i >= 0 && (poccs[i].is_empty() || first_rank1 == poccs[i].last_r1);
           --i) {
        if (poccs[i].is_empty()) {
          continue;
        }
        if (first_rank1 == poccs[i].last_r1) {
          offset_r1 += poccs[i].count_r1;
          if (first_rank2 == poccs[i].last_r2) {
            offset_r2 += poccs[i].count_r2;
          }
        }
      }
      prev_occur_count_t<IndexType> const left_eligible = [&]() {
        for (int32_t i = env.rank() - 1; i >= 0; --i) {
          if (!poccs[i].is_empty()) {
            return poccs[i];
          }
        }
        return prev_occur_count_t<IndexType>{0, 0, 0, 0};
      }();

      cur_rank = first_rank1;
      is_left_rank_same = left_eligible.same_ranks(first_rank1, first_rank2);
      if (is_left_rank_same) {
        cur_rank += (offset_r1 - offset_r2);
      } else if (left_eligible.same_first_rank(first_rank1)) {
        cur_rank += offset_r1;
      }
      offset_r1++;
      irss.emplace_back(irrs[0].index, cur_rank,
                        rank_state::NONE);  // check rank_state later
    }
    KASSERT(offset_r2 <= offset_r1);
    for (size_t i = 1; i < local_size; ++i) {
      if (irrs[i].rank1 == irrs[i - 1].rank1) {
        if (irrs[i].rank2 != irrs[i - 1].rank2) {
          cur_rank = irrs[i].rank1 + offset_r1;
        }
        ++offset_r1;
      } else {
        offset_r1 = 1;
        cur_rank = irrs[i].rank1;
      }
      irss.emplace_back(irrs[i].index, cur_rank, rank_state::NONE);
    }
  }
  if (!irrs.empty()) {
    if (irss.size() == 1) {
      irss[0].state = rank_state::NONE;
    } else if (irss.size() > 1) {
      for (size_t i = 1; i + 1 < local_size; ++i) {
        if (irss[i].rank != irss[i - 1].rank &&
            irss[i].rank != irss[i + 1].rank) {
          irss[i].state = rank_state::UNIQUE;
        }
      }
      irss[local_size - 1].state = rank_state::NONE;
    }
  }

  irrs.clear();
  irrs.shrink_to_fit();
  free_memory(std::move(irrs));

  if constexpr (debug) {
    env.barrier();
    if (env.rank() == 0) {
      std::cout << "Finished iteration " << iteration << " (w/o discarding)"
                << std::endl;
    }
    env.barrier();
  }
  ++iteration;
  return doubling_discarding<IndexType, return_isa>(irss, iteration, env);
}

}  // namespace dsss::suffix_sorting

/******************************************************************************/
