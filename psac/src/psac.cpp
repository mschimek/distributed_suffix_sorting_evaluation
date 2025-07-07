/*
 * Copyright 2015 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file    ldss.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Executes and times the suffix array construction using
 *          libdivsufsort.
 */

// This file has been modified by Manuel Haag, Matthias Schimek, 2025


// include MPI
#include <mpi.h>

#include <cstdint>

// C++ includes
#include <fstream>
#include <iostream>
#include <string>

// using TCLAP for command line parsing
#include <tclap/CmdLine.h>

// distributed suffix array construction
#include <alphabet.hpp>
#include <check_suffix_array.hpp>
#include <suffix_array.hpp>

// suffix tree construction
#include <check_suffix_tree.hpp>
#include <suffix_tree.hpp>

// parallel file block decompose
#include <mxx/comm.hpp>
#include <mxx/env.hpp>
#include <mxx/file.hpp>
#include <mxx/utils.hpp>
// Timer
#include <mxx/timer.hpp>

// added
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/printer.hpp>

#include "config.hpp"
#include "kamping/communicator.hpp"
#include "memory.hpp"
#include "mpi_io.hpp"
#include "print.hpp"
#include "sa_check.hpp"
#include "sorting_wrapper.hpp"
#include "uint_types.hpp"

// TODO differentiate between index types (input param or automatic based on
// size!)
// typedef uint64_t index_t;
#ifdef USE_40_BIT_INDEX
typedef dsss::uint40 index_t;
#else
typedef uint64_t index_t;
#endif

struct Parameters {
  size_t text_size = {0};
  std::string input_path;
  std::string output_path;
  double threshold_fast_resolval = 0.1;
  std::string json_output_path;
  bool check = false;
  bool use_ams = false;
  bool use_ips4o = false;
  std::size_t ams_levels = 1;
  std::size_t num_pe = 0;
  int seed = 0;
  std::size_t num_iterations = 1;
  std::string algorithm{"PSAC"};
  std::size_t external_iteration = 0;
#ifdef USE_40_BIT_INDEX
  bool use_40_bit_indextype = true;
#else
  bool use_40_bit_indextype = false;
#endif

  std::vector<std::pair<std::string, std::string>> config() const {
    std::vector<std::pair<std::string, std::string>> config_vector;
    config_vector.emplace_back("num_pe", std::to_string(num_pe));
    config_vector.emplace_back("textsize", std::to_string(text_size));
    config_vector.emplace_back("input_path", input_path);
    config_vector.emplace_back("threshold_fast_resolval", std::to_string(threshold_fast_resolval));
    config_vector.emplace_back("output_path", output_path);
    config_vector.emplace_back("json_output_path", json_output_path);
    config_vector.emplace_back("check", std::to_string(check));
    config_vector.emplace_back("use_ams", std::to_string(use_ams));
    config_vector.emplace_back("use_ips4o", std::to_string(use_ams));
    config_vector.emplace_back("ams_levels", std::to_string(ams_levels));
    config_vector.emplace_back("seed", std::to_string(seed));
    config_vector.emplace_back("use_40_bit_indextype", std::to_string(use_40_bit_indextype));
    config_vector.emplace_back("algorithm", algorithm);
    config_vector.emplace_back("num_iterations", std::to_string(num_iterations));
    config_vector.emplace_back("external_iteration", std::to_string(external_iteration));
    return config_vector;
  }

  friend std::ostream& operator<<(std::ostream& out, Parameters const& params) {
    for (auto const& config_entry : params.config()) {
      out << config_entry.first << "=" << config_entry.second << " ";
    }
    return out;
  }
};

int main(int argc, char* argv[]) {
  const auto mem_program_start = memory::get_max_mem_bytes();
  // set up MPI
  mxx::env e(argc, argv);
  mxx::env::set_exception_on_error();
  mxx::comm comm = mxx::comm();
  mxx::print_node_distribution(comm);
  kamping::measurements::counter().add("mem_program_start", mem_program_start,
                                       {kamping::measurements::GlobalAggregationMode::gather, kamping::measurements::GlobalAggregationMode::max});

  try {
    // define commandline usage
    TCLAP::CmdLine cmd("Parallel distributed suffix array and LCP construction.");
    TCLAP::ValueArg<std::string> fileArg("f", "file", "Input filename.", true, "", "filename");
    TCLAP::ValueArg<std::size_t> randArg("r", "random", "Random input size", true, 0, "size");
    cmd.xorAdd(fileArg, randArg);
    TCLAP::ValueArg<std::string> oArg("o", "outfile", "Output file base name.", false, "", "filename");
    cmd.add(oArg);
    TCLAP::ValueArg<int> seedArg("s", "seed", "Sets the seed for the ranom input generation", false, 0, "int");
    cmd.add(seedArg);
    TCLAP::SwitchArg lcpArg("l", "lcp", "Construct the LCP alongside the SA.", false);
    cmd.add(lcpArg);
    TCLAP::SwitchArg stArg("t", "tree", "Construct the Suffix Tree structute.", false);
    cmd.add(stArg);
    TCLAP::SwitchArg checkArg("c", "check", "Check correctness of SA (and LCP).", false);
    cmd.add(checkArg);

    // new arguments
    TCLAP::ValueArg<std::string> json_output_file("y", "json_output_path", "", false, "", "filename");
    cmd.add(json_output_file);
    TCLAP::ValueArg<uint64_t> totalFilesize("b", "textsize", "maximal file size that should be read", false, 0, "uint64_t");
    cmd.add(totalFilesize);
    TCLAP::ValueArg<std::size_t> iterationArg("z", "iterations", "", false, 1, "int");
    cmd.add(iterationArg);

    TCLAP::ValueArg<int> amsLevels("a", "ams_levels", "number of levels in AMS-sorter", false, 1, "int");
    cmd.add(amsLevels);

    TCLAP::SwitchArg amsFlag("m", "use_ams", "use AMS-sorter instead of mxx::sort", false);
    cmd.add(amsFlag);
    TCLAP::SwitchArg ips4oFlag("n", "use_ips4o", "use ips4o sorter", false);
    cmd.add(ips4oFlag);
    TCLAP::ValueArg<uint64_t> externalIterationArg("i", "iteration", "", false, 0u, "filename");
    cmd.add(externalIterationArg);

    TCLAP::ValueArg<double> thresholdFastResolval("F", "threshold_fast_resolval",
                                                  "Double in [0, 1], switch to fast resolval if remaning_elements < n * "
                                                  "threshold_fast_resolval",
                                                  false, 0.1, "double");
    cmd.add(thresholdFastResolval);

    cmd.parse(argc, argv);

    // configure sorter
    dsss::mpi::SortingWrapper& sorter = dsss::mpi::get_sorting_instance(comm);
    sorter.set_num_levels(amsLevels.getValue());
    sorter.set_use_ams(amsFlag.getValue());
    sorter.finalize_setting();
    dsss::LocalSortingWrapper& local_sorter = dsss::get_local_sorting_instance();
    local_sorter.set_use_ips4o(ips4oFlag.getValue());

    // configure psac
    auto& config = psac::get_config();
    config.threshold_fast_resolval = thresholdFastResolval.getValue();
    const Parameters params = [&]() {
      Parameters parameters;
      parameters.ams_levels = amsLevels.getValue();
      parameters.use_ams = amsFlag.getValue();
      parameters.use_ips4o = ips4oFlag.getValue();
      parameters.input_path = fileArg.getValue();
      parameters.seed = seedArg.getValue();
      parameters.output_path = oArg.getValue();
      parameters.text_size = totalFilesize.getValue();
      parameters.threshold_fast_resolval = thresholdFastResolval.getValue();
      parameters.json_output_path = json_output_file.getValue();
      parameters.check = checkArg.getValue();
      parameters.num_iterations = iterationArg.getValue();
      parameters.external_iteration = externalIterationArg.getValue();
      int comm_size = -1;
      MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
      parameters.num_pe = static_cast<std::size_t>(comm_size);
      return parameters;
    }();

    // report configuration
    if (comm.rank() == 0) {
      std::cerr << "ams_levels=" << amsLevels.getValue() << std::endl;
      std::cerr << "use_ams=" << amsFlag.getValue() << std::endl;
      std::cerr << "check=" << checkArg.getValue() << std::endl;
      std::cerr << "threshold_fast_resolval=" << thresholdFastResolval.getValue() << std::endl;
      uint64_t bits_index = 8 * sizeof(index_t);
      std::cerr << "bits_index_type=" << bits_index << std::endl;
#ifdef USE_40_BIT_INDEX
      std::cerr << "using 40 bit integer \n";
#else
      std::cerr << "using 64 bit integer \n";
#endif
    }

    std::vector<std::string> timer_output;
    std::vector<std::string> counter_output;

    for (std::size_t it = 0; it < params.num_iterations; ++it) {
      // read input file or generate input on master processor
      // block decompose input file
      std::string local_str;
      if (fileArg.getValue() != "") {
        // local_str = mxx::file_block_decompose(fileArg.getValue().c_str(), MPI_COMM_WORLD);
        local_str = mpi::distribute_string(comm, fileArg.getValue(), totalFilesize.getValue());
      } else {
        // TODO proper distributed random!
        local_str = rand_dna(randArg.getValue() / comm.size(), seedArg.getValue() * comm.rank());
      }

      // TODO differentiate between index types

      // run our distributed suffix array construction
      //

      kamping::measurements::counter().add("mem_before_sa_construction", memory::get_max_mem_bytes(),
                                           {kamping::measurements::GlobalAggregationMode::gather, kamping::measurements::GlobalAggregationMode::max});
      kamping::measurements::timer().synchronize_and_start("total_time");
      mxx::timer t;
      double start = t.elapsed();
      if (stArg.getValue()) {
        // construct SA+LCP+ST
        suffix_array<char, size_t, true> sa(comm);
        sa.construct(local_str.begin(), local_str.end());
        double sa_time = t.elapsed() - start;
        // build ST
        std::vector<size_t> local_st_nodes = construct_suffix_tree(sa, local_str.begin(), local_str.end(), comm);
        double st_time = t.elapsed() - sa_time;
        if (comm.rank() == 0) {
          std::cerr << "SA time: " << sa_time << " ms" << std::endl;
          std::cerr << "ST time: " << st_time << " ms" << std::endl;
          std::cerr << "Total  : " << sa_time + st_time << " ms" << std::endl;
        }
        if (checkArg.getValue()) {
          gl_check_suffix_tree(local_str, sa, local_st_nodes, comm);
        }
        if (oArg.getValue() != "") {
          std::cerr << "Error, output of ST not supported" << std::endl;
        }
      } else if (lcpArg.getValue()) {
        // construct SA+LCP
        suffix_array<char, index_t, true> sa(comm);
        sa.construct(local_str.begin(), local_str.end(), true);
        double end = t.elapsed() - start;
        if (comm.rank() == 0) std::cerr << "PSAC time: " << end << " ms" << std::endl;
        if (checkArg.getValue()) {
          gl_check_correct(sa, local_str.begin(), local_str.end(), comm);
        }
        if (oArg.getValue() != "") {
          // output suffix array as binary sa64
          mxx::write_ordered(oArg.getValue() + ".sa64", sa.local_SA, comm);
          mxx::write_ordered(oArg.getValue() + ".lcp64", sa.local_LCP, comm);
        }
      } else {
        // construct SA
        suffix_array<char, index_t, false> sa(comm);
        sa.construct(local_str.begin(), local_str.end(), true);
        double end = t.elapsed() - start;
        kamping::measurements::timer().stop();
        kamping::measurements::counter().add(
            "mem_after_sa_construction", memory::get_max_mem_bytes(),
            {kamping::measurements::GlobalAggregationMode::gather, kamping::measurements::GlobalAggregationMode::max});
        if (comm.rank() == 0) {
          std::cerr << "PSAC time: " << end << " ms" << std::endl;
          // print logged values
          std::cerr << "prefix_doubling_iterations=" << sa.prefix_doubling_iterations << std::endl;
          std::cerr << "initial_k=" << sa.initial_k << std::endl;
          std::cerr << "string_sizes=";
          print_vector(sa.logging_unfinished_elements, ",");
          std::cerr << "k=";
          print_vector(sa.logging_k, ",");
        }
        if (checkArg.getValue()) {
          kamping::Communicator<> kamping_comm;
          std::vector<uint8_t> local_chars(local_str.begin(), local_str.end());
          bool ok = dsss::check_suffixarray(sa.local_SA, local_chars, kamping_comm, comm);
          if (comm.rank() == 0) {
            std::cerr << "SA_ok=" << ok << "\n";
          }
          // d_check_sa(sa, local_str.begin(), local_str.end(), comm);
        }

        kamping::measurements::counter().add(
            "mem_after_sa_check", memory::get_max_mem_bytes(),
            {kamping::measurements::GlobalAggregationMode::gather, kamping::measurements::GlobalAggregationMode::max});
        if (oArg.getValue() != "") {
          // output suffix array as binary sa64
          mxx::write_ordered(oArg.getValue() + ".sa64", sa.local_SA, comm);
        }
      }

      auto config_vector = params.config();
      config_vector.emplace(config_vector.begin(), "iteration", std::to_string(it));
      std::stringstream sstream_counter;
      std::stringstream sstream_timer;
      kamping::measurements::SimpleJsonPrinter<double> printer_timer(sstream_timer, config_vector);
      kamping::measurements::SimpleJsonPrinter<std::int64_t> printer_counter(sstream_counter, config_vector);
      kamping::measurements::timer().aggregate_and_print(printer_timer);
      kamping::measurements::counter().aggregate_and_print(printer_counter);
      kamping::measurements::timer().clear();
      kamping::measurements::counter().clear();
      timer_output.push_back(sstream_timer.str());
      counter_output.push_back(sstream_counter.str());
    }
    auto remove_extension_if_present = [](std::string const& path) {
      auto pos = path.find(".json");
      return path.substr(0, pos);
    };
    auto print_as_jsonlist_to_file = [](std::vector<std::string> objects, std::string filename) {
      std::ofstream outstream(filename);
      outstream << "[" << std::endl;
      for (std::size_t i = 0; i < objects.size(); ++i) {
        if (i > 0) {
          outstream << "," << std::endl;
        }
        outstream << objects[i];
      }
      outstream << std::endl << "]" << std::endl;
    };

    std::string const output_path = remove_extension_if_present(params.json_output_path);
    if (comm.rank() == 0) {
      print_as_jsonlist_to_file(timer_output, params.json_output_path + "_timer.json");
      print_as_jsonlist_to_file(counter_output, params.json_output_path + "_counter.json");
    }

    // catch any TCLAP exception
  } catch (TCLAP::ArgException& e) {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    exit(EXIT_FAILURE);
  }

  // finalize MPI
  // MPI_Finalize();

  return 0;
}
