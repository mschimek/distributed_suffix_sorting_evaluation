/*******************************************************************************
 * benchmark/dsaca.cpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#include <tlx/cmdline_parser.hpp>

#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/printer.hpp>

#include "atomic_sorting/sorting_wrapper.hpp"
#include "mpi/allreduce.hpp"
#include "mpi/distribute_input.hpp"
#include "mpi/environment.hpp"

#include "suffix_sorting/inducing.hpp"
#include "suffix_sorting/prefix_doubling.hpp"
#include "suffix_sorting/sa_check.hpp"

#include "util/memory.hpp"
#include "util/random_string_generator.hpp"
#include "util/string.hpp"
#include "util/uint_types.hpp"

#include <fstream>

struct Parameters {
  std::size_t text_size = {0};
  std::string input_path;
  std::string output_path;
  std::string json_output_path;
  bool check = false;
  bool use_ams = false;
  bool use_disarding;
  std::size_t ams_levels = 1;
  std::size_t num_pe = 0;
  std::string algorithm{"dPD"};
  std::size_t external_iteration = 0;

  std::vector<std::pair<std::string, std::string>> config() const {
    std::vector<std::pair<std::string, std::string>> config_vector;
    config_vector.emplace_back("num_pe", std::to_string(num_pe));
    config_vector.emplace_back("textsize", std::to_string(text_size));
    config_vector.emplace_back("input_path", input_path);
    config_vector.emplace_back("output_path", output_path);
    config_vector.emplace_back("json_output_path", json_output_path);
    config_vector.emplace_back("check", std::to_string(check));
    config_vector.emplace_back("use_ams", std::to_string(use_ams));
    config_vector.emplace_back("use_ips4o", std::to_string(use_ams));
    config_vector.emplace_back("ams_levels", std::to_string(ams_levels));
    config_vector.emplace_back("algorithm", algorithm);
    config_vector.emplace_back("external_iteration",
                               std::to_string(external_iteration));
    return config_vector;
  }
};

size_t string_size = {0};
std::string input_path = "";
std::string output_path = "";
std::string json_output_path = "";
bool check = false;
bool doubling_discarding = false;
bool use_ams = false;
size_t ams_levels = 1;
size_t external_iteration = 0;

int32_t main(int32_t argc, char const *argv[]) {
  const auto mem_program_start = dsss::get_max_mem_bytes();
  dsss::mpi::environment env;
  tlx::CmdlineParser cp;

  kamping::measurements::counter().add(
      "mem_program_start", mem_program_start,
      {kamping::measurements::GlobalAggregationMode::gather,
       kamping::measurements::GlobalAggregationMode::max});

  using index_type = dsss::uint40;
  // using index_type = size_t;

  cp.set_description("Distributed Suffix Array Construction");
  cp.set_author("Florian Kurpicz <florian.kurpicz@tu-dortmund.de>");

  cp.add_string("input", input_path,
                "Path to input file. The special input 'random' generates"
                " a random text of the size given by parameter '-s'.");

  cp.add_size_t('s', "size", string_size,
                "Size (in bytes unless stated "
                "otherwise) of the string that use to test our suffix array "
                "construction algorithms.");

  cp.add_string('j', "json_output_path", json_output_path,
                "Path to which aggreated result files are written.");

  cp.add_flag('c', "check", check,
              "Check if the SA has been constructed "
              "correctly. This does not work with random text (no way to "
              " reproduce).");

  cp.add_string('o', "output", "<F>", output_path,
                "Filename for the output "
                "(SA). Note that the output is five times larger than the input"
                " file.");

  cp.add_flag('d', "discarding", doubling_discarding,
              "Compute the suffix array"
              " using prefix doubling with discarding (instead of inducing).");

  cp.add_flag('a', "use_ams", use_ams, "Use the AMS sorting algorithm.");
  cp.add_size_t('l', "ams_levels", ams_levels,
                "Number of levels to use in AMS.");
  cp.add_size_t('i', "iteration", external_iteration, "Iteration param.");

  if (!cp.process(argc, argv)) {
    return -1;
  }

  // print configuration
  if (env.rank() == 0) {
    std::cout << "use_discarding=" << doubling_discarding << std::endl;
    std::cout << "use_ams=" << use_ams << std::endl;
    std::cout << "ams_levels=" << ams_levels << std::endl;
  }

  // configure sorter
  auto &sorter = dsss::mpi::get_sorter_instance();
  sorter.set_sorter(dsss::mpi::AtomicSorters::SampleSort);
  if (use_ams) {
    sorter.set_sorter(dsss::mpi::AtomicSorters::Ams);
    sorter.set_num_levels(ams_levels);
  }
  sorter.finalize_setting();
  Parameters const params = [&]() {
    Parameters params;
    params.use_ams = use_ams;
    params.ams_levels = ams_levels;
    params.use_disarding = doubling_discarding;
    params.use_disarding = doubling_discarding;
    params.input_path = input_path;
    params.output_path = output_path;
    params.json_output_path = json_output_path;
    params.text_size = string_size;
    params.num_pe = env.size();
    params.external_iteration = external_iteration;
    params.check = check;
    return params;
  }();

  dsss::distributed_string distributed_strings;

  if (!input_path.compare("random")) {
    string_size /= env.size();
    dsss::random_indexed_string_set<index_type> rss(string_size, 255);
    distributed_strings = {env.rank() * string_size,
                           std::move(rss.data_container())};
  } else {
    if (string_size > 0) {
      distributed_strings =
          dsss::mpi::distribute_string(input_path, string_size);
    } else {
      distributed_strings = dsss::mpi::distribute_string(input_path);
    }
  }

  kamping::measurements::counter().add(
      "mem_before_sa_construction", dsss::get_max_mem_bytes(),
      {kamping::measurements::GlobalAggregationMode::gather,
       kamping::measurements::GlobalAggregationMode::max});

  std::vector<index_type> sa;

  kamping::measurements::timer().synchronize_and_start("total_time");
  auto start_time = MPI_Wtime();
  if (doubling_discarding) {
    sa = dsss::suffix_sorting::prefix_doubling_discarding<index_type>(
        std::move(distributed_strings));
  } else /*inducing*/ {
    sa = dsss::suffix_sorting::inducing<index_type>(
        std::move(distributed_strings));
  }
  auto end_time = MPI_Wtime();
  kamping::measurements::timer().stop();
  kamping::measurements::counter().add(
      "mem_after_sa_construction", dsss::get_max_mem_bytes(),
      {kamping::measurements::GlobalAggregationMode::gather,
       kamping::measurements::GlobalAggregationMode::max});

  if (env.rank() == 0) {
    std::cout << "total_time=" << end_time - start_time << std::endl;
  }

  if (!output_path.empty()) {
    if (env.rank() == 0) {
      std::cout << "Writing the SA to " << output_path << std::endl;
    }
    dsss::mpi::write_data(sa, output_path);
    env.barrier();
    if (env.rank() == 0) {
      std::cout << "Finished writing the SA" << std::endl;
    }
  }

  long max_mem = dsss::get_max_mem_bytes();
  auto all_max_mem = dsss::mpi::allgather(max_mem);
  if (env.rank() == 0) {
    std::cout << "max_mem=";
    dsss::print_vector(all_max_mem, ",");
  }

  if (check) {
    if (string_size > 0) {
      distributed_strings =
          dsss::mpi::distribute_string(input_path, string_size);
    } else {
      distributed_strings = dsss::mpi::distribute_string(input_path);
    }

    if (!output_path.empty()) {
      if (env.rank() == 0) {
        std::cout << "To check if export of the SA was successful, we first "
                  << "load the exported file ... ";
      }
      sa = dsss::mpi::read_data<index_type>(output_path);
      if (env.rank() == 0) {
        std::cout << "DONE" << std::endl;
      }
    }
    if (env.rank() == 0) {
      std::cout << "Checking SA ... ";
    }
    bool correct = dsss::suffix_sorting::check(sa, distributed_strings.string);
    kamping::measurements::counter().add(
        "mem_after_sa_check", dsss::get_max_mem_bytes(),
        {kamping::measurements::GlobalAggregationMode::gather,
         kamping::measurements::GlobalAggregationMode::max});

    if (!correct && env.rank() == 0) {
      std::cout << "ERROR: Not a correct SA!" << std::endl;
    } else if (env.rank() == 0) {
      std::cout << "Correct SA!" << std::endl;
    }
    if (env.rank() == 0) {
      std::cout << "SA_ok=" << correct << "\n";
    }
  }

  std::stringstream sstream_counter;
  std::stringstream sstream_timer;

  auto config_vector = params.config();
  kamping::measurements::SimpleJsonPrinter<double> printer_timer(sstream_timer,
                                                                 config_vector);
  kamping::measurements::SimpleJsonPrinter<std::int64_t> printer_counter(
      sstream_counter, config_vector);
  kamping::measurements::timer().aggregate_and_print(printer_timer);
  kamping::measurements::counter().aggregate_and_print(printer_counter);
  kamping::measurements::timer().clear();
  kamping::measurements::counter().clear();

  auto print_as_jsonlist_to_file = [](std::vector<std::string> objects,
                                      std::string filename) {
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
  auto remove_extension_if_present = [](std::string const &path) {
    auto pos = path.find(".json");
    return path.substr(0, pos);
  };

  std::string const output_path =
      remove_extension_if_present(params.json_output_path);
  if (env.rank() == 0) {
    print_as_jsonlist_to_file({sstream_timer.str()},
                              params.json_output_path + "_timer.json");
    print_as_jsonlist_to_file({sstream_counter.str()},
                              params.json_output_path + "_counter.json");
  }

  env.finalize();
  return 0;
}

/******************************************************************************/
