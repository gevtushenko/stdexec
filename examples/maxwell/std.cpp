/*
 * Copyright (c) NVIDIA
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common.hpp"

#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string_view>
#include <vector>
#include <barrier>

template <class Shape>
std::pair<Shape, Shape> 
even_share(Shape n, std::uint32_t rank, std::uint32_t size) noexcept {
  const auto avg_per_thread = n / size;
  const auto n_big_share = avg_per_thread + 1;
  const auto big_shares = n % size;
  const auto is_big_share = rank < big_shares;
  const auto begin = is_big_share ? n_big_share * rank
                                  : n_big_share * big_shares +
                                      (rank - big_shares) * avg_per_thread;
  const auto end = begin + (is_big_share ? n_big_share : avg_per_thread);

  return std::make_pair(begin, end);
}

void run_omp(float dt, bool write_vtk, int node_id, std::size_t n_inner_iterations,
             std::size_t n_outer_iterations, grid_t &grid,
             std::string_view scheduler_name) {
  fields_accessor accessor = grid.accessor();

  auto initializer = grid_initializer(dt, accessor);
  for (std::size_t i = 0; i < accessor.cells; i++) {
    initializer(i);
  }

  const std::size_t n_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(n_threads);
  std::barrier barrier(n_threads);

  report_performance(grid.cells, n_inner_iterations * n_outer_iterations,
                     node_id, scheduler_name, [&]() {
                       for (std::size_t tid = 0; tid < n_threads; tid++) {
                         threads[tid] = std::thread([=, &barrier] {
                           time_storage_t time{};
                           auto h_updater = update_h(accessor);
                           auto e_updater = update_e(time.get(), dt, accessor);

                           auto [begin, end] = even_share(accessor.cells, tid, n_threads);

                           std::size_t report_step = 0;
                           const bool writer_thread = write_vtk && tid == 0;
                           auto writer = dump_vtk(writer_thread, node_id, report_step, accessor);

                           for (; report_step < n_outer_iterations;) {
                             for (std::size_t compute_step = 0;
                                  compute_step < n_inner_iterations;
                                  compute_step++) {
                               for (std::size_t i = begin; i < end; i++) {
                                 h_updater(i);
                               }
                               barrier.arrive_and_wait();
                               for (std::size_t i = begin; i < end; i++) {
                                 e_updater(i);
                               }
                               barrier.arrive_and_wait();
                             }

                             writer();
                             if (write_vtk) {
                               barrier.arrive_and_wait();
                             }
                           }
                         });
                       }

                       for (std::size_t tid = 0; tid < n_threads; tid++) {
                         threads[tid].join();
                       }
                     });
}

int main(int argc, char *argv[]) {
  auto params = parse_cmd(argc, argv);

  if (value(params, "help") || value(params, "h")) {
    std::cout << "Usage: " << argv[0] << " [OPTION]...\n"
              << "\t--write-vtk\n"
              << "\t--write-results\n"
              << "\t--inner-iterations\n"
              << "\t--validate\n"
              << "\t--N\n"
              << std::endl;
    return 0;
  }

  const bool validate = value(params, "validate");
  const bool write_vtk = value(params, "write-vtk");
  const bool write_results = value(params, "write-results");
  const std::size_t n_inner_iterations = value(params, "inner-iterations", 100);
  const std::size_t n_outer_iterations = value(params, "outer-iterations", 10);
  const std::size_t N = value(params, "N", 2048);
  std::size_t run_distributed_default = 1;

  auto run = [&](std::string_view scheduler_name) {
    const auto node_id = 0;
    const auto n_nodes = 1;

    grid_t grid{N, 0, N * N};

    auto accessor = grid.accessor();
    auto dt = calculate_dt(accessor.dx, accessor.dy);

    run_omp(dt, write_vtk, node_id, n_inner_iterations, n_outer_iterations, grid, scheduler_name);

    if (validate) {
      validate_results(node_id, n_nodes, accessor);
    }
    if (write_results) {
      store_results(node_id, n_nodes, accessor);
    }

    run_distributed_default = 0;
  };

  report_header();

  run("CPU (OpenMP)");
}

