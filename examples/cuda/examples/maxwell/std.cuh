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
#pragma once

#include "common.cuh"

#include <thread>
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

void run_std(float dt, bool write_vtk, std::size_t n_inner_iterations,
             std::size_t n_outer_iterations, grid_t &grid,
             std::string_view method) {
  fields_accessor accessor = grid.accessor();

  // TODO Initialize by threads that access data to preserve locality (NUMA)
  auto initializer = grid_initializer(dt, accessor);
  for (std::size_t i = 0; i < accessor.cells; i++) {
    initializer(i);
  }

  const std::size_t n_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(n_threads);
  std::barrier barrier(n_threads);

  report_performance(grid.cells, n_inner_iterations * n_outer_iterations, method,
                     [&]() {
                       for (std::size_t tid = 0; tid < n_threads; tid++) {
                         threads[tid] = std::thread([=, &barrier] {
                           time_storage_t time{false};
                           auto h_updater = update_h(accessor);
                           auto e_updater = update_e(time.get(), dt, accessor);

                           auto [begin, end] = even_share(accessor.cells, tid, n_threads);

                           std::size_t report_step = 0;
                           const bool writer_thread = write_vtk && tid == 0;
                           auto writer = dump_vtk(writer_thread, report_step, accessor);

                           for (; report_step < n_outer_iterations; report_step++) {
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

                             writer(false);
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
