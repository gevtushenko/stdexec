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

#include <execution.hpp>
#include <../examples/schedulers/inline_scheduler.hpp>
#include <../examples/schedulers/static_thread_pool.hpp>

#include "common.hpp"

#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string_view>
#include <vector>

namespace ex = std::execution;

namespace detail {

template <class Sender, class Receiver>
struct operation_state_t {
  Sender sender_;
  Receiver receiver_;
  std::size_t n_{};

  friend void tag_invoke(std::execution::start_t,
                         operation_state_t &self) noexcept {
    for (std::size_t i = 0; i < self.n_; i++) {
      std::this_thread::sync_wait(std::move(self.sender_));
    }

    std::execution::set_value(std::move(self.receiver_));
  }
};

template <class SID>
struct repeat_n_sender_t {
  using S = std::__t<SID>;

  S sender_;
  std::size_t n_{};

  template <std::__decays_to<repeat_n_sender_t> Self, class Receiver>
  requires std::tag_invocable<std::execution::connect_t, S, Receiver>
  friend auto tag_invoke(std::execution::connect_t, Self &&self, Receiver &&r) {
    return operation_state_t<S, Receiver>{std::move(self.sender_),
                                          std::forward<Receiver>(r), self.n_};
  }

  template <class _Tag, class... _As>
    requires std::__callable<_Tag, const S&, _As...>
  friend auto tag_invoke(_Tag __tag, const repeat_n_sender_t& __self, _As&&... __as)
    noexcept(std::__nothrow_callable<_Tag, const S&, _As...>)
    -> std::__call_result_if_t<std::execution::tag_category<_Tag, std::execution::forwarding_sender_query>, _Tag, const S&, _As...> {
    return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
  }

  template <std::__decays_to<repeat_n_sender_t> Self, class Env>
  friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
    -> std::execution::dependent_completion_signatures<Env>;

  template <std::__decays_to<repeat_n_sender_t> Self, class Env>
  friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
    -> std::execution::make_completion_signatures<std::__member_t<Self, S>, Env> requires true;
};

struct repeat_n_t {
  template <class _Sender>
  auto operator()(std::size_t n, _Sender &&__sndr) const noexcept {
    return repeat_n_sender_t<std::__x<_Sender>>{std::forward<_Sender>(__sndr), n};
  }
};

}  // namespace detail

inline constexpr detail::repeat_n_t repeat_n{};


auto maxwell_eqs(float dt, float *time, bool write_results, int node_id,
                 std::size_t &report_step, std::size_t n_inner_iterations,
                 std::size_t n_outer_iterations, fields_accessor accessor,
                 std::execution::scheduler auto &&computer,
                 std::execution::scheduler auto &&writer) {
  auto write = dump_vtk(write_results, node_id, report_step, accessor);

  return repeat_n(                                                    //
      n_outer_iterations,                                             //
      repeat_n(                                                       //
          n_inner_iterations,                                         //
          ex::schedule(computer) |                                    //
              ex::bulk(accessor.cells, update_h(accessor)) |          //
              ex::bulk(accessor.cells, update_e(time, dt, accessor))  //
          ) |                                                         //
          ex::transfer(writer) |                                      //
          ex::then(std::move(write)));
}

void run(float dt, bool write_vtk, int node_id, std::size_t n_inner_iterations,
         std::size_t n_outer_iterations, grid_t &grid,
         std::string_view scheduler_name,
         std::execution::scheduler auto &&computer) {
  example::inline_scheduler writer{};

  time_storage_t time{};
  fields_accessor accessor = grid.accessor();

  std::this_thread::sync_wait(
      ex::schedule(computer) |
      ex::bulk(grid.cells, grid_initializer(dt, accessor)));

  std::size_t report_step = 0;
  auto snd = maxwell_eqs(dt, time.get(), write_vtk, node_id, report_step,
                         n_inner_iterations, n_outer_iterations, accessor,
                         computer, writer);

  report_performance(grid.cells, n_inner_iterations * n_outer_iterations,
                     node_id, scheduler_name,
                     [&]() { std::this_thread::sync_wait(std::move(snd)); });
}

int main(int argc, char *argv[]) {
  auto params = parse_cmd(argc, argv);

  if (value(params, "help") || value(params, "h")) {
    std::cout << "Usage: " << argv[0] << " [OPTION]...\n"
              << "\t--write-vtk\n"
              << "\t--write-results\n"
              << "\t--inner-iterations\n"
              << "\t--validate\n"
              << "\t--run-cpu\n"
              << "\t--run-thread-pool\n"
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

  auto run_on = [&](std::string_view scheduler_name,
                    std::execution::scheduler auto &&scheduler) {
    const auto [grid_begin, grid_end] = bulk_range(N * N, scheduler);
    const auto node_id = node_id_from(scheduler);
    const auto n_nodes = n_nodes_from(scheduler);

    grid_t grid{N, grid_begin, grid_end};

    auto accessor = grid.accessor();
    auto dt = calculate_dt(accessor.dx, accessor.dy);

    run(dt, write_vtk, node_id, n_inner_iterations, n_outer_iterations, grid,
        scheduler_name, std::forward<decltype(scheduler)>(scheduler));

    if (validate) {
      validate_results(node_id, n_nodes, accessor);
    }
    if (write_results) {
      store_results(node_id, n_nodes, accessor);
    }

    run_distributed_default = 0;
  };

  report_header();

  if (value(params, "run-cpu")) {
    run_on("CPU (inline)", example::inline_scheduler{});
  }
  if (value(params, "run-thread-pool")) {
    example::static_thread_pool thread_pool{std::thread::hardware_concurrency()};
    run_on("CPU (thread pool)", thread_pool.get_scheduler());
  }
}

