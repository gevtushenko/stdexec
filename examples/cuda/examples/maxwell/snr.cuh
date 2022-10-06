/*
 * Copyright (c) 2022 NVIDIA Corporation
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
#include "schedulers/detail/throw_on_cuda_error.cuh"

// #include <schedulers/stream.cuh>
#include <schedulers/inline_scheduler.hpp>
#include <schedulers/static_thread_pool.hpp>

namespace ex = std::execution;
// namespace stream = example::cuda::stream;

bool is_on_gpu() { return false; }

namespace repeat_n_detail {

  template <class SenderId, class ReceiverId>
    struct operation_state_t {
      using Sender = _P2300::__t<SenderId>;
      using Receiver = _P2300::__t<ReceiverId>;

      Sender sender_;
      Receiver receiver_;
      std::size_t n_{};

      friend void
      tag_invoke(std::execution::start_t, operation_state_t &self) noexcept {
        for (std::size_t i = 0; i < self.n_; i++) {
          std::this_thread::sync_wait((Sender&&)self.sender_);
        }
        ex::set_value((Receiver&&)self.receiver_);
      }

      operation_state_t(Sender&& sender, Receiver&& receiver, std::size_t n)
        : sender_{(Sender&&)sender}
        , receiver_{(Receiver&&)receiver}
        , n_(n)
      {}
    };

  template <class SenderId>
    struct repeat_n_sender_t {
      using Sender = _P2300::__t<SenderId>;

      using completion_signatures = std::execution::completion_signatures<
        std::execution::set_value_t(),
        std::execution::set_error_t(std::exception_ptr)>;

      Sender sender_;
      std::size_t n_{};

      template <_P2300::__decays_to<repeat_n_sender_t> Self, class Receiver>
        requires std::tag_invocable<std::execution::connect_t, Sender, Receiver> friend auto
      tag_invoke(std::execution::connect_t, Self &&self, Receiver &&r)
        -> operation_state_t<SenderId, _P2300::__x<Receiver>> {
        return operation_state_t<SenderId, _P2300::__x<Receiver>>(
          (Sender&&)self.sender_,
          (Receiver&&)r,
          self.n_);
      }

      template <_P2300::__none_of<std::execution::connect_t> Tag, class... Ts>
        requires std::tag_invocable<Tag, Sender, Ts...> friend decltype(auto)
      tag_invoke(Tag tag, const repeat_n_sender_t &s, Ts &&...ts) noexcept {
        return tag(s.sender_, std::forward<Ts>(ts)...);
      }
    };

  struct repeat_n_t {
    template <class Sender>
    repeat_n_sender_t<_P2300::__x<Sender>> operator()(std::size_t n, Sender &&__sndr) const noexcept {
      return repeat_n_sender_t<_P2300::__x<Sender>>{std::forward<Sender>(__sndr), n};
    }
  };

}

inline constexpr repeat_n_detail::repeat_n_t repeat_n{};

template <class SchedulerT>
[[nodiscard]] bool is_gpu_scheduler(SchedulerT &&scheduler) {
  auto snd = ex::schedule(scheduler) | ex::then([] { return is_on_gpu(); });
  auto [on_gpu] = std::this_thread::sync_wait(std::move(snd)).value();
  return on_gpu;
}

auto maxwell_eqs_snr(float dt,
                     float *time,
                     bool write_results,
                     std::size_t &report_step,
                     std::size_t n_inner_iterations,
                     std::size_t n_outer_iterations,
                     fields_accessor accessor,
                     std::execution::scheduler auto &&computer,
                     std::execution::scheduler auto &&writer) {
  auto write = dump_vtk(write_results, report_step, accessor);

  return repeat_n(
           n_outer_iterations,
             repeat_n(
               n_inner_iterations,
                 ex::schedule(computer)
               | ex::bulk(accessor.cells, update_h(accessor))
               | ex::bulk(accessor.cells, update_e(time, dt, accessor)))
           | ex::transfer(writer)
           | ex::then(std::move(write)));
}

void run_snr(float dt,
             bool write_vtk,
             std::size_t n_inner_iterations,
             std::size_t n_outer_iterations,
             grid_t &grid,
             std::string_view scheduler_name,
             std::execution::scheduler auto &&computer) {
  example::inline_scheduler writer{};

  time_storage_t time{is_gpu_scheduler(computer)};
  fields_accessor accessor = grid.accessor();

  std::this_thread::sync_wait(
    ex::schedule(computer) |
    ex::bulk(grid.cells, grid_initializer(dt, accessor)));

  std::size_t report_step = 0;
  auto snd = maxwell_eqs_snr(dt,
                             time.get(),
                             write_vtk,
                             report_step,
                             n_inner_iterations,
                             n_outer_iterations,
                             accessor,
                             computer,
                             writer);

  report_performance(grid.cells,
                     n_inner_iterations * n_outer_iterations,
                     scheduler_name,
                     [&snd] { std::this_thread::sync_wait(std::move(snd)); });
}

