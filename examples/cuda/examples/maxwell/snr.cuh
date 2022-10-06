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

// #include <schedulers/stream.cuh>
#include <schedulers/inline_scheduler.hpp>
#include <schedulers/static_thread_pool.hpp>

namespace ex = std::execution;
// namespace stream = example::cuda::stream;

bool is_on_gpu() { return false; }

namespace async_repeat_n_detail {

  template <class OpT>
    class receiver_t {
      using Sender = typename OpT::Sender;
      using Receiver = typename OpT::Receiver;

      OpT &op_state_;

    public:
      template <std::__one_of<ex::set_error_t, ex::set_stopped_t> _Tag, class... _Args _NVCXX_CAPTURE_PACK(_Args)>
        friend void tag_invoke(_Tag __tag, receiver_t&& __self, _Args&&... __args) noexcept {
          _NVCXX_EXPAND_PACK(_Args, __args,
            OpT &op_state = __self.op_state_;
            __tag((Receiver&&)op_state.receiver_, (_Args&&)__args...);
          )
        }

      friend void tag_invoke(ex::set_value_t, receiver_t&& __self) noexcept {
        using inner_op_state_t = ex::connect_result_t<Sender, receiver_t>;

        OpT &op_state = __self.op_state_;

        if (op_state.i_ == op_state.n_) {
          ex::set_value((Receiver&&)op_state.receiver_);
          return;
        }

        op_state.i_++;
        op_state.inner_op_state_.~inner_op_state_t();
        new (&op_state.inner_op_state_) inner_op_state_t{ex::connect((Sender&&)op_state.sender_, receiver_t{op_state})};
        ex::start(op_state.inner_op_state_);
      }

      friend auto tag_invoke(ex::get_env_t, const receiver_t& self)
        -> ex::env_of_t<Receiver> {
        return ex::get_env(self.op_state_.receiver_);
      }

      explicit receiver_t(OpT& op_state)
        : op_state_(op_state)
      {}
    };

  template <class SenderId, class ReceiverId>
    struct operation_state_t {
      using Sender = std::__t<SenderId>;
      using Receiver = std::__t<ReceiverId>;

      using inner_op_state_t = ex::connect_result_t<Sender, receiver_t<operation_state_t>>;

      Sender sender_;
      Receiver receiver_;
      inner_op_state_t inner_op_state_;
      std::size_t n_{};
      std::size_t i_{};

      friend void
      tag_invoke(std::execution::start_t, operation_state_t &self) noexcept {
        ex::start(self.inner_op_state_);
      }

      operation_state_t(Sender&& sender, Receiver&& receiver, std::size_t n)
        : sender_{(Sender&&)sender}
        , receiver_{(Receiver&&)receiver}
        , inner_op_state_(ex::connect((Sender&&)sender_, receiver_t<operation_state_t>{*this}))
        , n_(n)
      {}
    };

  template <class SenderId>
    struct repeat_n_sender_t {
      using Sender = std::__t<SenderId>;

      using completion_signatures = std::execution::completion_signatures<
        std::execution::set_value_t(),
        std::execution::set_error_t(std::exception_ptr)>;

      Sender sender_;
      std::size_t n_{};

      template <std::__decays_to<repeat_n_sender_t> Self, class Receiver>
        requires std::tag_invocable<std::execution::connect_t, Sender, Receiver> friend auto
      tag_invoke(std::execution::connect_t, Self &&self, Receiver &&r) 
        -> operation_state_t<SenderId, std::__x<Receiver>> {
        return operation_state_t<SenderId, std::__x<Receiver>>(
          (Sender&&)self.sender_,
          (Receiver&&)r,
          self.n_);
      }

      template <std::__none_of<std::execution::connect_t> Tag, class... Ts>
        requires std::tag_invocable<Tag, Sender, Ts...> friend decltype(auto)
      tag_invoke(Tag tag, const repeat_n_sender_t &s, Ts &&...ts) noexcept {
        return tag(s.sender_, std::forward<Ts>(ts)...);
      }
    };

  struct repeat_n_t {
    template <class Sender>
    repeat_n_sender_t<std::__x<Sender>> operator()(std::size_t n, Sender &&__sndr) const noexcept {
      return repeat_n_sender_t<std::__x<Sender>>{std::forward<Sender>(__sndr), n};
    }
  };

} 

inline constexpr async_repeat_n_detail::repeat_n_t async_repeat_n{};

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
             async_repeat_n(
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

