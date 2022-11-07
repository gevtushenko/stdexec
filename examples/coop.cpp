/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include <stdexec/execution.hpp>

#include <barrier>
#include <thread>
#include <vector>

namespace coop {
  struct set_value_t {
    template <class _Receiver, class... _As>
        requires stdexec::tag_invocable<set_value_t, _Receiver, _As...>
      void operator()(_Receiver&& __rcvr, _As&&... __as) const noexcept {
        static_assert(stdexec::nothrow_tag_invocable<set_value_t, _Receiver, _As...>);
        (void) tag_invoke(set_value_t{}, (_Receiver&&) __rcvr, (_As&&) __as...);
      }

    friend constexpr bool tag_invoke(stdexec::custom_completion_channel_t, set_value_t) noexcept {
      return true;
    }
  };

  inline constexpr set_value_t set_value{};

  struct context_state {
    std::size_t thread_id_{};
    std::size_t num_threads_{};
    std::barrier<>* barrier_{};
  };

  struct inline_scheduler {
    context_state state_;

    template <class R_>
      struct __op {
        using R = stdexec::__t<R_>;
        [[no_unique_address]] R rec_;
        friend void tag_invoke(stdexec::start_t, __op& op) noexcept try {
          stdexec::set_value((R&&) op.rec_);
        } catch(...) {
          stdexec::set_error((R&&) op.rec_, std::current_exception());
        }
      };

    struct __sender {
      using completion_signatures =
        stdexec::completion_signatures<
          stdexec::set_value_t(),
          stdexec::set_error_t(std::exception_ptr)>;

      template <class R>
        friend auto tag_invoke(stdexec::connect_t, __sender, R&& rec)
          noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
          -> __op<stdexec::__x<std::remove_cvref_t<R>>> {
          return {(R&&) rec};
        }

      friend inline_scheduler
      tag_invoke(stdexec::get_completion_scheduler_t<stdexec::set_value_t>, __sender) noexcept {
        return {};
      }
    };

    friend __sender tag_invoke(stdexec::schedule_t, const inline_scheduler&) noexcept {
      return {};
    }

    friend stdexec::forward_progress_guarantee tag_invoke(
        stdexec::get_forward_progress_guarantee_t,
        const inline_scheduler&) noexcept {
      return stdexec::forward_progress_guarantee::weakly_parallel;
    }

    bool operator==(const inline_scheduler&) const noexcept = default;
  };

  struct inline_context {
    std::size_t num_threads_{1};
    std::barrier<> barrier_;

    inline_context(std::size_t num_threads)
      : num_threads_(num_threads)
      , barrier_(num_threads_) {
    }

    inline_scheduler get_scheduler(std::size_t thread_id) {
      return {context_state{thread_id, num_threads_, &barrier_}};
    }
  };
}

int main() {
  const std::size_t num_threads{2};
  coop::inline_context ctx{num_threads};
  std::vector<std::thread> threads{num_threads};

  for (std::size_t thread_id = 0; thread_id < num_threads; thread_id++) {
    threads[thread_id] = std::thread([thread_id, &ctx] {
      auto sch = ctx.get_scheduler(thread_id);
      auto snd = stdexec::schedule(sch) 
               | stdexec::then([] { 
                   std::printf("+\n"); 
                 });
      stdexec::sync_wait(snd);
    });
  }

  for (auto& thread: threads) {
    thread.join();
  }
}

