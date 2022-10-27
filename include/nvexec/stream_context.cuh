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

#include "../stdexec/execution.hpp"
#include <type_traits>
#include <memory_resource>

#include "detail/config.cuh"
#include "stream/sync_wait.cuh"
#include "stream/bulk.cuh"
#include "stream/let_xxx.cuh"
#include "stream/schedule_from.cuh"
#include "stream/start_detached.cuh"
#include "stream/submit.cuh"
#include "stream/split.cuh"
#include "stream/then.cuh"
#include "stream/transfer.cuh"
#include "stream/upon_error.cuh"
#include "stream/upon_stopped.cuh"
#include "stream/when_all.cuh"
#include "stream/reduce.cuh"
#include "stream/ensure_started.cuh"

#include "stream/common.cuh"
#include "detail/queue.cuh"
#include "detail/throw_on_cuda_error.cuh"

namespace nvexec {
  namespace STDEXEC_STREAM_DETAIL_NS {
    template <std::execution::sender Sender, std::integral Shape, class Fun>
      using bulk_sender_th = bulk_sender_t<stdexec::__x<std::remove_cvref_t<Sender>>, Shape, stdexec::__x<std::remove_cvref_t<Fun>>>;

    template <std::execution::sender Sender>
      using split_sender_th = split_sender_t<stdexec::__x<std::remove_cvref_t<Sender>>>;

    template <std::execution::sender Sender, class Fun>
      using then_sender_th = then_sender_t<stdexec::__x<std::remove_cvref_t<Sender>>, stdexec::__x<std::remove_cvref_t<Fun>>>;

    template <class Scheduler, std::execution::sender... Senders>
      using when_all_sender_th = when_all_sender_t<false, Scheduler, stdexec::__x<std::decay_t<Senders>>...>;

    template <class Scheduler, std::execution::sender... Senders>
      using transfer_when_all_sender_th = when_all_sender_t<true, Scheduler, stdexec::__x<std::decay_t<Senders>>...>;

    template <std::execution::sender Sender, class Fun>
      using upon_error_sender_th = upon_error_sender_t<stdexec::__x<std::remove_cvref_t<Sender>>, stdexec::__x<std::remove_cvref_t<Fun>>>;

    template <std::execution::sender Sender, class Fun>
      using upon_stopped_sender_th = upon_stopped_sender_t<stdexec::__x<std::remove_cvref_t<Sender>>, stdexec::__x<std::remove_cvref_t<Fun>>>;

    template <class Let, std::execution::sender Sender, class Fun>
      using let_xxx_th = let_sender_t<stdexec::__x<std::remove_cvref_t<Sender>>, stdexec::__x<std::remove_cvref_t<Fun>>, Let>;

    template <std::execution::sender Sender>
      using transfer_sender_th = transfer_sender_t<stdexec::__x<Sender>>;

    template <std::execution::sender Sender>
      using ensure_started_th = ensure_started_sender_t<stdexec::__x<Sender>>;

    struct stream_scheduler {
      friend stream_context;

      template <std::execution::sender Sender>
        using schedule_from_sender_th = schedule_from_sender_t<stream_scheduler, stdexec::__x<std::remove_cvref_t<Sender>>>;

      template <class ReceiverId>
        struct operation_state_t : operation_state_base_t<ReceiverId> {
          using Receiver = stdexec::__t<ReceiverId>;

          cudaStream_t stream_{0};
          cudaError_t status_{cudaSuccess};

          operation_state_t(Receiver&& receiver, context_state_t context_state) 
            : operation_state_base_t<ReceiverId>((Receiver&&)receiver, context_state, false) {
          }

          friend void tag_invoke(std::execution::start_t, operation_state_t& op) noexcept {
            op.propagate_completion_signal(std::execution::set_value);
          }
        };

      struct sender_t : stream_sender_base {
        using completion_signatures =
          std::execution::completion_signatures<
            std::execution::set_value_t(),
            std::execution::set_error_t(cudaError_t)>;

        template <class R>
          friend auto tag_invoke(std::execution::connect_t, const sender_t& self, R&& rec)
            noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
            -> operation_state_t<stdexec::__x<std::remove_cvref_t<R>>> {
            return operation_state_t<stdexec::__x<std::remove_cvref_t<R>>>((R&&) rec, self.context_state_);
          }

        stream_scheduler make_scheduler() const {
          return stream_scheduler{context_state_};
        }

        template <class CPO>
        friend stream_scheduler
        tag_invoke(std::execution::get_completion_scheduler_t<CPO>, sender_t self) noexcept {
          return self.make_scheduler();
        }

        sender_t(context_state_t context_state) noexcept
          : context_state_(context_state) {
        }

        context_state_t context_state_;
      };

      template <std::execution::sender S>
        friend schedule_from_sender_th<S>
        tag_invoke(std::execution::schedule_from_t, const stream_scheduler& sch, S&& sndr) noexcept {
          return schedule_from_sender_th<S>(sch.context_state_, (S&&) sndr);
        }

      template <std::execution::sender S, std::integral Shape, class Fn>
        friend bulk_sender_th<S, Shape, Fn>
        tag_invoke(std::execution::bulk_t, const stream_scheduler& sch, S&& sndr, Shape shape, Fn fun) noexcept {
          return bulk_sender_th<S, Shape, Fn>{{}, (S&&) sndr, shape, (Fn&&)fun};
        }

      template <std::execution::sender S, class Fn>
        friend then_sender_th<S, Fn>
        tag_invoke(std::execution::then_t, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
          return then_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&)fun};
        }

      template <std::execution::sender S>
        friend ensure_started_th<S>
        tag_invoke(std::execution::ensure_started_t, const stream_scheduler& sch, S&& sndr) noexcept {
          return ensure_started_th<S>(sch.context_state_, (S&&) sndr);
        }

      template <stdexec::__one_of<
                  std::execution::let_value_t, 
                  std::execution::let_stopped_t, 
                  std::execution::let_error_t> Let, 
                std::execution::sender S, 
                class Fn>
        friend let_xxx_th<Let, S, Fn>
        tag_invoke(Let, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
          return let_xxx_th<Let, S, Fn>{{}, (S &&) sndr, (Fn &&) fun};
        }

      template <std::execution::sender S, class Fn>
        friend upon_error_sender_th<S, Fn>
        tag_invoke(std::execution::upon_error_t, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
          return upon_error_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&)fun};
        }

      template <std::execution::sender S, class Fn>
        friend upon_stopped_sender_th<S, Fn>
        tag_invoke(std::execution::upon_stopped_t, const stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
          return upon_stopped_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&)fun};
        }

      template <std::execution::sender... Senders>
        friend auto
        tag_invoke(std::execution::transfer_when_all_t, const stream_scheduler& sch, Senders&&... sndrs) noexcept {
          return transfer_when_all_sender_th<stream_scheduler, Senders...>(sch.context_state_, (Senders&&)sndrs...);
        }

      template <std::execution::sender... Senders>
        friend auto
        tag_invoke(std::execution::transfer_when_all_with_variant_t, const stream_scheduler& sch, Senders&&... sndrs) noexcept {
          return 
            transfer_when_all_sender_th<stream_scheduler, std::tag_invoke_result_t<std::execution::into_variant_t, Senders>...>(
                sch.context_state_, 
                std::execution::into_variant((Senders&&)sndrs)...);
        }

      template <std::execution::sender S, std::execution::scheduler Sch>
        friend auto
        tag_invoke(std::execution::transfer_t, const stream_scheduler& sch, S&& sndr, Sch&& scheduler) noexcept {
          return std::execution::schedule_from((Sch&&)scheduler, transfer_sender_th<S>(
            sch.context_state_, (S&&)sndr));
        }

      template <std::execution::sender S>
        friend split_sender_th<S>
        tag_invoke(std::execution::split_t, const stream_scheduler& sch, S&& sndr) noexcept {
          return split_sender_th<S>(sch.context_state_, (S&&)sndr);
        }

      friend sender_t tag_invoke(std::execution::schedule_t, const stream_scheduler& self) noexcept {
        return {self.context_state_};
      }

      friend std::true_type tag_invoke(stdexec::__has_algorithm_customizations_t, const stream_scheduler& self) noexcept {
        return {};
      }

      template <std::execution::sender S>
        friend auto
        tag_invoke(std::this_thread::sync_wait_t, const stream_scheduler& self, S&& sndr) {
          return sync_wait::sync_wait_t{}(self.context_state_, (S&&)sndr);
        }

      friend std::execution::forward_progress_guarantee tag_invoke(
          std::execution::get_forward_progress_guarantee_t,
          const stream_scheduler&) noexcept {
        return std::execution::forward_progress_guarantee::weakly_parallel;
      }

      bool operator==(const stream_scheduler& other) const noexcept {
        return context_state_.hub_ == other.context_state_.hub_;
      }

      stream_scheduler(context_state_t context_state)
        : context_state_(context_state) {
      }

    // private: TODO
      context_state_t context_state_;
    };

    template <stream_completing_sender Sender>
      void tag_invoke(std::execution::start_detached_t, Sender&& sndr) noexcept(false) {
        submit::submit_t{}((Sender&&)sndr, start_detached::detached_receiver_t{});
      }

    template <stream_completing_sender... Senders>
      when_all_sender_th<stream_scheduler, Senders...>
      tag_invoke(std::execution::when_all_t, Senders&&... sndrs) noexcept {
        return when_all_sender_th<stream_scheduler, Senders...>{context_state_t{nullptr, nullptr}, (Senders&&)sndrs...};
      }

    template <stream_completing_sender... Senders>
      when_all_sender_th<stream_scheduler, std::tag_invoke_result_t<std::execution::into_variant_t, Senders>...>
      tag_invoke(std::execution::when_all_with_variant_t, Senders&&... sndrs) noexcept {
        return when_all_sender_th<stream_scheduler, std::tag_invoke_result_t<std::execution::into_variant_t, Senders>...>{
          context_state_t{nullptr, nullptr}, 
          std::execution::into_variant((Senders&&)sndrs)...
        };
      }

    template <std::execution::sender S, class Fn>
      upon_error_sender_th<S, Fn>
      tag_invoke(std::execution::upon_error_t, S&& sndr, Fn fun) noexcept {
        return upon_error_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&)fun};
      }

    template <std::execution::sender S, class Fn>
      upon_stopped_sender_th<S, Fn>
      tag_invoke(std::execution::upon_stopped_t, S&& sndr, Fn fun) noexcept {
        return upon_stopped_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&)fun};
      }

    struct pinned_resource : public std::pmr::memory_resource {
      void* do_allocate(size_t bytes, size_t /* alignment */) override {
        void* ret;

        if (cudaError_t status = STDEXEC_DBG_ERR(cudaMallocHost(&ret, bytes)); status != cudaSuccess) {
          throw std::bad_alloc();
        }

        return ret;
      }

      void do_deallocate(void* ptr, size_t /* bytes */, size_t /* alignment */) override {
        STDEXEC_DBG_ERR(cudaFreeHost(ptr));
      }

      bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
      }
        
    private:
      std::pmr::memory_resource* _upstream;
    };
  }

  using STDEXEC_STREAM_DETAIL_NS::stream_scheduler;

  struct stream_context {
    STDEXEC_STREAM_DETAIL_NS::pinned_resource pinned_resource_{};
    std::pmr::monotonic_buffer_resource monotonic_resource_;
    std::pmr::synchronized_pool_resource resource_;

    STDEXEC_STREAM_DETAIL_NS::queue::task_hub_t hub_;

    stream_context()
      : monotonic_resource_(512 * 1024, &pinned_resource_)
      , resource_(&monotonic_resource_)
      , hub_(&resource_) {
    }

    stream_scheduler get_scheduler(stream_priority priority = stream_priority::normal) {
      return {STDEXEC_STREAM_DETAIL_NS::context_state_t(&resource_, &hub_, priority)};
    }
  };
}

