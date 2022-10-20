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

#include <stdexec/execution.hpp>
#include <type_traits>

#include "nvexec/stream_context.cuh"

namespace nvexec {
  namespace STDEXEC_STREAM_DETAIL_NS {
    template <std::execution::sender Sender, std::integral Shape, class Fun>
      using multi_gpu_bulk_sender_th = multi_gpu_bulk_sender_t<stdexec::__x<std::remove_cvref_t<Sender>>, Shape, stdexec::__x<std::remove_cvref_t<Fun>>>;

    struct multi_gpu_stream_scheduler {
      friend stream_context;

      template <std::execution::sender Sender>
        using schedule_from_sender_th = schedule_from_sender_t<stream_scheduler, stdexec::__x<std::remove_cvref_t<Sender>>>;

      template <class RId>
        struct operation_state_t : stream_op_state_base {
          using R = stdexec::__t<RId>;

          R rec_;
          cudaStream_t stream_{0};
          cudaError_t status_{cudaSuccess};

          template <stdexec::__decays_to<R> Receiver>
            operation_state_t(Receiver&& rec) : rec_((Receiver&&)rec) {
              status_ = STDEXEC_DBG_ERR(cudaStreamCreate(&stream_));
            }

          ~operation_state_t() {
            STDEXEC_DBG_ERR(cudaStreamDestroy(stream_));
          }

          cudaStream_t get_stream() {
            return stream_;
          }

          friend void tag_invoke(std::execution::start_t, operation_state_t& op) noexcept {
            if constexpr (stream_receiver<R>) {
              if (op.status_ == cudaSuccess) {
                std::execution::set_value((R&&)op.rec_);
              } else {
                std::execution::set_error((R&&)op.rec_, std::move(op.status_));
              }
            } else {
              if (op.status_ == cudaSuccess) {
                continuation_kernel
                  <std::decay_t<R>, std::execution::set_value_t>
                    <<<1, 1, 0, op.stream_>>>(op.rec_, std::execution::set_value);
              } else {
                continuation_kernel
                  <std::decay_t<R>, std::execution::set_error_t, cudaError_t>
                    <<<1, 1, 0, op.stream_>>>(op.rec_, std::execution::set_error, op.status_);
              }
            }
          }
        };

      struct sender_t : stream_sender_base {
        using completion_signatures =
          std::execution::completion_signatures<
            std::execution::set_value_t(),
            std::execution::set_error_t(cudaError_t)>;

        template <class R>
          friend auto tag_invoke(std::execution::connect_t, sender_t, R&& rec)
            noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
            -> operation_state_t<stdexec::__x<std::remove_cvref_t<R>>> {
            return operation_state_t<stdexec::__x<std::remove_cvref_t<R>>>((R&&) rec);
          }

        multi_gpu_stream_scheduler make_scheduler() const {
          return multi_gpu_stream_scheduler{num_devices_, hub_};
        }

        template <class CPO>
        friend multi_gpu_stream_scheduler
        tag_invoke(std::execution::get_completion_scheduler_t<CPO>, sender_t self) noexcept {
          return self.make_scheduler();
        }

        sender_t(int num_devices,
                 queue::task_hub_t* hub) noexcept
          : num_devices_(num_devices)
          , hub_(hub) {}

        int num_devices_;
        queue::task_hub_t * hub_;
      };

      template <std::execution::sender S>
        friend schedule_from_sender_th<S>
        tag_invoke(std::execution::schedule_from_t, const multi_gpu_stream_scheduler& sch, S&& sndr) noexcept {
          return schedule_from_sender_th<S>(sch.hub_, (S&&) sndr);
        }

      template <std::execution::sender S, std::integral Shape, class Fn>
        friend multi_gpu_bulk_sender_th<S, Shape, Fn>
        tag_invoke(std::execution::bulk_t, const multi_gpu_stream_scheduler& sch, S&& sndr, Shape shape, Fn fun) noexcept {
          return multi_gpu_bulk_sender_th<S, Shape, Fn>{{}, sch.num_devices_, (S&&) sndr, shape, (Fn&&)fun};
        }

      template <std::execution::sender S, class Fn>
        friend then_sender_th<S, Fn>
        tag_invoke(std::execution::then_t, const multi_gpu_stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
          return then_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&)fun};
        }

      template <stdexec::__one_of<std::execution::let_value_t, 
                                  std::execution::let_stopped_t, 
                                  std::execution::let_error_t> Let, 
                std::execution::sender S, 
                class Fn>
        friend let_xxx_th<Let, S, Fn>
        tag_invoke(Let, const multi_gpu_stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
          return let_xxx_th<Let, S, Fn>{{}, (S &&) sndr, (Fn &&) fun};
        }

      template <std::execution::sender S, class Fn>
        friend upon_error_sender_th<S, Fn>
        tag_invoke(std::execution::upon_error_t, const multi_gpu_stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
          return upon_error_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&)fun};
        }

      template <std::execution::sender S, class Fn>
        friend upon_stopped_sender_th<S, Fn>
        tag_invoke(std::execution::upon_stopped_t, const multi_gpu_stream_scheduler& sch, S&& sndr, Fn fun) noexcept {
          return upon_stopped_sender_th<S, Fn>{{}, (S&&) sndr, (Fn&&)fun};
        }

      template <std::execution::sender... Senders>
        friend auto
        tag_invoke(std::execution::transfer_when_all_t, const multi_gpu_stream_scheduler& sch, Senders&&... sndrs) noexcept {
          return transfer_when_all_sender_th<multi_gpu_stream_scheduler, Senders...>(sch.hub_, (Senders&&)sndrs...);
        }

      template <std::execution::sender... Senders>
        friend auto
        tag_invoke(std::execution::transfer_when_all_with_variant_t, const multi_gpu_stream_scheduler& sch, Senders&&... sndrs) noexcept {
          return 
            transfer_when_all_sender_th<multi_gpu_stream_scheduler, std::tag_invoke_result_t<std::execution::into_variant_t, Senders>...>(
                sch.hub_, 
                std::execution::into_variant((Senders&&)sndrs)...);
        }

      template <std::execution::sender S, std::execution::scheduler Sch>
        friend auto
        tag_invoke(std::execution::transfer_t, const multi_gpu_stream_scheduler& sch, S&& sndr, Sch&& scheduler) noexcept {
          return std::execution::schedule_from((Sch&&)scheduler, transfer_sender_th<S>(sch.hub_, (S&&)sndr));
        }

      template <std::execution::sender S>
        friend split_sender_th<S>
        tag_invoke(std::execution::split_t, const multi_gpu_stream_scheduler& sch, S&& sndr) noexcept {
          return split_sender_th<S>((S&&)sndr, sch.hub_);
        }

      template <std::execution::sender S>
        friend ensure_started_th<S>
        tag_invoke(std::execution::ensure_started_t, const multi_gpu_stream_scheduler& sch, S&& sndr) noexcept {
          return ensure_started_th<S>((S&&) sndr, sch.hub_);
        }

      friend sender_t tag_invoke(std::execution::schedule_t, const multi_gpu_stream_scheduler& self) noexcept {
        return {self.num_devices_, self.hub_};
      }

      template <std::execution::sender S>
        friend auto
        tag_invoke(std::this_thread::sync_wait_t, const multi_gpu_stream_scheduler& self, S&& sndr) {
          return sync_wait::sync_wait_t{}(self.hub_, (S&&)sndr);
        }

      friend std::execution::forward_progress_guarantee tag_invoke(
          std::execution::get_forward_progress_guarantee_t,
          const multi_gpu_stream_scheduler&) noexcept {
        return std::execution::forward_progress_guarantee::weakly_parallel;
      }

      bool operator==(const multi_gpu_stream_scheduler&) const noexcept = default;

      multi_gpu_stream_scheduler(
          int num_devices,
          const queue::task_hub_t* hub)
        : num_devices_(num_devices)
        , hub_(const_cast<queue::task_hub_t*>(hub)) {
      }

    // private: TODO
      int num_devices_{};
      queue::task_hub_t* hub_{};
    };
  }

  using STDEXEC_STREAM_DETAIL_NS::multi_gpu_stream_scheduler;

  struct multi_gpu_stream_context {
    int num_devices{};
    STDEXEC_STREAM_DETAIL_NS::queue::task_hub_t hub{};

    multi_gpu_stream_context() {
      // TODO Manage errors
      int current_device{};
      cudaGetDevice(&current_device);
      cudaGetDeviceCount(&num_devices);
      
      for (int dev_id = 0; dev_id < num_devices; dev_id++) {
        cudaSetDevice(dev_id);
        for (int peer_id = 0; peer_id < num_devices; peer_id++) {
          if (peer_id != dev_id) {
            int can_access{};
            cudaDeviceCanAccessPeer(&can_access, dev_id, peer_id);

            if (can_access) {
              cudaDeviceEnablePeerAccess(peer_id, 0);
            }
          }
        }
      }
      cudaSetDevice(current_device);
    }

    multi_gpu_stream_scheduler get_scheduler() {
      return {num_devices, &hub};
    }
  };
}

