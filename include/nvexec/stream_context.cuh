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
#include "stream/bulk.cuh"

#include "stream/common.cuh"
#include "detail/queue.cuh"
#include "detail/throw_on_cuda_error.cuh"

namespace nvexec {
  namespace STDEXEC_STREAM_DETAIL_NS {
    template <stdexec::sender Sender, std::integral Shape, class Fun>
      using bulk_sender_th = stdexec::__t<bulk_sender_t<stdexec::__id<std::decay_t<Sender>>, Shape, Fun>>;

    struct stream_scheduler {
      friend stream_context;

      template <class ReceiverId>
        struct operation_state_ {
          using Receiver = stdexec::__t<ReceiverId>;

          struct __t {
            using __id = operation_state_;

            cudaStream_t stream_{0};
            cudaError_t status_{cudaSuccess};

            __t(Receiver&& receiver, context_state_t context_state) {
            }

            friend void tag_invoke(stdexec::start_t, __t& op) noexcept {
              op.propagate_completion_signal(stdexec::set_value);
            }
          };
        };

      template <class ReceiverId>
        using operation_state_t = stdexec::__t<operation_state_<ReceiverId>>;

      struct sender_ {
        using __t = sender_;
        using __id = sender_;
        using completion_signatures_ =
          stdexec::completion_signatures<
            stdexec::set_value_t()>;

        template <class _Env>
          friend auto tag_invoke(stdexec::get_completion_signatures_t, const sender_&, _Env)
            -> completion_signatures_;

        template <class R>
          friend auto tag_invoke(stdexec::connect_t, const sender_& self, R&& rec)
            noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
            -> operation_state_t<stdexec::__id<std::remove_cvref_t<R>>> {
            return operation_state_t<stdexec::__id<std::remove_cvref_t<R>>>((R&&) rec, self.context_state_);
          }

        stream_scheduler make_scheduler() const {
          return stream_scheduler{context_state_};
        }

        template <class CPO>
          friend stream_scheduler
          tag_invoke(stdexec::get_completion_scheduler_t<CPO>, sender_ self) noexcept {
            return self.make_scheduler();
          }

        sender_(context_state_t context_state) noexcept
          : context_state_(context_state) {
        }

        context_state_t context_state_;
      };

      using sender_t = stdexec::__t<sender_>;

      template <stdexec::sender S, std::integral Shape, class Fn>
        friend bulk_sender_th<S, Shape, Fn>
        tag_invoke(stdexec::bulk_t, const stream_scheduler& sch, S&& sndr, Shape shape, Fn fun) noexcept {
          return bulk_sender_th<S, Shape, Fn>{{}, (S&&) sndr, shape, (Fn&&)fun};
        }

      friend sender_t tag_invoke(stdexec::schedule_t, const stream_scheduler& self) noexcept {
        return {self.context_state_};
      }

      friend std::true_type tag_invoke(stdexec::__has_algorithm_customizations_t, const stream_scheduler& self) noexcept {
        return {};
      }

      friend stdexec::forward_progress_guarantee tag_invoke(
          stdexec::get_forward_progress_guarantee_t,
          const stream_scheduler&) noexcept {
        return stdexec::forward_progress_guarantee::weakly_parallel;
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

    struct gpu_resource : public std::pmr::memory_resource {
      void* do_allocate(size_t bytes, size_t /* alignment */) override {
        void* ret;

        if (cudaError_t status = STDEXEC_DBG_ERR(cudaMalloc(&ret, bytes)); status != cudaSuccess) {
          throw std::bad_alloc();
        }

        return ret;
      }

      void do_deallocate(void* ptr, size_t /* bytes */, size_t /* alignment */) override {
        STDEXEC_DBG_ERR(cudaFree(ptr));
      }

      bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
      }
        
    private:
      std::pmr::memory_resource* _upstream;
    };

    struct managed_resource : public std::pmr::memory_resource {
      void* do_allocate(size_t bytes, size_t /* alignment */) override {
        void* ret;

        if (cudaError_t status = STDEXEC_DBG_ERR(cudaMallocManaged(&ret, bytes)); status != cudaSuccess) {
          throw std::bad_alloc();
        }

        return ret;
      }

      void do_deallocate(void* ptr, size_t /* bytes */, size_t /* alignment */) override {
        STDEXEC_DBG_ERR(cudaFree(ptr));
      }

      bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
      }
        
    private:
      std::pmr::memory_resource* _upstream;
    };

    template <class UnderlyingResource>
      class resource_storage {
        UnderlyingResource underlying_resource_{};
        std::pmr::monotonic_buffer_resource monotonic_resource_{512 * 1024, &underlying_resource_};
        std::pmr::synchronized_pool_resource resource_{&monotonic_resource_};

      public:
        std::pmr::memory_resource* get() {
          return &resource_;
        }
      };
  }

  using STDEXEC_STREAM_DETAIL_NS::stream_scheduler;

  struct stream_context {
    STDEXEC_STREAM_DETAIL_NS::resource_storage<STDEXEC_STREAM_DETAIL_NS::pinned_resource> pinned_resource_{};
    STDEXEC_STREAM_DETAIL_NS::resource_storage<STDEXEC_STREAM_DETAIL_NS::managed_resource> managed_resource_{};
    // STDEXEC_STREAM_DETAIL_NS::resource_storage<STDEXEC_STREAM_DETAIL_NS::gpu_resource> gpu_resource_{};

    static int get_device() {
      int dev_id{};
      cudaGetDevice(&dev_id);
      return dev_id;
    }

    int dev_id_{};
    STDEXEC_STREAM_DETAIL_NS::queue::task_hub_t hub_;

    stream_context() 
      : dev_id_(get_device())
      , hub_(dev_id_, pinned_resource_.get()) {
    }

    stream_scheduler get_scheduler(stream_priority priority = stream_priority::normal) {
      return {STDEXEC_STREAM_DETAIL_NS::context_state_t(
          pinned_resource_.get(), 
          managed_resource_.get(), 
          // gpu_resource_.get(), 
          &hub_, 
          priority)};
    }
  };
}

