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

#include <execution.hpp>
#include <type_traits>

namespace example::cuda::stream {

  struct scheduler_t;

  struct receiver_base_t {};

  struct operation_state_base_t {
    cudaStream_t stream_{0};
  };

  template <class S>
  concept stream_sender = 
      std::execution::sender<S> &&
      std::is_same_v<
          std::tag_invoke_result_t<
            std::execution::get_completion_scheduler_t<std::execution::set_value_t>, S>,
          scheduler_t>;
}

