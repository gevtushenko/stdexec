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

#include <algorithm>
#include <cassert>

template <int N = 1>
  requires (N > 0)
class flags_storage_t {
  int *flags_{};

public:
  class flags_t {
    int *flags_{};

    flags_t(int *flags)
      : flags_(flags) {
    }

  public:
    __device__ void set(int idx = 0) const { 
      assert(idx < N);
      atomicAdd(flags_ + idx, 1); 
    }

    friend flags_storage_t;
  };

  flags_storage_t(const flags_storage_t &) = delete;
  flags_storage_t(flags_storage_t &&) = delete;

  void operator()(const flags_storage_t &) = delete;
  void operator()(flags_storage_t &&) = delete;

  flags_t get() {
    return {flags_}; 
  }

  flags_storage_t() {
    cudaMallocManaged(&flags_, sizeof(int) * N);
    cudaMemset(flags_, 0, sizeof(int) * N);
  }

  ~flags_storage_t() {
    cudaFree(flags_);
    flags_ = nullptr;
  }

  bool is_set_n_times(int n) {
    return std::count(flags_, flags_ + N, n) == N;
  }

  bool all_set_once() {
    return is_set_n_times(1);
  }

  bool all_unset() { 
    return !all_set_once(); 
  }
};

