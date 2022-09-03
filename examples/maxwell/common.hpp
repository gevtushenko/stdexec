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

#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string_view>
#include <vector>

#include <string.h>
#include <math.h>

enum class field_id : int { er, hr, mh, hx, hy, ez, dz, fields_count };

struct fields_accessor {
  float dx;
  float dy;

  float width;
  float height;

  std::size_t n;
  std::size_t cells;

  std::size_t begin;
  std::size_t end;
  float *base_ptr;

  [[nodiscard]] std::size_t n_own_cells() const { return end - begin; }

  [[nodiscard]] float *get(field_id id) const {
    return base_ptr + static_cast<int>(id) * (n_own_cells() + 2 * n) + n;
  }
};

struct grid_t {
  float width = 160;
  float height = 160;

  std::size_t n{};
  std::size_t cells{};

  std::size_t begin_{};
  std::size_t end_{};
  std::size_t n_own_cells_{};
  std::unique_ptr<float> fields_{};

  grid_t(grid_t &&) = delete;
  grid_t(const grid_t &) = delete;

  explicit grid_t(std::size_t n, std::size_t begin, std::size_t end)
      : n(n),
        cells(n * n),
        begin_{begin},
        end_{end},
        n_own_cells_{end - begin},
        fields_(new float[static_cast<std::size_t>(n_own_cells_ + n * 2) *
                          static_cast<int>(field_id::fields_count)]) {}

  [[nodiscard]] fields_accessor accessor() const {
    return {height / n, width / n, width, height,       n,
            cells,      begin_,    end_,  fields_.get()};
  }
};

constexpr float C0 = 299792458.0f;  // Speed of light [metres per second]

bool is_circle_part(float x, float y, float object_x, float object_y,
                    float object_size) {
  return ((x - object_x) * (x - object_x) + (y - object_y) * (y - object_y) <=
          object_size * object_size);
}

float calculate_dt(float dx, float dy) {
  const float cfl = 0.3;
  return cfl * std::min(dx, dy) / C0;
}

struct grid_initializer_t {
  float dt;
  fields_accessor accessor;

  void operator()(std::size_t cell_id) const {
    const std::size_t row = cell_id / accessor.n;
    const std::size_t column = cell_id % accessor.n;
    cell_id -= accessor.begin;

    float er = 1.0f;
    float hr = 1.0f;

    const float x = static_cast<float>(column) * accessor.dx;
    const float y = static_cast<float>(row) * accessor.dy;

    const float soil_y = accessor.width / 2.2;
    const float object_y = soil_y - 22.0;
    const float object_size = 3.0;
    const float soil_er_hr = 1.3;

    if (y < soil_y) {
      const float middle_x = accessor.width / 2;
      const float object_x = middle_x;

      if (is_circle_part(x, y, object_x, object_y, object_size)) {
        er = hr = 200000;  /// Relative permeabuliti of Iron
      } else {
        er = hr = soil_er_hr;
      }
    }

    accessor.get(field_id::er)[cell_id] = er;
    accessor.get(field_id::hr)[cell_id] = hr;

    accessor.get(field_id::hx)[cell_id] = {};
    accessor.get(field_id::hy)[cell_id] = {};

    accessor.get(field_id::ez)[cell_id] = {};
    accessor.get(field_id::dz)[cell_id] = {};

    accessor.get(field_id::mh)[cell_id] = C0 * dt / hr;
  }
};

grid_initializer_t grid_initializer(float dt, fields_accessor accessor) {
  return {dt, accessor};
}

std::size_t right_nid(std::size_t cell_id, std::size_t col, std::size_t N) {
  return col == N - 1 ? cell_id - (N - 1) : cell_id + 1;
}

std::size_t left_nid(std::size_t cell_id, std::size_t col, std::size_t N) {
  return col == 0 ? cell_id + N - 1 : cell_id - 1;
}

struct h_field_calculator_t {
  fields_accessor accessor;

  // read: ez, mh, hx, hy
  // write: hx, hy
  // total: 6
  void operator()(std::size_t cell_id) const __attribute__((always_inline)) {
    const std::size_t N = accessor.n;
    const std::size_t column = cell_id % N;
    cell_id -= accessor.begin;

    const float *ez = accessor.get(field_id::ez);
    const float cell_ez = ez[cell_id];
    const float neighbour_ex = ez[cell_id + N];
    const float neighbour_ez = ez[right_nid(cell_id, column, N)];
    const float mh = accessor.get(field_id::mh)[cell_id];

    const float cex = (neighbour_ex - cell_ez) / accessor.dy;
    const float cey = (cell_ez - neighbour_ez) / accessor.dx;
    accessor.get(field_id::hx)[cell_id] -= mh * cex;
    accessor.get(field_id::hy)[cell_id] -= mh * cey;
  }
};

h_field_calculator_t update_h(fields_accessor accessor) { return {accessor}; }

struct e_field_calculator_t {
  float dt;
  float *time;
  fields_accessor accessor;
  std::size_t source_position;

  [[nodiscard]] float gaussian_pulse(float t, float t_0, float tau) const {
    return exp(-(((t - t_0) / tau) * (t - t_0) / tau));
  }

  [[nodiscard]] float calculate_source(float t, float frequency) const {
    const float tau = 0.5f / frequency;
    const float t_0 = 6.0f * tau;
    return gaussian_pulse(t, t_0, tau);
  }

  // reads: hx, hy, dz, er: 4
  // writes: dz, ez: 2
  // total: 6
  //
  // read: 7; write: 2; 9 memory accesses
  void operator()(std::size_t cell_id) const __attribute__((always_inline)) {
    const std::size_t N = accessor.n;
    const std::size_t column = cell_id % N;
    bool source_owner = cell_id == source_position;
    cell_id -= accessor.begin;

    float er = accessor.get(field_id::er)[cell_id];
    float cell_dz = accessor.get(field_id::dz)[cell_id];
    float *hx = accessor.get(field_id::hx);
    float *hy = accessor.get(field_id::hy);

    cell_dz += C0 * dt *
               ((hy[cell_id] - hy[left_nid(cell_id, column, N)]) / accessor.dx +
                ((hx - N)[cell_id] - hx[cell_id]) / accessor.dy);

    if (source_owner) {
      cell_dz += calculate_source(*time, 5E+7);
      *time += dt;
    }

    // read 2 values, write 1 value
    accessor.get(field_id::ez)[cell_id] = cell_dz / er;
    accessor.get(field_id::dz)[cell_id] = cell_dz;
  }
};

e_field_calculator_t update_e(float *time, float dt, fields_accessor accessor) {
  std::size_t source_position =
      accessor.n / 2 + (accessor.n * (accessor.n / 2));

  return {dt, time, accessor, source_position};
}

class result_dumper_t {
  bool write_results_{};
  std::size_t rank_{};
  std::size_t &report_step_;
  fields_accessor accessor_;

  void write_vtk(const std::string &filename) const {
    if (!write_results_) {
      return;
    }

    std::unique_ptr<float[]> h_ez;
    float *ez = accessor_.get(field_id::ez);

    if (rank_ == 0) {
      printf("\twriting report #%d", (int)report_step_);
      fflush(stdout);
    }

    FILE *f = fopen(filename.c_str(), "w");

    const std::size_t nx = accessor_.n;
    const float dx = accessor_.dx;
    const float dy = accessor_.dy;

    const std::size_t own_cells = accessor_.n_own_cells();

    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f, "vtk output\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(f, "POINTS %d double\n", (int)(own_cells * 4));

    const float y_offset = 0.0f;
    for (std::size_t own_cell_id = 0; own_cell_id < own_cells; own_cell_id++) {
      const std::size_t cell_id = own_cell_id + accessor_.begin;
      const std::size_t i = cell_id % nx;
      const std::size_t j = cell_id / nx;

      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 0),
              dy * static_cast<float>(j + 0) - y_offset);
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 1),
              dy * static_cast<float>(j + 0) - y_offset);
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 1),
              dy * static_cast<float>(j + 1) - y_offset);
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 0),
              dy * static_cast<float>(j + 1) - y_offset);
    }

    fprintf(f, "CELLS %d %d\n", (int)own_cells, (int)own_cells * 5);

    for (std::size_t own_cell_id = 0; own_cell_id < own_cells; own_cell_id++) {
      const std::size_t point_offset = own_cell_id * 4;
      fprintf(f, "4 %d %d %d %d\n", (int)(point_offset + 0),
              (int)(point_offset + 1), (int)(point_offset + 2),
              (int)(point_offset + 3));
    }

    fprintf(f, "CELL_TYPES %d\n", (int)own_cells);

    for (std::size_t own_cell_id = 0; own_cell_id < own_cells; own_cell_id++) {
      fprintf(f, "9\n");
    }

    fprintf(f, "CELL_DATA %d\n", (int)own_cells);
    fprintf(f, "SCALARS Ez double 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");

    for (std::size_t own_cell_id = 0; own_cell_id < own_cells; own_cell_id++) {
      fprintf(f, "%lf\n", ez[own_cell_id]);
    }

    fclose(f);

    if (rank_ == 0) {
      printf(".\n");
      fflush(stdout);
    }
  }

 public:
  result_dumper_t(bool write_results, int rank, std::size_t &report_step,
                  fields_accessor accessor)
      : write_results_(write_results),
        rank_(rank),
        report_step_(report_step),
        accessor_(accessor) {}

  void operator()() const {
    const std::string filename = std::string("output_") +
                                 std::to_string(rank_) + "_" +
                                 std::to_string(report_step_++) + ".vtk";

    write_vtk(filename);
  }
};

result_dumper_t dump_vtk(bool write_results, int rank, std::size_t &report_step,
                         fields_accessor accessor) {
  return {write_results, rank, report_step, accessor};
}

class time_storage_t {
  std::unique_ptr<float> time_{};

 public:
  time_storage_t() : time_(new float) {}

  [[nodiscard]] float *get() const { return time_.get(); }
};

std::string bin_name(int node_id) {
  return "out_" + std::to_string(node_id) + ".bin";
}

void copy_to_host(void *to, const void *from, std::size_t bytes) {
  memcpy(to, from, bytes);
}

void validate_results(int node_id, int n_nodes, fields_accessor accessor) {
  std::ifstream meta("meta.txt");

  std::size_t meta_cells{};
  meta >> meta_cells;

  if (meta_cells != accessor.cells && node_id == 0) {
    std::cerr << "Grid sizes should match. Validation is impossible."
              << std::endl;
  } else {
    int meta_n_nodes{};
    meta >> meta_n_nodes;

    const std::size_t buffer_size = 1024 * 1024;
    std::unique_ptr<char[]> ref_buffer(new char[buffer_size]);
    std::unique_ptr<char[]> loc_buffer(new char[buffer_size]);

    bool correct_result = true;

    const char *local_ez =
        reinterpret_cast<const char *>(accessor.get(field_id::ez));

    for (int nid = 0; nid < meta_n_nodes; nid++) {
      std::ifstream bin(bin_name(nid), std::ios::binary);
      std::size_t nid_begin{};
      std::size_t nid_end{};

      bin.read(reinterpret_cast<char *>(&nid_begin), sizeof(std::size_t));
      bin.read(reinterpret_cast<char *>(&nid_end), sizeof(std::size_t));

      const bool in_range =
          accessor.begin < nid_end && accessor.end > nid_begin;

      if (in_range) {
        const std::size_t overlap_begin = std::max(nid_begin, accessor.begin);
        const std::size_t overlap_end = std::min(nid_end, accessor.end);

        std::size_t local_overlap_begin =
            (overlap_begin - accessor.begin) * sizeof(float);
        const std::size_t local_overlap_end =
            (overlap_end - accessor.begin) * sizeof(float);
        const std::size_t reference_overlap_begin =
            (overlap_begin - nid_begin) * sizeof(float);

        bin.seekg(reference_overlap_begin, std::ios_base::cur);

        while (local_overlap_begin < local_overlap_end) {
          const std::size_t bytes_left =
              local_overlap_end - local_overlap_begin;
          const std::size_t bytes = std::min(bytes_left, buffer_size);

          bin.read(ref_buffer.get(), bytes);
          copy_to_host(loc_buffer.get(), local_ez + local_overlap_begin, bytes);

          if (memcmp(loc_buffer.get(), ref_buffer.get(), bytes) != 0) {
            correct_result = false;
            break;
          }

          local_overlap_begin += bytes;
        }
      }
    }

    if (!correct_result) {
      std::cerr << "Invalid result on " << node_id << std::endl;
    }
  }
}

void store_results(int node_id, int n_nodes, fields_accessor accessor) {
  if (node_id == 0) {
    std::ofstream meta("meta.txt");

    meta << accessor.cells << std::endl;
    meta << n_nodes << std::endl;
  }

  std::ofstream bin(bin_name(node_id), std::ios::binary);
  bin.write(reinterpret_cast<const char *>(&accessor.begin),
            sizeof(std::size_t));
  bin.write(reinterpret_cast<const char *>(&accessor.end), sizeof(std::size_t));

  float *ez = accessor.get(field_id::ez);
  std::size_t n_bytes = accessor.n_own_cells() * sizeof(float);

  bin.write(reinterpret_cast<const char *>(ez), n_bytes);
}

void report_header() {
  std::cout << std::fixed << std::showpoint << std::setw(18) << "scheduler"
            << ", " << std::setw(11) << "elapsed [s]"
            << ", " << std::setw(11) << "BW [GB/s]"
            << "\n";
}

template <class ActionT>
void report_performance(std::size_t cells, std::size_t iterations, int node_id,
                        std::string_view scheduler_name, ActionT compute) {
  auto begin = std::chrono::high_resolution_clock::now();
  compute();
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration<double>(end - begin).count();

  // Assume perfect locality
  const std::size_t memory_accesses_per_cell = 6 * 2;  // 8 + 9;
  const std::size_t memory_accesses =
      iterations * cells * memory_accesses_per_cell;
  const std::size_t bytes_accessed = memory_accesses * sizeof(float);

  const double bytes_per_second = static_cast<double>(bytes_accessed) / elapsed;
  const double gbytes_per_second = bytes_per_second / 1024 / 1024 / 1024;

  if (node_id == 0) {
    std::cout << std::setw(18) << scheduler_name << ", " << std::setw(11)
              << std::setprecision(3) << elapsed << ", " << std::setw(11)
              << std::setprecision(3) << gbytes_per_second << std::endl;
  }
}

template <class SchedulerT>
requires requires(SchedulerT &&scheduler) { scheduler.bulk_range(42); }
auto bulk_range(std::size_t n, SchedulerT &&scheduler) {
  return scheduler.bulk_range(n);
}

template <class SchedulerT>
auto bulk_range(std::size_t n, SchedulerT &&) {
  return std::make_pair(std::size_t{0}, n);
}

template <class SchedulerT>
requires requires(SchedulerT &&scheduler) { scheduler.node_id(); }
auto node_id_from(SchedulerT &&scheduler) { return scheduler.node_id(); }

template <class SchedulerT>
auto node_id_from(SchedulerT &&) {
  return 0;
}

template <class SchedulerT>
requires requires(SchedulerT &&scheduler) { scheduler.n_nodes(); }
auto n_nodes_from(SchedulerT &&scheduler) { return scheduler.n_nodes(); }

template <class SchedulerT>
auto n_nodes_from(SchedulerT &&) {
  return 1;
}

bool contains(std::string_view str, char c) {
  return str.find(c) != std::string_view::npos;
}

std::pair<std::string_view, std::string_view> split(std::string_view str,
                                                    char by = '=') {
  auto it = str.find(by);
  return std::make_pair(str.substr(0, it),
                        str.substr(it + 1, str.size() - it - 1));
}

[[nodiscard]] std::map<std::string_view, std::size_t> parse_cmd(int argc,
                                                                char *argv[]) {
  std::map<std::string_view, std::size_t> params;
  const std::vector<std::string_view> args(argv + 1, argv + argc);

  for (auto arg : args) {
    if (arg.starts_with("--")) {
      arg = arg.substr(2, arg.size() - 2);
    }

    if (arg.starts_with("-")) {
      arg = arg.substr(1, arg.size() - 1);
    }

    if (contains(arg, '=')) {
      auto [name, value] = split(arg);
      std::from_chars(value.begin(), value.end(), params[name]);
    } else {
      params[arg] = 1;
    }
  }

  return params;
}

[[nodiscard]] std::size_t value(
    const std::map<std::string_view, std::size_t> &params,
    std::string_view name, std::size_t default_value = 0) {
  if (params.count(name)) {
    return params.at(name);
  }
  return default_value;
}

