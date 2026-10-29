// Program entry points and argument parsing, private to the driver.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace basalt::program {
struct Options;
}

namespace basalt::kernels {
struct RopeScaling;
}

namespace basalt::program::cli {

void usage();
int serve_subcommand(int argc, char** argv);
bool parse_i64_list(const char* text, std::vector<int64_t>& out, std::string& error);

// An exit code for help or an invalid argument; otherwise the driver can start the engine.
std::optional<int> parse_options(int argc, char** argv, Options& o, std::string& ppl_base, std::string& ppl_out);

// Resolve defaults after the model's sections and GPU roles are known.
std::optional<int> resolve_model_options(Options& o, bool multi_gpu);

// Supply the server's seed prompt and check the token input, PLE options and KV format.
std::optional<int> prepare_input_options(Options& o);

std::optional<int> validate_kv_streaming(const Options& o);

// Normalize the draft window options; returns whether serving uses one-token windows.
bool normalize_speculation_options(Options& o);

// Check CLI ranges and conflicts before allocating weights; seed the RoPE type for model resolution.
std::optional<int> validate_generation_options(const Options& o, bool multi_gpu, int min_vram_reserve_mib,
                                              basalt::kernels::RopeScaling& rope_cfg);

}  // namespace basalt::program::cli
