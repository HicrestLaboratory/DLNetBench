/*********************************************************************
 *
 * Description: 
 * Author: Jacopo Raffi
 *
 *********************************************************************/

#ifndef UTILS_HPP
#define UTILS_HPP

#include <assert.h>
#include <cstdint>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <cerrno>
#include <filesystem>
#include <cstdlib>

#ifdef PROXY_CUDA   
    #include <cuda_runtime.h>
    #include <ccutils/cuda/cuda_macros.hpp>
#endif

#ifdef PROXY_HIP
    #include "tmp_hip_ccutils.hpp" 
    #include <hip/hip_runtime.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

/**
 * Get the base path of the DNNProxy folder.
 * 
 * @param argc Number of command-line arguments.
 * @param argv Command-line arguments.
 * @param rank MPI rank (for error messages).
 * @return The fs::path to the DNNProxy folder.
 * @throws std::runtime_error if the folder does not exist or HOME is not set.
 */
fs::path get_dnnproxy_base_path(int argc, char* argv[], int rank) {
    fs::path base_path;

    // If the user provided a base path as the last argument, prepend $HOME
    if (argc > 1) {
        const char* home = std::getenv("HOME");
        if (!home) {
            if (rank == 0)
                std::cerr << "Error: HOME environment variable not set.\n";
            throw std::runtime_error("HOME not set");
        }
        // Prepend home to the user-provided relative path
        base_path = fs::path(home) / argv[argc - 1];
    } else {
        // Default fallback if no argument is provided
        const char* home = std::getenv("HOME");
        if (!home) {
            if (rank == 0)
                std::cerr << "Error: HOME environment variable not set and no base path provided.\n";
            throw std::runtime_error("HOME not set and no base path provided");
        }
        base_path = fs::path(home) / "DNNProxy";
    }

    if (!fs::exists(base_path) || !fs::is_directory(base_path)) {
        if (rank == 0)
            std::cerr << "Error: DNNProxy folder does not exist at: " << base_path << "\n";
        throw std::runtime_error("DNNProxy folder not found");
    }

    return base_path;
}


/**
 * @brief Computes the average and standard deviation of a list of message sizes.
 *
 * @param sizes A vector containing the sizes of messages.
 * @param sharding_multiplier An optional multiplier to scale each size (default is 1).
 * @return A pair where the first element is the average size and the second is the standard deviation.
 */
std::pair<float, float> compute_msg_stats(const std::vector<uint64_t>& sizes, uint sharding_multiplier = 1) {
    float avg = 0.0f;
    for (uint64_t s : sizes)
        avg += s * sharding_multiplier;
    avg /= sizes.size();

    float stddev = 0.0f;
    for (uint64_t s : sizes) {
        float diff = s * sharding_multiplier - avg;
        stddev += diff * diff;
    }
    stddev = std::sqrt(stddev / sizes.size());

    return {avg, stddev};
}

/**
 * @brief Extracts the value part from a line formatted as "key: value".
 *
 * @param line The input line containing a key-value pair.
 * @return The extracted value as a string, trimmed of whitespace.
 */
std::string extract_value(const std::string &line) {
    size_t pos = line.find(':');
    if (pos == std::string::npos)
        return ""; // no delimiter → empty string

    std::string value = line.substr(pos + 1);

    // trim whitespace
    value.erase(0, value.find_first_not_of(" \t\r\n"));
    value.erase(value.find_last_not_of(" \t\r\n") + 1);

    return value;
}

/**
 * @brief Reads model statistics from a stats file and returns them in a map
 *TODO: change thi func and use a JSON instead
 * The file has this format:
 *   - Forward Flops:<value>
 *   - Backward Flops:<value>
 *   - Model Size:<value>
 *   - Average Forward Time (s):<value>
 *   - Average Backward Time (s):<value>
 *   - Batch Size:<value>
 *   - FFN_Average_Forward_Time (us):15125
 *   - FFN_Average_Backward_Time (us):24139
 *   - Experts: 4
 * Each parsed value is stored in the returned map with keys:
 * "forwardFlops", "backwardFlops", "modelSize", "avgForwardTime", "avgBackwardTime", "batchSize", "ffn_avgForwardTime", "ffn_avgBackwardTime", "experts".
 *
 * @param file_name Path to the model statistics file.
 * @return std::map<std::string, float>.
 */
std::map<std::string, uint64_t> get_model_stats(std::string filename){
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << std::strerror(errno) << "\n";
        throw std::runtime_error("Could not open model stats file: " + filename);
    }

    std::map<std::string, uint64_t> model_stats;
    std::string line;

    std::getline(file, line);
    uint64_t forwardFlops = std::stoull(extract_value(line));

    // Backward Flops
    std::getline(file, line);
    uint64_t backwardFlops = std::stoull(extract_value(line));

    // Model Size
    std::getline(file, line);
    uint64_t modelSize = std::stoull(extract_value(line));

    // Average Forward Time  (should be double, not uint64_t)
    std::getline(file, line);
    uint64_t avgForwardTime = std::stod(extract_value(line));

    // Average Backward Time (should be double)
    std::getline(file, line);
    uint64_t avgBackwardTime = std::stod(extract_value(line));

    // Batch size
    std::getline(file, line);
    uint64_t batch_size = std::stoull(extract_value(line));

    // FFN Average Forward Time (us)
    std::getline(file, line);
    uint64_t ffn_avgForwardTime = std::stoull(extract_value(line));

    // FFN Average Backward Time (us)
    std::getline(file, line);
    uint64_t ffn_avgBackwardTime = std::stoull(extract_value(line));

    std::getline(file, line); // Experts (optional)
    uint64_t experts = std::stoull(extract_value(line));

    model_stats["forwardFlops"] = forwardFlops;
    model_stats["backwardFlops"] = backwardFlops;
    model_stats["modelSize"] = modelSize;
    model_stats["avgForwardTime"] = avgForwardTime;
    model_stats["avgBackwardTime"] = avgBackwardTime;
    model_stats["batchSize"] = batch_size;
    model_stats["ffn_avgForwardTime"] = ffn_avgForwardTime;
    model_stats["ffn_avgBackwardTime"] = ffn_avgBackwardTime;
    model_stats["experts"] = experts;

    return model_stats;   
}


/**
 * @brief Extracts the number of layers from a model configuration JSON file.
 * The function reads the JSON file and sums up the number of encoder and decoder blocks
 * to determine the total number of layers in the model.
 * @param filename The path to the JSON configuration file.
 * @return The total number of layers in the model.
*/
uint count_layers(std::string filename){
    std::ifstream f(filename);
    json data = json::parse(f);

    uint num_layers = 0;

    if(data.contains("num_encoder_blocks")){
        num_layers += data["num_encoder_blocks"].get<uint>();
    }

    if(data.contains("num_decoder_blocks")){
        num_layers += data["num_decoder_blocks"].get<uint>();
    }

    return num_layers;
}

#endif // UTILS_HPP