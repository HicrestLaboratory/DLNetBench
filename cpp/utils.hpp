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

using json = nlohmann::json;

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
 *
 * The file has this format:
 *   - Forward Flops:<value>
 *   - Backward Flops:<value>
 *   - Model Size:<value>
 *   - Average Forward Time (s):<value>
 *   - Average Backward Time (s):<value>
 *   - Batch Size:<value>
 *
 * Each parsed value is stored in the returned map with keys:
 * "forwardFlops", "backwardFlops", "modelSize", "avgForwardTime", "avgBackwardTime".
 *
 * @param file_name Path to the model statistics file.
 * @return std::map<std::string, float>.
 */
std::map<std::string, uint64_t> get_model_stats(std::string filename){
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file\n";
        return {};
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

    model_stats["forwardFlops"] = forwardFlops;
    model_stats["backwardFlops"] = backwardFlops;
    model_stats["modelSize"] = modelSize;
    model_stats["avgForwardTime"] = avgForwardTime;
    model_stats["avgBackwardTime"] = avgBackwardTime;
    model_stats["batchSize"] = batch_size;

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

/**
* @enum Device
* @brief Enum to specify the device type for a tensor.
*/
enum class Device { CPU, GPU };


//TODO: support GPU tensors

/**
* @class Tensor
* @brief A lightweight wrapper for a contiguous buffer of data that can reside on CPU or GPU.
*
* This class manages memory allocation and deallocation automatically depending on the device.
* It supports both CPU (host) memory using calloc and GPU (device) memory using cudaMalloc.
*
* @tparam T The data type of the tensor elements (e.g., float, double, half).
*/
template<typename T, Device device = Device::CPU>
class Tensor {
public:
    T* data = nullptr;
    uint64_t size = 0;


    /**
    * @brief Constructs a tensor of given size on a specified device.
    *
    * Allocates memory on the CPU using calloc or on the GPU using cudaMalloc.
    *
    * @param size_ Number of elements in the tensor
    * @param dev Device type (CPU by default)
    */
    Tensor(uint64_t size_, Device dev = Device::CPU) : size(size_) {
        if constexpr (device == Device::CPU) data = (T*)calloc(size, sizeof(T));
    }

    /**
    * @brief Destructor that frees the allocated memory depending on the device.
    */
    ~Tensor() {
        if(data) {
            if(device == Device::CPU) free(data);
            // else cudaFree(data);
        }
    }
};



#endif // UTILS_HPP