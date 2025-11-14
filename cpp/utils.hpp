/*********************************************************************
 *
 * Description: 
 * Author: Jacopo Raffi
 *
 *********************************************************************/


#ifndef UTILS_HPP
#define UTILS_HPP

#include <assert.h>
#include <string>
#include <fstream>
#include <iostream>
#include <map>


/**
 * @brief Reads model statistics from a stats file and returns them in a map
 *
 * The file has this format:
 *   - Forward Flops:<value>
 *   - Backward Flops:<value>
 *   - Model Size (Bytes):<value>
 *   - Average Forward Time (s):<value>
 *   - Average Backward Time (s):<value>
 *
 * Each parsed value is stored in the returned map with keys:
 * "forwardFlops", "backwardFlops", "modelSize", "avgForwardTime", "avgBackwardTime".
 *
 * @param file_name Path to the model statistics file.
 * @return std::map<std::string, float>.
 */
std::map<std::string, float> get_model_stats(std::string file_name){
    std::ifstream file(file_name);

    if (!file.is_open()) {
        std::cerr << "Failed to open file\n";
        return {};
    }

    std::map<std::string, float> model_stats;
    std::string line;

    // Forward Flops
    std::getline(file, line);
    float forwardFlops = std::stoll(line.substr(line.find(':') + 1));

    // Backward Flops
    std::getline(file, line);
    float backwardFlops = std::stoll(line.substr(line.find(':') + 1));

    // Model Size
    std::getline(file, line);
    float modelSize = std::stoll(line.substr(line.find(':') + 1));

    // Average Forward Time
    std::getline(file, line);
    float avgForwardTime = std::stod(line.substr(line.find(':') + 1));

    // Average Backward Time
    std::getline(file, line);
    float avgBackwardTime = std::stod(line.substr(line.find(':') + 1));

    model_stats["forwardFlops"] = forwardFlops;
    model_stats["backwardFlops"] = backwardFlops;
    model_stats["modelSize"] = modelSize;
    model_stats["avgForwardTime"] = avgForwardTime;
    model_stats["avgBackwardTime"] = avgBackwardTime;

    return model_stats;   
}


#endif // UTILS_HPP