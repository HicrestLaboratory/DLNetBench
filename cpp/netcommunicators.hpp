#pragma once

#include <mpi.h>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <map>
#include <set>

// ==========================================
// 1. Helper Functions (String & MPI)
// ==========================================

// Gathers a string from every rank in the communicator
inline char **allgather_strings(int strlen_local, const char *str, MPI_Comm mycomm) {
    int size;
    MPI_Comm_size(mycomm, &size);

    int *lengths = (int *)malloc(sizeof(int) * size);
    MPI_Allgather(&strlen_local, 1, MPI_INT, lengths, 1, MPI_INT, mycomm);

    int *displs = (int *)malloc(sizeof(int) * size);
    int total_len = 0;
    for (int i = 0; i < size; i++) {
        displs[i] = total_len;
        total_len += lengths[i];
    }

    char *aggbuf = (char *)malloc(sizeof(char) * total_len);
    MPI_Allgatherv(str, strlen_local, MPI_CHAR, aggbuf, lengths, displs, MPI_CHAR, mycomm);

    char **strings = (char **)malloc(sizeof(char *) * size);
    for (int i = 0; i < size; i++) {
        strings[i] = (char *)malloc(lengths[i] + 1);
        memcpy(strings[i], aggbuf + displs[i], lengths[i]);
        strings[i][lengths[i]] = '\0';
    }

    free(lengths);
    free(displs);
    free(aggbuf);
    return strings;
}

inline const char* patternStr(const char* pad, int nrep) {
    static char s[20000];
    s[0] = '\0';
    for (int i=0; i<nrep; i++) strcat(s, pad);
    return s;
}

inline const char* emptyStr(int nrep) {
    return patternStr(" ", nrep);
}

inline const char* printLable(const char* lable, int size) {
    static char s[20000];
    int len = strlen(lable);
    int npad = size - len;
    if (npad < 0) npad = 0;
    int sshift = (npad/2 + npad%2);
    int fshift = (npad/2);
    
    s[0] = '\0';
    strcat(s, patternStr(" ", sshift));
    strcat(s, lable);
    strcat(s, patternStr(" ", fshift));
    return s;
}

// ==========================================
// 2. The Graph Data Structure
// ==========================================

struct NetworkGraph {
    int nl1;     
    int shight;  
    int max_nps; 
    int max_ppn; 

    int l1_width;
    int node_width;
    int max_lable_len;

    // gnodes[i] contains list of {NodeName, ProcessCount} for the i-th L1 switch
    std::vector<std::pair<std::string,int>> *gnodes;  
    
    // gswitch[level] contains list of {SwitchName, L1ChildCount}
    std::vector<std::pair<std::string,int>> *gswitch; 

    NetworkGraph() : gnodes(nullptr), gswitch(nullptr) {}

    // The main drawing function
    void netPrint(FILE *fp = stdout) {
        if (gnodes == nullptr) return;

        // Print Switches (Top down)
        for(int level=0; level<shight*3; level++) {
            for (auto e: gswitch[level/3]) {
                int width = std::get<1>(e)*l1_width + 2*(std::get<1>(e)-1);
                if (level%3 != 1) 
                    fprintf(fp, "*%s*", patternStr("-", width));
                else 
                    fprintf(fp, "|%s|", printLable(std::get<0>(e).c_str(), width));
            }
            fprintf(fp, "\n");
        }

        // Print Nodes (Bottom up)
        for (int j=0; j<max_nps*5; j++) {
            for (int i=0; i<nl1; i++) {
                if (i >= nl1) break; 
                auto& vec = gnodes[i];
                
                // If this L1 switch has a node at this index
                if (vec.size() > j/5) {
                    auto node_pair = vec[j/5];
                    std::string name = node_pair.first;
                    int p_count = node_pair.second;

                    if (j%5 == 1) fprintf(fp, " |%s| ", printLable(name.c_str(), node_width));
                    else if (j%5 == 2) fprintf(fp, " |%s%s| ", patternStr(" _ ", p_count), emptyStr(node_width-3*p_count));
                    else if (j%5 == 3) fprintf(fp, " |%s%s| ", patternStr("| |", p_count), emptyStr(node_width-3*p_count));
                    else fprintf(fp, " *%s* ", patternStr("-", node_width));
                } else {
                    fprintf(fp, "  %s  ", patternStr(" ", node_width));
                }
            }
            fprintf(fp, "\n");
        }
    }
};

// ==========================================
// 3. Main Logic: Build & Print
// ==========================================

inline void print_topology_graph(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 1. Get Topology Address
    const char* env_addr = std::getenv("SLURM_TOPOLOGY_ADDR");
    std::string my_addr_str;

    if (env_addr) {
        my_addr_str = std::string(env_addr);
    } else {
        // Fallback for non-SLURM testing
        if (rank == 0) fprintf(stderr, "Warning: SLURM_TOPOLOGY_ADDR not set. Using dummy data.\n");
        my_addr_str = "SwitchRoot.SwitchL1-" + std::to_string(rank/4) + ".Node" + std::to_string(rank/2);
    }

    // 2. Gather all strings to Rank 0
    char** all_addrs = allgather_strings(my_addr_str.length(), my_addr_str.c_str(), comm);

    if (rank != 0) {
        // Cleanup on workers
        for(int i=0; i<size; i++) free(all_addrs[i]);
        free(all_addrs);
        return; 
    }

    // --- RANK 0 PROCESSING STARTS HERE ---

    // 3. Parse Paths
    std::vector<std::vector<std::string>> paths;
    int max_depth = 0;
    int max_len = 0;

    for(int i=0; i<size; i++) {
        std::vector<std::string> path;
        std::stringstream ss(all_addrs[i]);
        std::string segment;
        while(std::getline(ss, segment, '.')) {
            path.push_back(segment);
            if(segment.length() > (size_t)max_len) max_len = (int)segment.length();
        }
        if(path.size() > (size_t)max_depth) max_depth = (int)path.size();
        paths.push_back(path);
    }
    
    // Cleanup raw strings
    for(int i=0; i<size; i++) free(all_addrs[i]);
    free(all_addrs);

    if (max_depth < 2) {
        fprintf(stderr, "Error: Topology depth too shallow to graph.\n");
        return;
    }

    // 4. Organize Data for the Graph
    // The graph structure expects us to know:
    // - Unique L1 switches (the parents of nodes)
    // - Unique Higher switches
    // - Which nodes belong to which L1

    // A. Identify all Unique L1 Switches (The second to last element in path)
    std::set<std::string> l1_names;
    std::vector<std::string> l1_ordered;
    
    for(const auto& p : paths) {
        std::string l1 = p[max_depth-2];
        if(l1_names.find(l1) == l1_names.end()) {
            l1_names.insert(l1);
            l1_ordered.push_back(l1);
        }
    }

    // B. Build Nodes Structure: gnodes[l1_index] -> list of {NodeName, Count}
    NetworkGraph graph;
    graph.nl1 = l1_ordered.size();
    graph.shight = max_depth - 1; // Everything except the node itself is a switch level
    graph.gnodes = new std::vector<std::pair<std::string, int>>[graph.nl1];

    int max_ppn = 0;
    int max_nps = 0;

    for(int i=0; i<graph.nl1; i++) {
        std::string current_l1 = l1_ordered[i];
        std::map<std::string, int> node_counts;
        
        // Count processes per node under this L1
        for(const auto& p : paths) {
            if(p[max_depth-2] == current_l1) {
                node_counts[p[max_depth-1]]++;
            }
        }

        for(auto const& [node, count] : node_counts) {
            graph.gnodes[i].push_back({node, count});
            if(count > max_ppn) max_ppn = count;
        }
        if((int)graph.gnodes[i].size() > max_nps) max_nps = (int)graph.gnodes[i].size();
    }

    // C. Build Switches Structure: gswitch[level] -> list of {SwitchName, L1_Children_Count}
    // Note: Level 0 is Root, Level N is L1.
    graph.gswitch = new std::vector<std::pair<std::string, int>>[graph.shight];

    for(int lvl=0; lvl < graph.shight; lvl++) {
        std::vector<std::string> seen_switches;
        
        // We iterate through the ORDERED L1 switches to maintain left-to-right drawing consistency
        // For each L1 switch, we trace its parent at 'lvl'
        for(const auto& l1 : l1_ordered) {
            // Find a path that belongs to this L1 to get its parents
            std::vector<std::string> representative_path;
            for(const auto& p : paths) {
                if(p[max_depth-2] == l1) { representative_path = p; break; }
            }

            std::string sw_name = representative_path[lvl];
            
            // If we haven't added this switch to the graph yet
            bool found = false;
            for(auto& pair : graph.gswitch[lvl]) {
                if(pair.first == sw_name) {
                    pair.second++; // Increment L1 children count (width)
                    found = true;
                    break;
                }
            }
            if(!found) {
                graph.gswitch[lvl].push_back({sw_name, 1});
            }
        }
    }

    // 5. Printing Configuration
    graph.max_nps = max_nps;
    graph.max_ppn = max_ppn;
    graph.max_lable_len = max_len;
    graph.node_width = (3*max_ppn > max_len) ? 3*max_ppn : max_len;
    graph.l1_width = graph.node_width + 2;

    // 6. Draw
    printf("\n=== Topology Graph ===\n");
    graph.netPrint(stdout);
    printf("======================\n");

    // Cleanup
    delete[] graph.gnodes;
    delete[] graph.gswitch;
}