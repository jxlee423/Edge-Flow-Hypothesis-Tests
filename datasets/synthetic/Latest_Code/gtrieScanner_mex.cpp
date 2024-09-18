#include "mex.h"
#include <cstdlib>
#include <cstdio>
#include <string>
#include <algorithm>

void runGtrieScanner(const char *input_file, const char *output_file_base) {
    char command[1024];
    snprintf(command, sizeof(command), "wsl /home/elijah/gtrieScanner_src_01/gtrieScanner -s 4 -g \"%s\" -u -o \"%s.txt\" -oc \"%s_sub.txt\" -f simple", input_file, output_file_base, output_file_base);
    mexPrintf("Running command: %s\n", command); 
    int status = system(command);
    mexPrintf("Command execution status: %d\n", status); 
}

// Helper function to convert Windows path to WSL path
std::string convertPathToWSL(const std::string &winPath) {
    std::string wslPath = winPath;
    std::replace(wslPath.begin(), wslPath.end(), '\\', '/');
    if (wslPath.length() > 1 && wslPath[1] == ':') {
        wslPath = "/mnt/" + std::string(1, std::tolower(wslPath[0])) + wslPath.substr(2);
    }
    return wslPath;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("gtrieScanner:nrhs", "Two inputs required: input_file and output_file_base.");
    }

    if (!mxIsChar(prhs[0]) || !mxIsChar(prhs[1])) {
        mexErrMsgIdAndTxt("gtrieScanner:notString", "Both inputs must be strings.");
    }

    char *input_file = mxArrayToString(prhs[0]);
    char *output_file_base = mxArrayToString(prhs[1]);

    // Convert Windows paths to WSL paths
    std::string input_file_wsl = convertPathToWSL(input_file);
    std::string output_file_base_wsl = convertPathToWSL(output_file_base);

    runGtrieScanner(input_file_wsl.c_str(), output_file_base_wsl.c_str());

    mxFree(input_file);
    mxFree(output_file_base);

    if (nlhs > 0) {
        plhs[0] = mxCreateDoubleScalar(0);
    }
}
