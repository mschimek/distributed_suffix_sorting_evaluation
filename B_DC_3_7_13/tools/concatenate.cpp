#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib> // for std::stoul
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] 
                  << " <input_file> <n_bytes> <repetitions> <output_file>\n";
        return 1;
    }

    // Parse command-line arguments
    std::string inputFile = argv[1];
    size_t n = std::stoul(argv[2]);      // Number of bytes to read
    size_t r = std::stoul(argv[3]);      // Number of repetitions
    std::string outputFile = argv[4];

    // Open the input file in binary mode
    std::ifstream in(inputFile, std::ios::binary);
    if (!in) {
        std::cerr << "Error opening input file: " << inputFile << "\n";
        return 1;
    }

    // Read n bytes from the input file into a buffer
    std::vector<char> buffer(n);
    in.read(buffer.data(), n);
    std::streamsize bytesRead = in.gcount();
    if (bytesRead < static_cast<std::streamsize>(n)) {
        std::cerr << "Warning: Only " << bytesRead 
                  << " bytes were read from " << inputFile << "\n";
        buffer.resize(bytesRead); // Adjust buffer size if fewer bytes were read
    }
    in.close();

    // Open the output file in binary mode
    std::ofstream out(outputFile, std::ios::binary);
    if (!out) {
        std::cerr << "Error opening output file: " << outputFile << "\n";
        return 1;
    }

    // Write the buffer r times to the output file
    for (size_t i = 0; i < r; ++i) {
        out.write(buffer.data(), buffer.size());
    }
    out.close();

    std::cout << "Successfully wrote " << r << " repetitions of " 
              << buffer.size() << " bytes to " << outputFile << "\n";

    return 0;
}
