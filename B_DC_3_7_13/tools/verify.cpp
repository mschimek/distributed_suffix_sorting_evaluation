#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

uint64_t bytesToUint64(const std::vector<unsigned char>& buffer) {
    uint64_t value = 0;
    for (size_t i = 0; i < buffer.size(); ++i) {
        value |= static_cast<uint64_t>(buffer[i]) << (8 * i);
    }
    return value;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <input_file1> <byte_chunk1> <input_file2> <byte_chunk2>\n";
    return 1;
  }

  const char* inputFile1 = argv[1];
  const char* inputFile2 = argv[3];
  size_t byteChunk1 = std::stoul(argv[2]);
  size_t byteChunk2 = std::stoul(argv[4]);

  // Open the binary files.
  std::ifstream file1(inputFile1, std::ios::binary);
  if (!file1) {
    std::cerr << "Error: Cannot open file " << inputFile1 << "\n";
    return 1;
  }
  std::ifstream file2(inputFile2, std::ios::binary);
  if (!file2) {
    std::cerr << "Error: Cannot open file " << inputFile2 << "\n";
    return 1;
  }

  // Loop until both files reach EOF.
  while (true) {
    // Read byteChunk1 bytes from file1.
    std::vector<unsigned char> buffer1(byteChunk1, 0);
    file1.read(reinterpret_cast<char*>(buffer1.data()), byteChunk1);
    std::streamsize bytesRead1 = file1.gcount();

    // Read byteChunk2 bytes from file2.
    std::vector<unsigned char> buffer2(byteChunk2, 0);
    file2.read(reinterpret_cast<char*>(buffer2.data()), byteChunk2);
    std::streamsize bytesRead2 = file2.gcount();

    // If no more bytes are read from both files, exit the loop.
    if (bytesRead1 == 0 && bytesRead2 == 0) break;

    // Resize the buffers if fewer bytes were read.
    if (static_cast<size_t>(bytesRead1) < byteChunk1)
      buffer1.resize(bytesRead1);
    if (static_cast<size_t>(bytesRead2) < byteChunk2)
      buffer2.resize(bytesRead2);

    // Convert the byte buffers into uint64_t values.
    uint64_t value1 = bytesToUint64(buffer1);
    uint64_t value2 = bytesToUint64(buffer2);

    // Compare and output the result.
  //  std::cout << "Value from file1: " << value1
  //            << ", Value from file2: " << value2 << " -> ";
    if (value1 == value2)  {
      //std::cout << "Equal";
    }
    else if (value1 > value2) {
      std::cout << "File1 is greater";
      std::cout << std::endl;
    }
    else {
      std::cout << "File2 is greater";
      std::cout << std::endl;
    }
  }

  return 0;
}
