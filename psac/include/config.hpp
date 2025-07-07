// This file has been added by Manuel Haag, Matthias Schimek, 2025
#pragma once

namespace psac {

struct Configuration {
  double threshold_fast_resolval = 0.1;
};

// singleton instance
inline Configuration& get_config() {
  static Configuration config;
  return config;
}
}  // namespace psac
