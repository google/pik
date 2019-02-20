// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "benchmark/benchmark_file_io.h"

#include <cstdio>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <sstream>

#include "pik/os_specific.h"

namespace pik {

const char kPathSeparator = '/';

// RAII, ensures dir is closed even when returning early.
class DirWrapper {
 public:
  DirWrapper(const DirWrapper& other) = delete;
  DirWrapper& operator=(const DirWrapper& other) = delete;

  explicit DirWrapper(const std::string& pathname)
      : dir_(opendir(pathname.c_str())) {}

  ~DirWrapper() {
    if (dir_ != nullptr) {
      const int err = closedir(dir_);
      PIK_CHECK(err == 0);
    }
  }

  operator DIR*() const { return dir_; }

 private:
  DIR* const dir_;
};

// Checks if the file exists, either as file or as directory
bool PathExists(const std::string& fname) {
  struct stat s;
  if (stat(fname.c_str(), &s) != 0) return false;
  return true;
}

// Checks if the file exists and is a regular file.
bool IsRegularFile(const std::string& fname) {
  struct stat s;
  if (stat(fname.c_str(), &s) != 0) return false;
  return S_ISREG(s.st_mode);
}

// Checks if the file exists and is a directory.
bool IsDirectory(const std::string& fname) {
  struct stat s;
  if (stat(fname.c_str(), &s) != 0) return false;
  return S_ISDIR(s.st_mode);
}

// Recursively makes dir, or successfully does nothing if it already exists.
Status MakeDir(const std::string& dirname) {
  size_t pos = 0;
  for (pos = dirname.size(); pos > 0; pos--) {
    if (pos == dirname.size() || dirname[pos] == kPathSeparator) {
      // Found existing dir or regular file, break and then start creating
      // from here (in the latter case we'll get error below).
      if (PathExists(dirname.substr(0, pos + 1))) {
        pos += 1;  // Skip past this existing path
        break;
      }
    }
  }
  for (; pos <= dirname.size(); pos++) {
    if (pos == dirname.size() || dirname[pos] == kPathSeparator) {
      std::string subdir = dirname.substr(0, pos + 1);
      if (mkdir(subdir.c_str(), 0777) && errno != EEXIST) {
        return PIK_FAILURE("Failed to create directory");
      }
    }
  }
  if (!IsDirectory(dirname)) return PIK_FAILURE("Failed to create directory");
  return true;  // success
}

Status DeleteFile(const std::string& fname) {
  if (!IsRegularFile(fname)) {
    return PIK_FAILURE("Trying to delete non-regular file");
  }
  if (std::remove(fname.c_str())) return PIK_FAILURE("Failed to delete file");
  return true;
}

std::string FileBaseName(const std::string& fname) {
  size_t pos = fname.rfind('/');
  if (pos == std::string::npos) return fname;
  return fname.substr(pos + 1);
}

std::string FileDirName(const std::string& fname) {
  size_t pos = fname.rfind('/');
  if (pos == std::string::npos) return "";
  return fname.substr(0, pos);
}

std::string FileExtension(const std::string& fname) {
  size_t pos = fname.rfind('.');
  if (pos == std::string::npos) return "";
  return fname.substr(pos);
}

std::string JoinPath(const std::string& first, const std::string& second) {
  PIK_CHECK(second.empty() || second[0] != kPathSeparator);
  return (!first.empty() && first.back() == kPathSeparator) ?
      (first + second) : (first + kPathSeparator + second);
}

// Can match a single file, or multiple files in a directory (non-recursive)
// using a limited form of glob: must be of the form "dir/*" or "dir/*suffix".
Status MatchFiles(const std::string& pattern, std::vector<std::string>* list) {
  std::string dirname = FileDirName(pattern);
  std::string basename = FileBaseName(pattern);
  if (!basename.empty() && basename[0] == '*') {
    std::string suffix = basename.substr(1);
    if (suffix.find_first_of("*?[") != std::string::npos ||
        dirname.find_first_of("*?[") != std::string::npos) {
      return PIK_FAILURE("Glob pattern '*' must be the first character of the"
          " filename, e.g. dirname/*.png, other patterns currently not"
          " supported.");
    }
    DirWrapper dir(dirname);
    if (!dir) return PIK_FAILURE("directory doesn't exist");
    for (;;) {
      dirent* ent = readdir(dir);
      if (!ent) break;
      std::string name = ent->d_name;
      // If there was a suffix, only add if it matches (e.g. ".png")
      if (name.size() >= suffix.size() &&
          name.substr(name.size() - suffix.size()) == suffix) {
        std::string path = JoinPath(dirname, name);;
        if (IsRegularFile(path)) {
          list->push_back(path);
        }
      }
    }
    return true;
  }
  // No *, so a single file is intended
  list->push_back(pattern);
  return true;
}

Status MakeTempFilename(const std::string& directory,
    const std::string& file_prefix, std::string* result) {
  std::string tempname = JoinPath(directory, file_prefix) + "XXXXXX";
  std::vector<char> tempchars(tempname.begin(), tempname.end());
  tempchars.push_back(0);
  int fd = mkstemp(tempchars.data());
  if (fd == -1) {
    return PIK_FAILURE("Failed to create temp file");
  }
  close(fd);
  *result = tempchars.data();
  return result;
}

}  // namespace pik
