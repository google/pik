// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// File utilities for benchmarking and testing, but which are not needed for
// main pik itself.

#ifndef PIK_BENCHMARK_BENCHMARK_FILE_IO_H_
#define PIK_BENCHMARK_BENCHMARK_FILE_IO_H_

#include <string>
#include <vector>

#include "pik/file_io.h"

namespace pik {

// Checks if the file exists, either as file or as directory
bool PathExists(const std::string& fname);

// Checks if the file exists and is a regular file.
bool IsRegularFile(const std::string& fname);

// Checks if the file exists and is a directory.
bool IsDirectory(const std::string& fname);

// Recursively makes dir, or successfully does nothing if it already exists.
Status MakeDir(const std::string& dirname);

// Deletes a single regular file.
Status DeleteFile(const std::string& fname);

// Returns value similar to unix basename, except it returns empty string if
// fname ends in '/'.
std::string FileBaseName(const std::string& fname);
// Returns value similar to unix dirname, except returns up to before the last
// slash if fname ends in '/'.
std::string FileDirName(const std::string& fname);

// Returns the part of the filename starting from the last dot, or empty
// string if there is no dot.
std::string FileExtension(const std::string& fname);


// Can match a single file, or multiple files in a directory (non-recursive)
// using a limited form of glob: must be of the form "dir/*" or "dir/*suffix".
Status MatchFiles(const std::string& pattern, std::vector<std::string>* list);

std::string JoinPath(const std::string& first, const std::string& second);

Status MakeTempFilename(const std::string& directory,
    const std::string& file_prefix, std::string* result);

}  // namespace pik

#endif  // PIK_BENCHMARK_BENCHMARK_FILE_IO_H_
