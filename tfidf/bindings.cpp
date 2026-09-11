#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for std::vector and std::string to Python lists and strings
#include "tfidf.hpp"

namespace py = pybind11;

// (compiled file, C++ obj)
PYBIND11_MODULE(tfidf_python, m) {
    py::class_<TfidfKeywordExtractor>(m, "TfidfKeywordExtractor")
        // Expose the constructor
        .def(py::init<const std::vector<std::string>&, unsigned int>(),
             py::arg("papers"), py::arg("top_k") = 5)
        // Expose the keyword extraction method
        .def("extract_keywords", &TfidfKeywordExtractor::extract_keywords,
             "Extracts top-K TF-IDF keywords for the loaded papers");
}