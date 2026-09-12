#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <sstream>
#include <queue>
#include <algorithm>
#include <cctype>

/**
 * @class TfidfKeywordExtractor
 * @brief Computes TF-IDF scores across a document corpus to extract domain-specific keywords.
 *
 * Utilizes a two-pass architecture:
 * 1. Pass 1: Accumulates corpus-wide document frequencies (DF) across all papers.
 * 2. Pass 2: Computes local term frequencies (TF), calculates smooth TF-IDF weights,
 *    and maintains the top-K terms per document using a bounded min-heap.
 */
class TfidfKeywordExtractor {
private:
    /// Maximum number of keywords to extract per paper.
    unsigned int top_k;

    /// In-memory corpus of paper texts (e.g., concatenated title and abstract).
    std::vector<std::string> papers;

    /// Global document frequency map mapping each term to the number of documents it appears in.
    std::unordered_map<std::string, int> doc_freq;

    /**
     * @brief Builds corpus-level document frequency (DF) statistics across all documents.
     *
     * Iterates over all papers, tokenizes words separated by whitespace, normalizes
     * characters to lowercase, filters tokens with length <= 2, and records each unique
     * term per document to update the global `doc_freq` mapping.
     */
    void build_df_counts(void);

public:
    /**
     * @brief Constructs the extractor and immediately calculates corpus document frequencies.
     * 
     * @param papers A list of document texts (e.g., titles + abstracts) representing the entire corpus.
     * @param top_k The target number of top-ranking keywords to return per document.
     */
    TfidfKeywordExtractor(const std::vector<std::string>& papers, unsigned int top_k);

    /**
     * @brief Computes TF-IDF weights and extracts the top-K keywords for every document in the corpus.
     *
     * Evaluates local word frequencies for each document, calculates smooth IDF weights
     * using the formula ln((1 + N) / (1 + DF)) + 1, maintains top-scoring words via a
     * bounded min-heap, and returns the extracted keywords sorted in descending order of score.
     *
     * @return A 2D matrix of shape [N_papers, top_k] where each inner vector contains
     *         the top keywords for the corresponding paper.
     */
    std::vector<std::vector<std::string>> extract_keywords(void) const;

    /**
     * @brief Computes and returns corpus-wide inverse document frequency (IDF) scores.
     *
     * Evaluates smooth IDF weights for all vocabulary terms identified during corpus
     * ingestion using the formula ln((1 + N) / (1 + DF)) + 1.0.
     *
     * @return A hash map mapping each unique term to its smooth IDF weight.
     */
    std::unordered_map<std::string, double> get_idf_map(void) const;
};