#include <gtest/gtest.h>
#include "tfidf.hpp"
#include <vector>
#include <string>


TEST(TfidfExtractorTest, BasicShapeAndExtraction) {
    std::vector<std::string> corpus = {
        "artificial intelligence and machine learning",
        "machine learning in healthcare",
        "artificial intelligence in finance"
    };
    
    // Request top 2 keywords per document
    TfidfKeywordExtractor extractor(corpus, 2);
    auto results = extractor.extract_keywords();
    
    // Should return exactly 3 vectors (one for each doc)
    EXPECT_EQ(results.size(), 3);
    
    // Each vector should have at most 2 keywords
    EXPECT_LE(results[0].size(), 2);
    EXPECT_LE(results[1].size(), 2);
    EXPECT_LE(results[2].size(), 2);
}

TEST(TfidfExtractorTest, IdfPrioritizesRareWords) {
    std::vector<std::string> corpus = {
        "common unique",
        "common other",
        "common another"
    };
    
    // 'common' appears in 3/3 docs. 'unique' appears in 1/3 docs.
    // Even though local TF is the same (1), IDF will give 'unique' a higher score.
    TfidfKeywordExtractor extractor(corpus, 1);
    auto results = extractor.extract_keywords();
    
    // The top keyword for Doc 1 MUST be "unique", not "common"
    EXPECT_EQ(results[0].size(), 1);
    EXPECT_EQ(results[0][0], "unique");
}

TEST(TfidfExtractorTest, NormalizesAndAggregatesCapitalization) {
    std::vector<std::string> corpus = {"ApPlE aPplE APPLE banana"};
    
    TfidfKeywordExtractor extractor(corpus, 5);
    auto results = extractor.extract_keywords();
    
    // "apple" should be aggregated into a single count, making TF = 3
    // It should beat "banana" (TF = 1)
    EXPECT_EQ(results[0].size(), 2);
    EXPECT_EQ(results[0][0], "apple"); 
    EXPECT_EQ(results[0][1], "banana");
}


TEST(TfidfExtractorTest, IgnoresShortWords) {
    // Words length <= 2 ("a", "to", "is", "it") should be skipped
    std::vector<std::string> corpus = {"a to is it cat dog"};
    
    TfidfKeywordExtractor extractor(corpus, 5);
    auto results = extractor.extract_keywords();
    
    EXPECT_EQ(results[0].size(), 2);
    // Neither "a" nor "is" should be in the results
    EXPECT_NE(results[0][0], "is");
}

TEST(TfidfExtractorTest, HandlesFewerWordsThanTopK) {
    std::vector<std::string> corpus = {"apple"};
    
    // We ask for top 10, but the document only has 1 valid word
    TfidfKeywordExtractor extractor(corpus, 10);
    auto results = extractor.extract_keywords();
    
    EXPECT_EQ(results[0].size(), 1);
    EXPECT_EQ(results[0][0], "apple");
}

TEST(TfidfExtractorTest, HandlesEmptyCorpus) {
    std::vector<std::string> corpus = {};
    
    TfidfKeywordExtractor extractor(corpus, 5);
    auto results = extractor.extract_keywords();
    
    EXPECT_TRUE(results.empty());
}

TEST(TfidfExtractorTest, HandlesEmptyDocuments) {
    // Corpus with an empty string, a whitespace string, and a valid string
    std::vector<std::string> corpus = {"", "    ", "apple banana"};
    
    TfidfKeywordExtractor extractor(corpus, 2);
    auto results = extractor.extract_keywords();
    
    EXPECT_EQ(results.size(), 3);
    EXPECT_TRUE(results[0].empty()); // "" -> no keywords
    EXPECT_TRUE(results[1].empty()); // "   " -> no keywords
    EXPECT_EQ(results[2].size(), 2); // "apple banana" -> 2 keywords
}

// --- IDF Map Tests ---

TEST(TfidfExtractorTest, IdfMapContainsCorrectVocabulary) {
    std::vector<std::string> corpus = {"apple banana", "apple orange"};
    
    TfidfKeywordExtractor extractor(corpus, 5);
    auto idf_map = extractor.get_idf_map();
    
    // Total unique words > 2 chars: "apple", "banana", "orange"
    EXPECT_EQ(idf_map.size(), 3);
    EXPECT_TRUE(idf_map.find("apple") != idf_map.end());
    EXPECT_TRUE(idf_map.find("banana") != idf_map.end());
    EXPECT_TRUE(idf_map.find("orange") != idf_map.end());
}

TEST(TfidfExtractorTest, IdfMapCalculatesCorrectScores) {
    std::vector<std::string> corpus = {
        "common rare",
        "common unique",
        "common another"
    };
    
    TfidfKeywordExtractor extractor(corpus, 5);
    auto idf_map = extractor.get_idf_map();
    
    // N = 3 documents
    // "common" appears in 3 docs. DF = 3. 
    // IDF = ln((1 + 3) / (1 + 3)) + 1.0 = ln(1) + 1.0 = 1.0
    EXPECT_DOUBLE_EQ(idf_map["common"], 1.0);
    
    // "rare" appears in 1 doc. DF = 1.
    // IDF = ln((1 + 3) / (1 + 1)) + 1.0 = ln(4 / 2) + 1.0 = ln(2) + 1.0
    double expected_rare_idf = std::log(4.0 / 2.0) + 1.0;
    EXPECT_DOUBLE_EQ(idf_map["rare"], expected_rare_idf);
}