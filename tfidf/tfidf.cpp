#include "tfidf.hpp"


TfidfKeywordExtractor::TfidfKeywordExtractor(const std::vector<std::string>& papers, unsigned int top_k)
    :papers(std::move(papers)), top_k(top_k){
        this -> build_df_counts();
    };


void TfidfKeywordExtractor::build_df_counts(void){
    this->doc_freq.clear();

    for(const auto& paper : this -> papers){
        std::unordered_set<std::string> unique_words_per_doc;

        std::stringstream stream(paper);
        std::string word;

        while(stream >> word){
            // make every word lowercase to easy compability
            std::transform(word.begin(), word.end(), word.begin(),
                [](unsigned char ch) -> unsigned char {
                    return (ch < 128) ? std::tolower(ch) : ch;
                }
            );
            
            // ignore short words
            if (word.length() > 2) {
                unique_words_per_doc.insert(word);
            }

        }

        // increment global frequency of the word among all docs
        for (const auto& w : unique_words_per_doc) {
            this -> doc_freq[w]++;
        }
        
    }
}


std::vector<std::vector<std::string>> TfidfKeywordExtractor::extract_keywords(void) const{
    std::vector<std::vector<std::string>> all_keywords;
    all_keywords.reserve(this -> papers.size());

    // custom datatype for min heap - (score, word)
    using ScoredWord = std::pair<double, std::string>;

    for(const auto& paper : this -> papers){
        // for every paper we track the freq of words and their tf-idf scores
        std::unordered_map<std::string, int> tf_count;
        std::priority_queue<ScoredWord, std::vector<ScoredWord>, std::greater<ScoredWord>> min_heap;
        
        std::stringstream stream(paper);
        std::string word;

        while(stream >> word){
            std::transform(word.begin(), word.end(), word.begin(),
                [](unsigned char ch) -> unsigned char {
                    return (ch < 128) ? std::tolower(ch) : ch;
                }
            );
            
            if (word.length() > 2) {
                tf_count[word] += 1;
            }

        }

        const double N = static_cast<double>(this->papers.size());

        for(auto& it : tf_count){
            int tf = it.second;     // freq per doc
            int df = this -> doc_freq.at(it.first);     // in how many docs did this word appear

            // idf score shows how rare is the word among all docs
            // if it appears in all docs, idf score = 1 since ln(1) = 0, 0 + 1 = 1
            double idf_score = std::log((1.0 + N) / (1.0 + df)) + 1.0;

            // final score is big if word is rare among docs and frequent in this curr doc
            double score = tf * idf_score;
            
            // delete the top element if it's score is the less than the curr one
            if(min_heap.size() < this -> top_k){
                min_heap.emplace(score, it.first);
            } else if(min_heap.top().first < score){
                min_heap.pop();
                min_heap.emplace(score, it.first);
            }
        }    
        
        std::vector<std::string> doc_keywords;
        doc_keywords.reserve(min_heap.size());

        // iterate through heap and get top_k elements
        while(!min_heap.empty()){
            doc_keywords.push_back(min_heap.top().second);
            min_heap.pop();
        }

        // sort words in descending order of freqs
        std::reverse(doc_keywords.begin(), doc_keywords.end());
        all_keywords.push_back(doc_keywords);
       
    }

    return all_keywords;
}


std::unordered_map<std::string, double> TfidfKeywordExtractor::get_idf_map(void) const{
    std::unordered_map<std::string, double> idf_map;    // {word: idf_score}
    idf_map.reserve(this->doc_freq.size());
    
    // num of papers
    const double N = static_cast<double>(this->papers.size());
    
    for (const auto& it : this->doc_freq) {
        // ratio of total papers to in how many papers word appears
        double idf = std::log((1.0 + N) / (1.0 + it.second)) + 1.0;
        idf_map[it.first] = idf;
    }
    
    return idf_map;
}