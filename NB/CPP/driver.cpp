#include <string>
#include <vector>

#include "Helpers.h"
#include "CSR.h"
#include "Recommender.h"


int main()
{
  bool running = true;
  std::string training_file = "train1.txt";
  std::string test_file = "test1.txt";

  std::string out_file = "Output.txt";

  int k_val = 3;
  int n_val = 5;

  std::vector<std::pair<float, float>> results;

  for(int i = 1; i < 6; i++)
  {
    std::string training_file = "TrainingSet/SmallTrainingSet" + std::to_string(i) + ".txt";
    std::string test_file = "TestingSet/SmallTestingSet" + std::to_string(i) + ".txt";
    Recommender my_recommender(training_file, test_file, k_val, n_val, out_file);
    my_recommender.recommendations();

    results.push_back(my_recommender.test_recs_HR());
  }

  for(size_t i = 0; i < results.size(); i++)
  {
    std::cout << "Result: " << (i + 1) << std::endl;
    std::cout << "HitRate: " << results[i].first << std::endl;
    std::cout << "ARHR: " << results[i].second << std::endl;
  }

  return 0;
}
