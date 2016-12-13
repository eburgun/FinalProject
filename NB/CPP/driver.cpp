#include <string>

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

  Recommender my_recommender(training_file, test_file, k_val, n_val, out_file);

  my_recommender.recommendations();

  std::pair<float, float> hit_rate = my_recommender.test_recs_HR();
  Helpers::print_line("Hit rate");
  Helpers::print_line(hit_rate.first);

  Helpers::print_line("ARHR");
  Helpers::print_line(hit_rate.second);
  return 0;
}
