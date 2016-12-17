#ifndef _RECOMMENDER_H_
#define _RECOMMENDER_H_

#include <cmath> //for sqrt
#include <tuple> //for k_list
#include <map>
#include <algorithm> //for std::find(item in vector)
#include "Helpers.h"

//access the float by using a kvPair
typedef std::map<std::pair<int, int>, float> TupleMap;
typedef TupleMap::iterator It;
typedef std::vector<std::tuple<float, int, std::vector<int>>> KList;

class Recommender
{
public:
  Recommender(std::string training_file, std::string test_file, int k_val, int n_val, std::string out_file);
  ~Recommender(void);

  void recommendations(void);
  void rank_k_vals(std::vector<int> & items_array, KList & k_list);
  void change_n_value(int new_n);
  void change_k_value(int new_k);
  void change_out_file(std::string new_file_name);
  std::pair<float, float> test_recs_HR(void);

  void add_cold_start_users(std::string cold_start_file);
  std::vector<std::vector<int>> get_recs_of_cold_start_users(void);

private:
  std::vector<int> get_recs_for_user(int user_id, int count);
  void build_nk_array(void);
  void rebuild_nk_array(void);
  void build_wilson_score_intervals(void);
  KList pull_k_top_values(int user_id);

  //Sorting related
  void quick_sort(KList & k_list);
  void partition(KList & k_list, int left, int right);
  float pivot(KList & k_list, int left, int right);
  void swap(KList & k_list, int a, int b);
  void insertion_sort(KList & k_list, int left, int right);

  //member variables
  int n_value;
  int k_value;
  std::string out_file;
  CSR* training_data;
  CSR* training_transpose;
  CSR* test_data;

  bool nk_built;
  bool k_changed;

  //cashe/dictionary
  TupleMap cosine_dict;

  //for build/rebuild_nk_array functions
  std::vector<int> length_array;
  std::vector<std::vector<std::pair<float, int>>> nk_array;

  //for wilson score lower bound
  std::vector<double> wslb;

  std::vector<KList> user_recs;

  //for management
  float time_spent_sorting;
  float time_spent_ranking;
};

#include "Recommender.cpp"

#endif //_RECOMMENDER_H_ is defined
