Recommender::Recommender(std::string training_file, std::string test_file, int k_val, int n_val, std::string out_file)
{
  // clock_t timer = ::Helpers::TimerStart();

  this->n_value = n_val;
  this->k_value = k_val;
  this->out_file = out_file;
  this->training_data = new CSR(training_file);
  this->training_transpose = CSR::transpose(this->training_data);
  this->test_data = new CSR(test_file);
  //cosine_dict is defined in header
  this->nk_built = false;
  this->k_changed = false;

  // ::Helpers::TimerEnd("Setup finished in: ", timer);
}

void Recommender::add_cold_start_users(std::string cold_start_file)
{
  CSR * cold_start_data = new CSR(cold_start_file);
  this->training_data->add_new_users(cold_start_data);
  delete this->training_transpose;
  this->training_transpose = CSR::transpose(this->training_data);
}

std::vector<std::vector<int>> Recommender::get_recs_of_cold_start_users(void)
{
  //need to build the nk_array to account for new users and ratings
  this->build_nk_array();

  std::vector<std::vector<int>> recs;
  for(int i = this->training_data->first_new_user_id; i < this->training_data->nrows; i++)
  {
    //get top 20 recommendations for user with id i
    recs.push_back(this->get_recs_for_user(i, 20));
  }

  return recs;
}

Recommender::~Recommender(void)
{
  delete this->training_data;
  delete this->training_transpose;
  delete this->test_data;
}

void Recommender::recommendations(void)
{
  //reset benchmarking statistics
  this->time_spent_sorting = 0;
  this->time_spent_ranking = 0;

  if(!this->nk_built)
    this->build_nk_array();

  if(this->k_changed)
    this->rebuild_nk_array();

  clock_t timer = ::Helpers::TimerStart();
  for(int i = 0; i < this->training_data->nrows; i++)
  {
    //TODO: print progress
    this->user_recs.push_back(this->pull_k_top_values(i));
  }
  Helpers::TimerEnd("Finished recommendations in: ", timer);
  // std::cout << "Time spent sorting: " << this->time_spent_sorting << std::endl;
  std::cout << "Time spent ranking: " << this->time_spent_ranking << std::endl;
}

//gets the top 'count' recommendations for user with id
//this assumes that you need it for cold start users, hence the building of nk_array
std::vector<int> Recommender::get_recs_for_user(int id, int count)
{

  std::vector<int> recs;
  KList list = this->pull_k_top_values(id);

  for(size_t i = 0; i < list.size() && i < count; i++)
  {
    recs.push_back(std::get<1>(list[i]));
  }

  return recs;
}

void Recommender::build_nk_array(void)
{
  clock_t timer = ::Helpers::TimerStart();

  //setup length array for each item
  //initialize all to 0
  this->length_array = std::vector<int>(this->training_transpose->nrows, 0);

  //fill the length array with the sum of all item's ratings squared
  for(size_t i = 0; i < this->length_array.size(); i++)
  {
    int item_start = this->training_transpose->row_ptr[i];
    int item_finish = this->training_transpose->row_ptr[i + 1];

    while(item_start < item_finish)
    {
      this->length_array[i] += (this->training_transpose->val[item_start] * this->training_transpose->val[item_start]);
      item_start += 1;
    }
  }

  //set up the wilson score interval array
  this->build_wilson_score_intervals();

  //set up cosine array for each item and fill it with 0s
  float cosine_array[this->training_transpose->nrows];
  for(int x = 0; x < this->training_transpose->nrows; x++)
  {
    cosine_array[x] = 0.0;
  }

  //building the nk_array
  //dimentions: item count rows, 2 * k recirds, containing a pair of cosine simil and col of the pair of cosine
  printf("\rAllocating Memory.");
  std::vector<std::pair<float, int>> nk_temp_array_for_person(2 * this->k_value, std::make_pair(0.0, 0));
  this->nk_array = std::vector<std::vector<std::pair<float, int>>>(this->training_transpose->nrows, nk_temp_array_for_person);

  //for each item
  for(int i = 0; i < this->training_transpose->nrows; i++)
  {
    //first user that rated item i
    int user_start = this->training_transpose->row_ptr[i];
    //last user that rated item i
    int user_finish = this->training_transpose->row_ptr[i + 1];
    //for each user that rated item i
    while(user_start < user_finish)
    {
      //first item this user rated
      int col_start = this->training_data->row_ptr[this->training_transpose->col_ind[user_start]];
      //last item this user rated
      int col_finish = this->training_data->row_ptr[this->training_transpose->col_ind[user_start] + 1];
      //for each item of the user that rated item i
      while(col_start < col_finish)
      {
        //if this item isn't item i
        if(this->training_data->col_ind[col_start] != i){
          //cosine numerator of this item += (item's rating * this users rating)
          cosine_array[this->training_data->col_ind[col_start]] += (this->training_data->val[col_start] * this->training_transpose->val[user_start]);
        }
        col_start ++;
      }
      user_start += 1;
    }

    //for each item again
    for(int j = 0; j < this->training_transpose->nrows; j++)
    {
      bool store_cosine = true;
      //if both item i and item j have ratings
      if(this->length_array[i] != 0 && this->length_array[j] != 0)
      {
        //divide the numerator of item j with both items
        cosine_array[j] /= ( std::sqrt(this->length_array[i]) * std::sqrt(this->length_array[j]) );
      }
      else //if one of item i or j doesn't have ratings
      {
        cosine_array[j] = 0;
        store_cosine = false;
      }

      //incorporate wilson score lower bound to deal with cold start
      //add some weight to make sure wslb is accounted for
      int wslb_weight = 1;
      //this way items without ratings can never pass over items with ratings
      cosine_array[j] = (cosine_array[j] + wslb_weight) * this->wslb[j];


      //store into dictionary only if significant with 10 ratings
      bool passes_min_of_wsi = this->wslb[j] > 0.00005; //for at least 10 ratings
      if(store_cosine && passes_min_of_wsi)
      {
        if(i < j)
        {
          //if cosine similarity between item i and j doesn't exist
          if(this->cosine_dict.find(std::make_pair(i, j)) == this->cosine_dict.end() &&
            this->cosine_dict.find(std::make_pair(j, i)) == this->cosine_dict.end())
          {
              //save the cosine similarity betweem item i and j
              this->cosine_dict[std::make_pair(i, j)] = cosine_array[j];
          }
        }
        //do the converse of top branch
        if(j < i)
        {
          if(this->cosine_dict.find(std::make_pair(i, j)) == this->cosine_dict.end() &&
            this->cosine_dict.find(std::make_pair(j, i)) == this->cosine_dict.end())
          {
            this->cosine_dict[std::make_pair(j, i)] = cosine_array[j];
          }
        }
      }

      //insertion sort of k_size items
      int k_size = this->nk_array[i].size();
      if(cosine_array[j] > this->nk_array[i][k_size - 1].first)
      {
        this->nk_array[i][k_size - 1].first = cosine_array[j];
        this->nk_array[i][k_size - 1].second = j;

        int k = k_size - 1;
        while(this->nk_array[i][k].first > this->nk_array[i][k - 1].first && k > 0)
        {
          std::pair<float, int> temp = this->nk_array[i][k - 1];
          this->nk_array[i][k - 1] = this->nk_array[i][k];
          this->nk_array[i][k] = temp;
          k -= 1;
        }
      }
    }

    for(int j = 0; j < this->training_transpose->nrows; j++)
    {
      cosine_array[j] = 0.0;
    }
  }
  //housekeeping
  this->k_changed = false;
  this->nk_built = true;
  Helpers::TimerEnd("\rBuild Time: ", timer);
}

void Recommender::rebuild_nk_array(void)
{
  clock_t timer = ::Helpers::TimerStart();

  std::vector<std::pair<float, int>> nk_temp_array_for_person(2 * this->k_value, std::make_pair(0.0, 0));
  this->nk_array = std::vector<std::vector<std::pair<float, int>>>(this->training_transpose->nrows, nk_temp_array_for_person);

  for(int i = 0; i < this->training_transpose->nrows; i++)
  {
    for(int j = 0; j < this->training_transpose->nrows; j++)
    {
      float cur_simil = 0;
      if(this->length_array[i] != 0 && this->length_array[j] != 0)
      {
        if(i < j)
        {
          cur_simil = this->cosine_dict.find(std::make_pair(i, j))->second;
        }
        else if(j < i)
        {
          cur_simil = this->cosine_dict.find(std::make_pair(j, i))->second;
        }
      }

      //insertion sort of the item similarities
      if(cur_simil > this->nk_array[i][this->nk_array[i].size() - 1].first)
      {
        this->nk_array[i][this->nk_array[i].size() - 1].first = cur_simil;
        this->nk_array[i][this->nk_array[i].size() - 1].second = j;

        int k = this->nk_array[i].size() - 1;
        while(this->nk_array[i][k].first > this->nk_array[i][k - 1].first && k > 0)
        {
          std::pair<float, int> temp = this->nk_array[i][k - 1];
          this->nk_array[i][k - 1] = this->nk_array[i][k];
          this->nk_array[i][k] = temp;
          k -= 1;
        }
      }
    }
  }

  this->k_changed = false;
  Helpers::TimerEnd("Rebuild Time: ", timer);
}

void Recommender::build_wilson_score_intervals(void)
{

  //inspiration https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
  double n = this->training_data->nitems;
  this->wslb.reserve(n);

  for(size_t i = 0; i < this->length_array.size(); i++)
  {
    //using the lower bound of wilson score interval with 95% confidence
    //pessimistically weight which items are most popular
    //the most popular items will be useful for cold start cases
    //assumes normal distribution
    double p = (double)this->length_array[i]; //count of positive ratings
    double z = 1.96; // for 95% ci
    double pPer = p / n; //% of positive ratings

    double center = (pPer + (z * z) / (2 * n));
    double difference = (z * sqrt((pPer * ( 1 - pPer ) + z * z / (4 * n)) / n)) / (1 + z * z / n);
    double wsi = center - difference; //lower bound

    this->wslb.push_back(wsi);
    // if(p > 20)
      // std::cout << "Item: " << (i + 1) << ": " << p << ", " << wsi << std::endl;
  }
}

KList Recommender::pull_k_top_values(int user_id)
{
  int user_item_count = this->training_data->row_ptr[user_id + 1] - this->training_data->row_ptr[user_id];
  std::vector<int> user_items(user_item_count, 0);

  //fill the vector with the user's item's column values
  for(int i = 0; i < user_item_count; i++)
  {
    user_items[i] = this->training_data->col_ind[this->training_data->row_ptr[user_id] + i];
  }

  KList k_list;
  std::vector<int> in_list;

  for(int i = 0; i < user_item_count; i++)
  {
    int j = 0;
    int item_count = 0;
    while(j < this->nk_array.size() && item_count < this->k_value)
    {
      bool is_in_user_items = std::find(user_items.begin(), user_items.end(), this->nk_array[i][j].second) != user_items.end();
      bool is_in_in_list = std::find(in_list.begin(), in_list.end(), this->nk_array[i][j].second) != in_list.end();
      if(!is_in_user_items && !is_in_in_list)
      {
        in_list.push_back(this->nk_array[i][j].second);
        k_list.push_back(std::make_tuple(0.0, this->nk_array[i][j].second, std::vector<int>(1, j)));
        item_count += 1;
      }

      j += 1;
    }
  }

  clock_t timer = Helpers::TimerStart();
  this->rank_k_vals(user_items, k_list);
  this->time_spent_ranking += (clock() - timer) / (float)CLOCKS_PER_SEC;

  timer = Helpers::TimerStart();
  this->quick_sort(k_list);
  this->time_spent_sorting += (clock() - timer) / (float)CLOCKS_PER_SEC;

  //build the list of size n
  KList n_values;
  for(int i = 0; i < this->n_value; i++)
  {
    n_values.push_back(k_list[i]);
  }

  return n_values;
}

void Recommender::rank_k_vals(std::vector<int> & items_array, KList & k_list)
{
  for(size_t i = 0; i < k_list.size(); i++)
  {
    for(size_t j = 0; j < items_array.size(); j++)
    {
      if(std::get<1>(k_list[i]) < items_array[j])
      {
        std::get<0>(k_list[i]) += this->cosine_dict.find(std::make_pair(std::get<1>(k_list[i]), items_array[j]))->second;
        std::get<2>(k_list[i]).push_back(items_array[j]);
      }
      else if (items_array[j] < std::get<1>(k_list[i]))
      {
        std::get<0>(k_list[i]) += this->cosine_dict.find(std::make_pair(items_array[j], std::get<1>(k_list[i])))->second;
        std::get<2>(k_list[i]).push_back(items_array[j]);
      }
    }
  }
}

void Recommender::change_n_value(int new_n)
{
  this->n_value = new_n;
}

void Recommender::change_k_value(int new_n)
{
  this->k_value = new_n;
  this->k_changed = true;
}

void Recommender::change_out_file(std::string new_file_name)
{
  this->out_file = new_file_name;
}

//Sorting methods
void Recommender::quick_sort(KList & k_list)
{
  clock_t timer = Helpers::TimerStart();
  this->partition(k_list, 0, k_list.size() - 1);
  this->time_spent_sorting += (clock() - timer) / (float)CLOCKS_PER_SEC;
}

void Recommender::partition(KList & k_list, int left, int right)
{
  if((right - left) > 5)
  {
    int pivot_value = this->pivot(k_list, left, right);

    int i = left + 1;
    int j = right;

    while(i < j)
    {
      i += 1;
      while(pivot_value < std::get<0>(k_list[i]))
      {
        i += 1;
      }
      while(std::get<0>(k_list[j]) < pivot_value)
      {
        j -= 1;
      }
      if(i < j)
      {
        this->swap(k_list, i, j);
      }
    }

    this->swap(k_list, j, left + 1);

    this->partition(k_list, left, j - 1);
    this->partition(k_list, j + 1, right);
  }
  else
  {
    this->insertion_sort(k_list, left, right);
  }
}

float Recommender::pivot(KList & k_list, int left, int right)
{
  int center = (left + right) / 2;

  if(std::get<0>(k_list[center]) < std::get<0>(k_list[right]))
  {
    this->swap(k_list, right, center);
  }

  if(std::get<0>(k_list[left]) < std::get<0>(k_list[right]))
  {
    this->swap(k_list, right, left);
  }

  if(std::get<0>(k_list[left]) < std::get<0>(k_list[center]))
  {
    this->swap(k_list, center, left);
  }

  this->swap(k_list, center, left + 1);

  return std::get<0>(k_list[left + 1]);
}

void Recommender::swap(KList & k_list, int a, int b)
{
  auto temp = k_list[a];
  k_list[a] = k_list[b];
  k_list[b] = temp;
}

void Recommender::insertion_sort(KList & k_list, int left, int right)
{
  for(int i = 0; i < (right - left + 1); i++)
  {
    auto temp = k_list[i + left];
    int j = i + left;
    while(j > left && std::get<0>(temp) > std::get<0>(k_list[j - 1]))
    {
      k_list[j] = k_list[j - 1];
      j -= 1;
    }
    k_list[j] = temp;
  }
}

std::pair<float, float> Recommender::test_recs_HR()
{
  float hits = 0.0;
  float a_RHR = 0.0;

  for(int i = 0; i < this->test_data->nrows; i++)
  {
    bool in_rec = false;
    int j_val = 0;

    for(int j = 0; j < this->user_recs[i].size(); j++)
    {
      if(this->test_data->col_ind[i] == std::get<1>(this->user_recs[i][j]))
      {
        in_rec = true;
        j_val = j + 1;
      }
    }

    if(in_rec)
    {
      hits += 1.0;
    }
    if(j_val != 0)
    {
      a_RHR += (1.0 / j_val);
    }
  }

  float hit_rate = hits / this->test_data->nrows;
  a_RHR /= this->test_data->nrows;

  return std::make_pair(hit_rate, a_RHR);
}
