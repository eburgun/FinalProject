CSR::CSR(std::string file_name)
{
	std::ifstream my_file(file_name);

	if (!my_file.is_open())
	{
		std::cout << "Failed to read file: " << file_name << std::endl;
		return;
	}

	double time_spent;
	clock_t start = clock();

	std::string line;

	//read header line
	getline(my_file, line);

	std::vector<int> items = ::Helpers::split(line);

	int rows = items[0];
	int cols = items[1];
	int total_elements = items[2];

	//declare the member variables
	this->nrows = rows;
	this->ncols = cols;
	this->nitems = total_elements;

	this->val_size = total_elements;
	this->col_ind_size = total_elements;
	this->row_ptr_size = rows + 1;

	this->val = new int[val_size];
	this->col_ind = new int[col_ind_size];
	this->row_ptr = new int[row_ptr_size];

	this->row_averages = new float[rows];

	std::vector<int> values;
	int running_total = 0;
	this->row_ptr[0] = running_total + 1; //for first element in row_ptr
	for (int cur_row = 0; cur_row < rows; cur_row++)
	{
		getline(my_file, line);
		values = ::Helpers::split(line);

		//import to val array
		for (size_t j = 1; j < values.size(); j += 2)
		{
      this->col_ind[running_total] = values[j - 1] - 1;
			this->val[running_total] = values[j];
			running_total += 1;
		}

		this->row_ptr[cur_row + 1] = running_total; //update row_ptr

		// to see progress
		// if (cur_row % 100 == 0)
		// 	printf("\rImport Progress: %.2f%%", (double)(cur_row * 100) / rows);
	}

	my_file.close();

	time_spent = (clock() - start) / (double)CLOCKS_PER_SEC;
	// std::cout << "\rImported " << file_name << " in: " << time_spent << " seconds." << std::endl;

	//compute row averages
	this->compute_row_averages();

}

CSR::CSR(int rows_, int cols_, int items_): nrows(rows_), ncols(cols_), nitems(items_),
	val_size(items_), col_ind_size(items_), row_ptr_size(rows_ + 1),
	val(new int[items_]), col_ind(new int[items_]), row_ptr(new int[rows_ + 1])
{

}

void CSR::add_new_users(CSR* new_users)
{
  this->first_new_user_id = this->nrows;

  this->nrows += new_users->nrows;
	this->nitems += new_users->nitems;

  int new_val_size = this->val_size + new_users->val_size;
  int new_col_ind_size = this->col_ind_size + new_users->col_ind_size;
  int new_row_ptr_size = this->row_ptr_size + new_users->nrows;

  int* new_val = new int[new_val_size];
  int* new_col_ind = new int[new_col_ind_size];
  int* new_row_ptr = new int[new_row_ptr_size];

  //copy existing col_ind and val data
  for(int i = 0; i < this->val_size; i++)
  {
    new_val[i] = this->val[i];
    new_col_ind[i] = this->col_ind[i];
  }

  //fill in the new col and val data from the new data CSR
  for(int i = 0; i < new_users->val_size; i++)
  {
    new_val[i + this->val_size] = new_users->val[i];
    new_col_ind[i + this->val_size] = new_users->col_ind[i];
  }

  //copy existing row_ptr data
  for(int i = 0; i < this->row_ptr_size; i++)
  {
    new_row_ptr[i] = this->row_ptr[i];
  }

  //fill in the row ptr of the new rows
  for(int i = 0; i < new_users->nrows; i++)
  {
    new_row_ptr[i + this->row_ptr_size] = new_row_ptr[this->row_ptr_size - 1] + new_users->row_ptr[i + 1] - 1;
  }

  //delete old data
  delete[] this->val;
  delete[] this->col_ind;
  delete[] this->row_ptr;

  //switch to new data
  this->val = new_val;
  this->col_ind = new_col_ind;
  this->row_ptr = new_row_ptr;

  this->val_size = new_val_size;
  this->col_ind_size = new_col_ind_size;
  this->row_ptr_size = new_row_ptr_size;
}

CSR::~CSR()
{
	// std::cout << "Calling CSR destructor...";
	delete[] val;
	delete[] col_ind;
	delete[] row_ptr;
	delete[] row_averages;

	val = nullptr;
	col_ind = nullptr;
	row_ptr = nullptr;
	row_averages = nullptr;

	// std::cout << " Done!" << std::endl;
}

int CSR::get_element(int row, int col)
{
	if (row < 1 || row > nrows || col < 1 || col > ncols)
	{
		throw std::out_of_range("Invalid index value (remember that indexing is Matlab style)");
	}

	//input is matlab style but storage is still 0 index based
	for (int p = row_ptr[row - 1]; p < row_ptr[row]; p++) {
		if (col_ind[p - 1] == col)
			return val[p - 1];
	}

	return 0;
}

CSR* CSR::transpose(const CSR* in_)
{
	//Clock in
	double time_spent;
	clock_t start = clock();

	//initialize the size of transpose obj
	CSR* trans =  new CSR(in_->ncols, in_->nrows, in_->nitems);

	/* ======== ROW_PTR BEGIN ======== */

	//set row_ptr first value to 1
	trans->row_ptr[0] = 1;

	//pad row_ptr with 0s
	for (int i = 1; i < trans->row_ptr_size; i++)
	{
		trans->row_ptr[i] = 0;
	}

	//stack nnz presence from each column to trans->row_ptr
	for (int i = 0; i < in_->nrows; i++)
	{
		for (int j = in_->row_ptr[i]; j < in_->row_ptr[i + 1]; j++)
		{
			int iT = in_->col_ind[j-1];
			trans->row_ptr[iT + 1] += 1;
		}
	}

	//make row_ptr cumulative
	for (int i = 1; i < trans->row_ptr_size; i++)
	{
		trans->row_ptr[i] += trans->row_ptr[i - 1];
	}

	/* ======== ROW_PTR END ======== */

	/* ======== COL_IND & VAL BEGIN ======== */

	//pad row_counts with 0s
	int row_counts[trans->nrows];
	for (int i = 0; i < trans->nrows; i++)
	{
		row_counts[i] = 0;
	}

	for (int i = 0; i < in_->nrows; i++)
	{
		for (int j = in_->row_ptr[i]; j < in_->row_ptr[i + 1]; j++)
		{
			int rowT = in_->col_ind[j - 1]; //j - 1 because 0 index based
			int col = trans->row_ptr[rowT] + row_counts[rowT] - 1;
			trans->col_ind[col] = i;
			trans->val[col] = in_->val[j - 1];
			row_counts[rowT]++;
		}
	}

	/* ======== COL_IND & VAL END ======== */

	//Clock out
	time_spent = (clock() - start) / (double)CLOCKS_PER_SEC;
	// std::cout << "Transposed in: " << time_spent << " seconds." << std::endl;

	trans->row_averages = new float[trans->nrows];
	trans->compute_row_averages();

	return trans;
}

bool CSR::is_transpose_of(CSR* other)
{
	if (this->ncols != other->nrows || this->nrows != other->ncols)
	{
		return false;
	}

	for (int i = 1; i < this->nrows + 1; i++)
	{
		for (int j = 1; j < this->ncols + 1; j++)
		{
			if (this->get_element(i, j) != other->get_element(j, i))
				return false;
		}
	}

	return true;
}

//check equality of this vs rhs(right hand side)
bool CSR::operator == (const CSR & rhs) const
{
	//self reference
	if (this == &rhs)
	{
		return true;
	}

	//check for same sized matrices
	if (this->ncols != rhs.ncols || this->nrows != rhs.nrows || this->nitems != rhs.nitems ||
		this->col_ind_size != rhs.col_ind_size || this->val_size != rhs.val_size || this->row_ptr_size != rhs.row_ptr_size)
	{
		return false;
	}

	//check for row_ptr equality
	for (int i = 0; i < this->row_ptr_size; i++)
	{
		if (this->row_ptr[i] != rhs.row_ptr[i])
			return false;
	}

	//check for val and col_ind equality
	for (int i = 0; i < this->col_ind_size; i++)
	{
		if (this->col_ind[i] != rhs.col_ind[i] || this->val[i] != rhs.val[i])
			return false;
	}

	return true;
}

float CSR::compute_cosines(int row_i, int row_j, bool UseUserAverages)
{

	if (row_i < 0 || row_i >= nrows || row_j < 0 || row_j >= nrows)
	{
		throw std::out_of_range("Doing this computation 0-index based");
	}

	int count_i = row_ptr[row_i + 1] - row_ptr[row_i];
	int count_j = row_ptr[row_j + 1] - row_ptr[row_j];
	float avg_i = UseUserAverages ? row_averages[row_i] : 0.0;
	float avg_j = UseUserAverages ? row_averages[row_j] : 0.0;

	int num_i = 0, num_j = 0;
	float cosine = 0, length_i = 0, length_j = 0;

	while (num_i < count_i && num_j < count_j)
	{
		int cur_i = row_ptr[row_i] + num_i - 1;
		int cur_j = row_ptr[row_j] + num_j - 1;

		if (col_ind[cur_i] == col_ind[cur_j])
		{
			cosine += (val[cur_i] - avg_i) * (val[cur_j] - avg_j); /* */
			length_i += (val[cur_i] - avg_i) * (val[cur_i] - avg_i);
			length_j += (val[cur_j] - avg_j) * (val[cur_j] - avg_j);
			num_i += 1;
			num_j += 1;
		}
		else if (col_ind[cur_i] > col_ind[cur_j])
		{
			length_j += (val[cur_j] - avg_j) * (val[cur_j] - avg_j);
			num_j += 1;
		}
		else //if (col_ind[cur_i] < col_ind[cur_j])
		{
			length_i += (val[cur_i] - avg_i) * (val[cur_i] - avg_i);
			num_i += 1;
		}
	}

  while(num_i < count_i)
  {
    int cur_i = row_ptr[row_i] + num_i;
    length_i += val[cur_i] * val[cur_i];
    num_i += 1;
  }

  while(num_j < count_j)
  {
    int cur_j = row_ptr[row_j] + num_j;
    length_j += val[cur_j] * val[cur_j];
    num_j += 1;
  }

	float denominator = length_i * length_j;
	if (denominator)
		cosine /= std::sqrt(denominator);
	else
		cosine = 0;

	return cosine;
}

float* CSR::compute_cosines_for(int row_i, bool UseUserAverages)
{
	float* cosines_temp = new float[nrows];

	//skip compute for self
	for (int j = 0; j < row_i; j++)
	{
		cosines_temp[j] = compute_cosines(row_i, j, UseUserAverages);
	}

	cosines_temp[row_i] = 1; //for self

	for (int j = row_i + 1; j < nrows; j++)
	{
		cosines_temp[j] = compute_cosines(row_i, j, UseUserAverages);
	}

	return cosines_temp;
}

void CSR::compute_row_averages() {

	//assume values have been imported

	//compute averages for each row
	int current_sum;
	int col_ind_first; //inclusive
	int col_ind_last; //not inclusive
	for (int row = 0; row < nrows; row++)
	{
		current_sum = 0;
		col_ind_first = this->row_ptr[row] - 1; //because this isn't matlab
		col_ind_last = this->row_ptr[row + 1] - 1;
		for (int col_ind_i = col_ind_first; col_ind_i < col_ind_last; col_ind_i++)
		{
			current_sum += this->val[col_ind_i];
		}

		//save the result to reuse many times later
		this->row_averages[row] = (float)current_sum / (col_ind_last - col_ind_first);
	}
}
