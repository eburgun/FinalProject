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
	this->rows = rows;
	this->columns = cols;
	this->nonZeroValues = total_elements;

	this->ratingVals_size = total_elements;
	this->columnIndex_size = total_elements;
	this->rowPtr_size = rows + 1;

	this->ratingVals = new int[ratingVals_size];
	this->columnIndex = new int[columnIndex_size];
	this->rowPtr = new int[rowPtr_size];

	this->row_averages = new float[rows];

	std::vector<int> ratingValsues;
	int running_total = 0;
	this->rowPtr[0] = running_total + 1; //for first element in rowPtr
	for (int cur_row = 0; cur_row < rows; cur_row++)
	{
		getline(my_file, line);
		ratingValsues = ::Helpers::split(line);

		//import to ratingVals array
		for (size_t j = 1; j < ratingValsues.size(); j += 2)
		{
      this->columnIndex[running_total] = ratingValsues[j - 1] - 1;
			this->ratingVals[running_total] = ratingValsues[j];
			running_total += 1;
		}

		this->rowPtr[cur_row + 1] = running_total; //update rowPtr

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

CSR::CSR(int rows_, int cols_, int items_): rows(rows_), columns(cols_), nonZeroValues(items_),
	ratingVals_size(items_), columnIndex_size(items_), rowPtr_size(rows_ + 1),
	ratingVals(new int[items_]), columnIndex(new int[items_]), rowPtr(new int[rows_ + 1])
{

}

CSR::~CSR()
{
	// std::cout << "Calling CSR destructor...";
	delete[] ratingVals;
	delete[] columnIndex;
	delete[] rowPtr;
	delete[] row_averages;

	ratingVals = nullptr;
	columnIndex = nullptr;
	rowPtr = nullptr;
	row_averages = nullptr;

	// std::cout << " Done!" << std::endl;
}

int CSR::getElement(int row, int col)
{
	if (row < 1 || row > rows || col < 1 || col > columns)
	{
		throw std::out_of_range("InratingValsid index ratingValsue (remember that indexing is Matlab style)");
	}

	//input is matlab style but storage is still 0 index based
	for (int p = rowPtr[row - 1]; p < rowPtr[row]; p++) {
		if (columnIndex[p - 1] == col)
			return ratingVals[p - 1];
	}

	return 0;
}

CSR* CSR::transpose(const CSR* in_)
{
	//Clock in
	double time_spent;
	clock_t start = clock();

	//initialize the size of transpose obj
	CSR* trans =  new CSR(in_->columns, in_->rows, in_->nonZeroValues);

	/* ======== rowPtr BEGIN ======== */

	//set rowPtr first ratingValsue to 1
	trans->rowPtr[0] = 1;

	//pad rowPtr with 0s
	for (int i = 1; i < trans->rowPtr_size; i++)
	{
		trans->rowPtr[i] = 0;
	}

	//stack nnz presence from each column to trans->rowPtr
	for (int i = 0; i < in_->rows; i++)
	{
		for (int j = in_->rowPtr[i]; j < in_->rowPtr[i + 1]; j++)
		{
			int iT = in_->columnIndex[j-1];
			trans->rowPtr[iT + 1] += 1;
		}
	}

	//make rowPtr cumulative
	for (int i = 1; i < trans->rowPtr_size; i++)
	{
		trans->rowPtr[i] += trans->rowPtr[i - 1];
	}

	/* ======== rowPtr END ======== */

	/* ======== columnIndex & ratingVals BEGIN ======== */

	//pad row_counts with 0s
	int row_counts[trans->rows];
	for (int i = 0; i < trans->rows; i++)
	{
		row_counts[i] = 0;
	}

	for (int i = 0; i < in_->rows; i++)
	{
		for (int j = in_->rowPtr[i]; j < in_->rowPtr[i + 1]; j++)
		{
			int rowT = in_->columnIndex[j - 1]; //j - 1 because 0 index based
			int col = trans->rowPtr[rowT] + row_counts[rowT] - 1;
			trans->columnIndex[col] = i;
			trans->ratingVals[col] = in_->ratingVals[j - 1];
			row_counts[rowT]++;
		}
	}

	/* ======== columnIndex & ratingVals END ======== */

	//Clock out
	time_spent = (clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "Transposed in: " << time_spent << " seconds." << std::endl;

	trans->row_averages = new float[trans->rows];
	trans->compute_row_averages();

	return trans;
}

bool CSR::is_transpose_of(CSR* other)
{
	if (this->columns != other->rows || this->rows != other->columns)
	{
		return false;
	}

	for (int i = 1; i < this->rows + 1; i++)
	{
		for (int j = 1; j < this->columns + 1; j++)
		{
			if (this->getElement(i, j) != other->getElement(j, i))
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
	if (this->columns != rhs.columns || this->rows != rhs.rows || this->nonZeroValues != rhs.nonZeroValues ||
		this->columnIndex_size != rhs.columnIndex_size || this->ratingVals_size != rhs.ratingVals_size || this->rowPtr_size != rhs.rowPtr_size)
	{
		return false;
	}

	//check for rowPtr equality
	for (int i = 0; i < this->rowPtr_size; i++)
	{
		if (this->rowPtr[i] != rhs.rowPtr[i])
			return false;
	}

	//check for ratingVals and columnIndex equality
	for (int i = 0; i < this->columnIndex_size; i++)
	{
		if (this->columnIndex[i] != rhs.columnIndex[i] || this->ratingVals[i] != rhs.ratingVals[i])
			return false;
	}

	return true;
}

float CSR::compute_cosines(int row_i, int row_j, bool UseUserAverages)
{

	if (row_i < 0 || row_i >= rows || row_j < 0 || row_j >= rows)
	{
		throw std::out_of_range("Doing this computation 0-index based");
	}

	int count_i = rowPtr[row_i + 1] - rowPtr[row_i];
	int count_j = rowPtr[row_j + 1] - rowPtr[row_j];
	float avg_i = UseUserAverages ? row_averages[row_i] : 0.0;
	float avg_j = UseUserAverages ? row_averages[row_j] : 0.0;

	int num_i = 0, num_j = 0;
	float cosine = 0, length_i = 0, length_j = 0;

	while (num_i < count_i && num_j < count_j)
	{
		int cur_i = rowPtr[row_i] + num_i - 1;
		int cur_j = rowPtr[row_j] + num_j - 1;

		if (columnIndex[cur_i] == columnIndex[cur_j])
		{
			cosine += (ratingVals[cur_i] - avg_i) * (ratingVals[cur_j] - avg_j); /* */
			length_i += (ratingVals[cur_i] - avg_i) * (ratingVals[cur_i] - avg_i);
			length_j += (ratingVals[cur_j] - avg_j) * (ratingVals[cur_j] - avg_j);
			num_i += 1;
			num_j += 1;
		}
		else if (columnIndex[cur_i] > columnIndex[cur_j])
		{
			length_j += (ratingVals[cur_j] - avg_j) * (ratingVals[cur_j] - avg_j);
			num_j += 1;
		}
		else //if (columnIndex[cur_i] < columnIndex[cur_j])
		{
			length_i += (ratingVals[cur_i] - avg_i) * (ratingVals[cur_i] - avg_i);
			num_i += 1;
		}
	}

  while(num_i < count_i)
  {
    int cur_i = rowPtr[row_i] + num_i;
    length_i += ratingVals[cur_i] * ratingVals[cur_i];
    num_i += 1;
  }

  while(num_j < count_j)
  {
    int cur_j = rowPtr[row_j] + num_j;
    length_j += ratingVals[cur_j] * ratingVals[cur_j];
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
	float* cosines_temp = new float[rows];

	//skip compute for self
	for (int j = 0; j < row_i; j++)
	{
		cosines_temp[j] = compute_cosines(row_i, j, UseUserAverages);
	}

	cosines_temp[row_i] = 1; //for self

	for (int j = row_i + 1; j < rows; j++)
	{
		cosines_temp[j] = compute_cosines(row_i, j, UseUserAverages);
	}

	return cosines_temp;
}

void CSR::compute_row_averages() {

	//assume ratingValsues have been imported

	//compute averages for each row
	int current_sum;
	int columnIndex_first; //inclusive
	int columnIndex_last; //not inclusive
	for (int row = 0; row < rows; row++)
	{
		current_sum = 0;
		columnIndex_first = this->rowPtr[row] - 1; //because this isn't matlab
		columnIndex_last = this->rowPtr[row + 1] - 1;
		for (int columnIndex_i = columnIndex_first; columnIndex_i < columnIndex_last; columnIndex_i++)
		{
			current_sum += this->ratingVals[columnIndex_i];
		}

		//save the result to reuse many times later
		this->row_averages[row] = (float)current_sum / (columnIndex_last - columnIndex_first);
	}
}
