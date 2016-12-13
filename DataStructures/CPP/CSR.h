#ifndef _CSR_H_
#define _CSR_H_

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <ctime>
#include <vector>

#include "Helpers.h"

class CSR
{
public:
  CSR(std::string file_name);
  CSR(int rows_, int cols_, int items_);
	~CSR();

	int getElement(int row, int col);
	static CSR* transpose(const CSR* stor);
	bool is_transpose_of(CSR* other);
	bool operator == (const CSR & rhs) const;
	void init_cosine_storage();
	float compute_cosines(int row_i, int row_j, bool UseUserAverages);
	float* compute_cosines_for(int row_i, bool UseUserAverages);

	//virtual dimentions
	int rows = 0;
	int columns = 0;
	int nonZeroValues = 0;

	//size of base storage
	int ratingVals_size = 0;
	int columnIndex_size = 0;
	int rowPtr_size = 0;

	//base storage
	int* ratingVals = nullptr;
	int* columnIndex = nullptr;
	int* rowPtr = nullptr;

  //special storage to avoid recomputing
	float* row_averages = nullptr;

  void compute_row_averages();

};

#include "CSR.cpp"

#endif //_CSR_H_ defined
