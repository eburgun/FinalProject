
#ifndef _HELPERS_H_
#define _HELPERS_H_

#include <iostream>
#include <fstream>

#include <stdexcept>

#include <string>
#include <sstream>
#include <vector>
#include <ctime>
//#include "CSR.h"

#include <limits> // for infinity

namespace Helpers
{
	using namespace std;

	vector<int> split(string str)
	{

		string itemStr;
		int itemInt;
		vector<int> items;
		items.reserve(4096);
		stringstream ss(str);

		while (ss >> itemStr)
		{
			itemInt = atoi(itemStr.c_str());
			items.push_back(itemInt);
		}

		return items;
	}

	int ToInt(string str)
	{
		int itemInt;
		stringstream ss(str);
		ss >> itemInt;
		return itemInt;
	}

	double ToDouble(string str)
	{
		double itemDouble;
		stringstream ss(str);
		ss >> itemDouble;
		return itemDouble;
	}

	string prompt(string str)
	{
		cout << str << endl;
		string response;
		cin >> response;

		return response;
	}

	int repeat_until_in_range(int min, int max, string strPrompt, int n_tries)
	{
		int result; //TODO: changed form result = NULL might not work
		string actualPrompt = "";
		for (int i = 0; i < n_tries; i++)
		{
			actualPrompt = string("Try ") + to_string(i + 1) + "/" + to_string(n_tries) +
				": " + strPrompt + " Range: [" + to_string(min) + ", " + to_string(max) + "]";
			result = ToInt(prompt(actualPrompt));
			if (result >= min && result <= max)
				return result;
		}

		throw out_of_range("Didn't enter a ratingValsid ratingValsue within the given parameters.");
	}

	template <typename T>
	void print_line(T str)
	{
		cout << str << endl;
	}

	//return a sorted list of cosines with max length of k
	vector<pair<int, float>> sort_cosines(float* cosines, int length, int k)
	{
		vector<pair<int, float>> sorted_cosines;
		//iterate over every element in the cosines that's positive
		for (int i = 0; i < length; i++)
		{
			float ratingVals = cosines[i];
			if (ratingVals > 0)
			{
				//if vector isn't full then simply add
				if (sorted_cosines.size() < k)
				{
					sorted_cosines.push_back(pair<int, float>(i, ratingVals));
				}
				else //if vector isn't full then replace lowest ratingValsue
				{
					int lowest_ind = -1;
					float lowest_ratingVals = std::numeric_limits<float>::max();
					for (int j = 0; j < k; j++)
					{
						if (sorted_cosines[j].second < lowest_ratingVals)
						{
							lowest_ind = j;
							lowest_ratingVals = sorted_cosines[j].second;
						}
					}
					//if the lowest_ind changed and proposed ratingVals is greater than lowest ratingVals
					if (lowest_ind != -1 && ratingVals > lowest_ratingVals)
					{
						sorted_cosines[lowest_ind].first = i;
						sorted_cosines[lowest_ind].second = ratingVals;
					}
				}
			}
			//else do nothing with unsimilar data
		}

		//sort the very few elements we have with bubble sort
		for (int i = 0; i < sorted_cosines.size(); i++)
		{
			for (int j = 0; j < sorted_cosines.size(); j++)
			{
				if (sorted_cosines[j].second < sorted_cosines[i].second)
					swap(sorted_cosines[i], sorted_cosines[j]);
			}
		}

		return sorted_cosines;
	}

	//start the clock
	clock_t TimerStart()
	{
		return clock();
	}

	//end the clock and also announce the time spent
	void TimerEnd(string prepended_str, clock_t startClock)
	{
		double time_spent = (clock() - startClock) / (double)CLOCKS_PER_SEC;
		std::cout << prepended_str << time_spent << " seconds." << std::endl;
	}
}

#endif // _HELPERS_H_
