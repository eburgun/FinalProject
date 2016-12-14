//use this to create a gradient between full cold start cases to full non-cold start cases
//source data is N first ratings of the small training data based on how many ratings you want in the cold start cases
//N ranges from 0 to 100 in increments of 10

#include <string>
#include <fstream>
#include <ostream>
#include <sstream>
#include <iostream>

int main(void)
{
  std::string sourceFileName = "OriginalDataSets/SmallTrainSet.txt";

  //take i * 10 first rating of each user and store these as new files
  for(int i = 1; i < 11; i++)
  {
    std::ifstream sourceFile(sourceFileName);
    std::string inputLine;
    //skip first row
    std::getline(sourceFile, inputLine);

    if (!sourceFile.is_open())
    {
      std::cout << "Failed to read file: " << sourceFileName << std::endl;
      return 0;
    }

    std::string folderPath = "ColdTests/Ratings/" + std::to_string(i) + "0/";
    std::string fileName = "SmallColdStartTrain.txt";
    int numRows = 943;
    int numCols = 1682;
    int numRatings = i * 10; //per row

    //setup output file
    std::ofstream outputFile(folderPath + fileName);
    //write first row
    outputFile << numRows << " " << numCols << " " << (numRows * numRatings) << std::endl;

    for(int row = 0; row < numRows; row++)
    {
      std::getline(sourceFile, inputLine);
      std::string token;
      std::stringstream ss(inputLine);
      for(int col = 0; col < i * 20 && ss >> token; col++)
      {
        outputFile << token << " ";
      }
      outputFile << std::endl;
    }

    outputFile.close();
    sourceFile.close();

  }
  return 0;
}
