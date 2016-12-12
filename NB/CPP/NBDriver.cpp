#include <stdio.h>
#include "NBRecommender.h"

int main()
{
    std::string outPut = "Output.txt";
    std::string trainingFile = "train1.txt";
    std::string testFile = "test1.txt";
    int kVal = 10;
    int nVal = 5;
    NBRecommender * rec = new NBRecommender(trainingFile,testFile, outPut, kVal, nVal);
    delete rec;
    return 0;
}