# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:24:04 2016

@author: eburgun
"""

import math

class CSR:
    
    
    """
    Initiation method that takes in a file name and starts the file processing.
    """
    def __init__(self,fileName):
        self.column_idx = []
        self.rating = []
        self.row_ptr = []
        self.columns = 0
        self.rows = 0
        self.nonzero_values = 0
        dataFile = open(fileName)
        for i, line in enumerate(dataFile):
            self.processLine(line,i)  
        dataFile.close()
    """
    The primary method for processing the data from text file.
    
    """
    def processLine(self,line,increment):
        data_array = line.split(" ")
        
        if increment == 0:
            self.rows = int(data_array[0])
            self.columns = int(data_array[1])
            self.nonzero_values = int(data_array[2])
            self.rating = [0] * self.nonzero_values
            self.column_idx = [0] * self.nonzero_values
            self.row_ptr = [0] * (self.rows + 1)
            
        else:
            self.row_ptr[increment] = len(data_array)/2 + self.row_ptr[increment-1] 
            
            for i in xrange(len(data_array)/2):
                self.column_idx[self.row_ptr[increment-1] + i] = int(data_array[2*i]) - 1
                self.rating[self.row_ptr[increment-1] + i] = int(data_array[2*i + 1])  
                
    """
    Simple method for returning the rating in any position for the matrix
    """         
    def getElem(self,row, col):
        nrow = self.row_ptr[row]
        
        while(self.column_idx[nrow] <= col):
            if(self.column_idx[nrow] == col):
                return self.rating[nrow]
            nrow += 1
            
        return 0    
    
    """
    Antiquated cosSimil method used for testing the updated cosine similarity method.
    """
    def cosSimil(self,rowi,rowj):
        nrowi = self.row_ptr[rowi + 1] - self.row_ptr[rowi]
        nrowj = self.row_ptr[rowj + 1] - self.row_ptr[rowj]
        ni = 0
        nj = 0
        cosine = 0
        lengthi = 0
        lengthj = 0
        
        while ni < nrowi and nj < nrowj:
            ci = self.row_ptr[rowi] + ni
            cj = self.row_ptr[rowj] + nj
            if self.column_idx[ci] == self.column_idx[cj]:
                cosine += self.rating[ci] * self.rating[cj]
                lengthi += self.rating[ci] * self.rating[ci]
                lengthj += self.rating[cj] * self.rating[cj]
                ni += 1
                nj += 1
            elif self.column_idx[cj] < self.column_idx [ci]:
                lengthj += self.rating[cj] * self.rating[cj]
                nj += 1
            else:
                lengthi += self.rating[ci] * self.rating[ci]
                ni += 1 
                
        while ni < nrowi:
            ci = self.row_ptr[rowi] + ni
            lengthi += self.rating[ci] * self.rating[ci]
            ni += 1
            
        while nj < nrowj:
            cj = self.row_ptr[rowj] + nj
            lengthj += self.rating[cj] * self.rating[cj]
            nj += 1
            
        if (lengthi * lengthj > 0):
            
            cosine /= math.sqrt(lengthi) * math.sqrt(lengthj)
            return cosine
        else:
            return 0       
            
    """
    Transpose method that creates new versions of all of the pieces and then replaces the curent versions with the new versions.
    """
    def transpose(self):
        nrow = self.columns
        ncol = self.rows
        nrow_ptr = [0] * (self.columns + 1)
        row_count = [0] * self.columns
        ncol_idx = [0] * self.nonzero_values
        nvalue = [0] * self.nonzero_values
        
        for i in xrange(len(self.column_idx)):
            nrow_ptr[self.column_idx[i]+1] += 1
            
        for i in xrange(len(nrow_ptr)-1):
            nrow_ptr[i+1] += nrow_ptr[i]

        for i in xrange(self.rows):
            for j in xrange(self.row_ptr[i+1] - self.row_ptr[i]):
                i2 = self.column_idx[j + self.row_ptr[i]]
                ncol_idx[nrow_ptr[i2] + row_count[i2]] = i
                nvalue[nrow_ptr[i2] + row_count[i2]] = self.rating[j + self.row_ptr[i]]
                row_count[i2] += 1
                
        self.rows = nrow
        self.columns = ncol
        self.row_ptr = nrow_ptr
        self.column_idx = ncol_idx
        self.rating = nvalue
    """
    Method that is used to check if multiple transpositions alters the matrix structure.    
    """
    def checkColumnOrder(self):
        cumul = 0
        for i in range(self.rows):
            for j in range(self.row_ptr[i+1] - self.row_ptr[i]):
                if j >= 1:
                    x = self.column_idx[j + self.row_ptr[i] - 1]
                    y =  self.column_idx[j + self.row_ptr[i]]
                    
                    if x >= y:
                        cumul += 1
                
        print cumul
