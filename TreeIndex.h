#ifndef TREEINDEX_H
#define TREEINDEX_H

#include <vector>
#include <queue>
#include <cstddef>
#include <vector>
#include <string>

class DataVector
{
private:
    vector<double> v;

public:
    // constructor
    DataVector(size_t dimension = 0);

    // destructor
    ~DataVector();

    // copy constructor
    DataVector(const DataVector &other);

    // copy assignment operator
    DataVector &operator=(const DataVector &other);

    // sets the dimension of the vector
    void setDimension(size_t dimension = 0);

    // addition operator to add two vectors
    DataVector operator+(const DataVector &other) const;

    // subtraction operator to subtract two vectors
    DataVector operator-(const DataVector &other) const;

    // multiplication operator to multiply two vectors
    double operator*(const DataVector &other) const;

    // calculates norm of the vector
    double norm() const;

    // calculates distance between two vectors
    double dist(const DataVector &other) const;

    // to push back the value in the vector
    void push_back(double value);

    size_t size() const;

    double &operator[](int index);

    const double &DataVector::operator[](int index) const;
};

using namespace std;

class VectorDataset
{
public:
    vector<DataVector> set;

    VectorDataset(vector<DataVector> &data);
    VectorDataset();

    // to access ith vector in the dataset
    DataVector &operator[](size_t);

    // returns size of the dataset
    size_t size();

    // to append data vector into dataset
    void push_back(DataVector v);

    // to read dataset from a file
    void read_dataset(const string &filename);
};

class TreeIndex
{
public:
    std::vector<DataVector> set;
    static TreeIndex *staticvar;

    TreeIndex(std::vector<DataVector>::iterator begin, std::vector<DataVector>::iterator end);
    TreeIndex();
    ~TreeIndex();
    static TreeIndex &GetInstance();
    void readdata(std::vector<DataVector> &arr);
    std::vector<DataVector> &getdata();
};

class KDTreeIndex : public TreeIndex
{
public:
    struct Node;
    Node *root = nullptr;
    static KDTreeIndex *staticvar;

    KDTreeIndex();
    ~KDTreeIndex();
    static KDTreeIndex &GetInstance();
    void Maketree(std::vector<DataVector> &dataset);
    std::vector<double> searchKNearest(DataVector &target, int k);

private:
    bool choose(const std::vector<DataVector> &S, std::vector<DataVector>::iterator begin, std::vector<DataVector>::iterator end, int &splitDimension, double &median);
    Node *kdtreevec(std::vector<DataVector> &dataset, std::vector<DataVector>::iterator begin, std::vector<DataVector>::iterator end);
    void clearTree(Node *node);
    double calculateDistance(const DataVector &a, const DataVector &b);
    void searchLeafKNearest(Node *leafNode, const DataVector &target, int k, std::priority_queue<double> &nearestNeighbors);
};

class RPTreeIndex : public TreeIndex
{
public:
    struct Node;
    Node *root = nullptr;
    static RPTreeIndex *staticvar;

    RPTreeIndex();
    ~RPTreeIndex();
    static RPTreeIndex &GetInstance();
    void Maketree(std::vector<DataVector> &dataset);
    std::vector<double> searchKNearest(DataVector &target, int k);

private:
    void chooserule(std::vector<DataVector> &dataset, std::vector<DataVector>::iterator begin, std::vector<DataVector>::iterator end, double &delta);
    Node *rptreevec(std::vector<DataVector> &dataset, std::vector<DataVector>::iterator begin, std::vector<DataVector>::iterator end);
    void clearTree(Node *node);
    double findFarthestPoint(const DataVector &point, const std::vector<DataVector> &dataset, std::vector<DataVector>::iterator begin, std::vector<DataVector>::iterator end);
    double calculateDistance(const DataVector &a, const DataVector &b);
    void searchLeafKNearest(Node *leafNode, const DataVector &target, int k, std::priority_queue<double> &nearestNeighbors);
};

#endif // TREEINDEX_H
