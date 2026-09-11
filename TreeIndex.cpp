#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <queue>
#include <string>
#include <cstddef>
#include "TreeIndex.h"
#include <algorithm>
#include <ctime>
#include <cmath>
using namespace std;
int kk = 100;

DataVector::DataVector(size_t dimension)
{
    setDimension(dimension);
}

DataVector::~DataVector() {}

DataVector::DataVector(const DataVector &other)
{
    v = other.v;
}

void DataVector::push_back(double value)
{
    v.push_back(value);
}

DataVector &DataVector::operator=(const DataVector &other)
{
    if (this != &other)
    {
        v = other.v;
    }
    return *this;
}

void DataVector::setDimension(size_t dimension)
{
    v.clear();
    v.resize(dimension, 0.0);
}

DataVector DataVector::operator+(const DataVector &other) const
{
    size_t size = v.size();
    DataVector res(size);
    if (size == other.v.size())
    {
        for (size_t i = 0; i < size; i++)
        {
            res.v[i] = v[i] + other.v[i];
        }
    }
    return res;
}

DataVector DataVector::operator-(const DataVector &other) const
{
    size_t size = v.size();
    DataVector res(size);
    if (size == other.v.size())
    {
        for (size_t i = 0; i < size; i++)
        {
            res.v[i] = v[i] - other.v[i];
        }
    }
    return res;
}

double DataVector::operator*(const DataVector &other) const
{
    size_t size = v.size();
    double res = 0.0;
    if (size == other.v.size())
    {
        for (size_t i = 0; i < size; i++)
        {
            res += v[i] * other.v[i];
        }
    }
    return res;
}

double DataVector::norm() const
{
    size_t size = v.size();
    double res = 0.0;
    for (size_t i = 0; i < size; i++)
    {
        res += v[i] * v[i];
    }
    return sqrt(res);
}

double DataVector::dist(const DataVector &other) const
{
    size_t size = v.size();
    double res = 0.0;
    if (size == other.v.size())
    {
        for (size_t i = 0; i < size; i++)
        {
            res += ((v[i] - other.v[i]) * (v[i] - other.v[i]));
        }
    }
    return sqrt(res);
}

size_t DataVector::size() const
{
    return v.size();
}

double &DataVector::operator[](int index)
{
    return v[index];
}

const double &DataVector::operator[](int index) const
{
    return v[index];
}

VectorDataset::VectorDataset(vector<DataVector> &data) : set(data) {}

VectorDataset::VectorDataset() {}

DataVector &VectorDataset::operator[](size_t idx)
{
    return set[idx];
}

size_t VectorDataset::size()
{
    return set.size();
}

void VectorDataset::push_back(DataVector v)
{
    set.push_back(v);
}

void VectorDataset::read_dataset(const string &filename)
{
    ifstream file(filename);
    set.clear();

    if (file.is_open())
    {
        string line;
        while (getline(file, line))
        {
            istringstream line_stream(line);

            DataVector vector;

            double value;
            while (line_stream >> value)
            {
                vector.push_back(value);
            }

            set.push_back(vector);
        }

        file.close();
    }
    else
    {
        cerr << "Unable to open file: " << filename << "\n";
    }
}

class TreeIndex
{
public:
    vector<DataVector> set;
    static TreeIndex *staticvar;
    TreeIndex(vector<DataVector>::iterator begin, vector<DataVector>::iterator end) : set(begin, end) {}

public:
    TreeIndex()
    {
        if (!staticvar)
            staticvar = this;
    }
    ~TreeIndex() {}
    static TreeIndex &GetInstance()
    {
        if (!staticvar)
            staticvar = new TreeIndex;
        return *staticvar;
    }
    void readdata(vector<DataVector> &arr)
    {
        (*staticvar).set.clear();
        for (int i = 0; i < arr.size(); i++)
        {
            (*staticvar).set.push_back(arr[i]);
        }
    }
    vector<DataVector> &getdata()
    {
        return set;
    }
};
TreeIndex *TreeIndex::staticvar = nullptr;

class KDTreeIndex : public TreeIndex
{
public:
    struct Node
    {
        TreeIndex splitInfo;
        Node *left;
        Node *right;

        Node(vector<DataVector>::iterator begin, vector<DataVector>::iterator end) : splitInfo(begin, end), left(nullptr), right(nullptr) {}
    };
    KDTreeIndex()
    {
        if (!staticvar)
            staticvar = this;
    }
    ~KDTreeIndex() {}

public:
    Node *root = nullptr;
    static KDTreeIndex *staticvar;
    static KDTreeIndex &GetInstance()
    {
        if (!staticvar)
            staticvar = new KDTreeIndex;
        return *staticvar;
    }
    bool choose(const std::vector<DataVector> &S, std::vector<DataVector>::iterator begin, std::vector<DataVector>::iterator end, int &splitDimension, double &median)
    {
        int dimension = S[0].size();
        double maxSpread = -1.0;

        for (int i = 0; i < dimension; ++i)
        {
            std::vector<double> values;
            std::for_each(begin, end, [&values, i](const DataVector &vec)
                          { values.push_back(vec[i]); });

            std::sort(values.begin(), values.end());
            double spread = values[values.size() - 1] - values[0];

            if (spread > maxSpread)
            {
                maxSpread = spread;
                splitDimension = i;
                median = values[(values.size() - 1) / 2];
            }
        }
        return true;
    }

    Node *kdtreevec(vector<DataVector> &dataset, vector<DataVector>::iterator begin, vector<DataVector>::iterator end)
    {
        if ((end - begin) <= kk)
        {
            return new Node(begin, end);
        }

        int split;
        double median;
        if (!choose(dataset, begin, end, split, median))
        {
            cout << "Error in finding spread.\n";
        }
        int lcount = 0, rcount = 0;
        std::for_each(begin, end, [&split, &median, &lcount, &rcount](const DataVector &vec)
                      {
            if (vec[split] <= median)
            {
                lcount++;
            }
            else
            {
                rcount++;
            } });
        std::sort(begin, end, [split](const DataVector &a, const DataVector &b)
                  { return a[split] < b[split]; });
        vector<DataVector> tmp(1);
        tmp[0].setDimension(4);
        tmp[0][0] = split;
        tmp[0][1] = median;
        tmp[0][2] = lcount;
        tmp[0][3] = rcount;
        Node *currentNode = new Node(tmp.begin(), tmp.end());

        // Recursively build the left and right subtrees
        currentNode->left = kdtreevec(dataset, begin, begin + lcount);
        currentNode->right = kdtreevec(dataset, begin + lcount, end);

        return currentNode;
    }
    void clearTree(Node *node)
    {
        if (node == nullptr)
            return;

        // Recursively delete left and right subtrees
        clearTree(node->left);
        clearTree(node->right);

        // Delete the current node
        delete node;
    }

    void Maketree(vector<DataVector> &dataset)
    {
        // Clear existing data
        clearTree(root);

        // Build the tree
        root = kdtreevec(dataset, dataset.begin(), dataset.end());
    }
    double calculateDistance(const DataVector &a, const DataVector &b)
    {
        double distance = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            distance += pow(a[i] - b[i], 2);
        }
        return sqrt(distance);
    }
    void searchLeafKNearest(Node *leafNode, const DataVector &target, int k, priority_queue<double> &nearestNeighbors)
    {
        if (leafNode == nullptr)
            return;

        // Iterate over the points in the leaf node and update nearest neighbors
        for (const DataVector &point : leafNode->splitInfo.set)
        {
            double distance = calculateDistance(target, point);

            nearestNeighbors.push(distance);

            // Keep only the top k nearest neighbors
            if (nearestNeighbors.size() > k)
            {
                nearestNeighbors.pop();
            }
        }
    }
    vector<double> searchKNearest(DataVector &target, int k)
    {
        priority_queue<double> nearestNeighbors;

        if (root == nullptr)
            return vector<double>(0);

        Node *curr = root;

        // Traverse the tree to find the leaf node containing the target
        while (curr->left != nullptr || curr->right != nullptr)
        {
            if (target[curr->splitInfo.set[0][0]] > curr->splitInfo.set[0][1])
            {
                if (curr->right != nullptr)
                {
                    curr = curr->right;
                }
                else
                {
                    break;
                }
            }
            else
            {
                if (curr->left != nullptr)
                {
                    curr = curr->left;
                }
                else
                {
                    break;
                }
            }
        }

        // Perform a refined search in the leaf node
        searchLeafKNearest(curr, target, k, nearestNeighbors);

        // Extract the k-nearest neighbors from the priority queue
        vector<double> result;
        while (!nearestNeighbors.empty())
        {
            result.push_back(nearestNeighbors.top());
            nearestNeighbors.pop();
        }

        // Reverse the result vector to get the nearest neighbors in the correct order
        reverse(result.begin(), result.end());

        return result;
    }
};
KDTreeIndex *KDTreeIndex::staticvar = nullptr;

class RPTreeIndex : public TreeIndex
{
public:
    struct Node
    {
        TreeIndex splitInfo;
        Node *left;
        Node *right;

        Node(vector<DataVector>::iterator begin, vector<DataVector>::iterator end) : splitInfo(begin, end), left(nullptr), right(nullptr) {}
    };
    RPTreeIndex()
    {
        if (!staticvar)
            staticvar = this;
    }
    ~RPTreeIndex() {}

public:
    Node *root = nullptr;
    static RPTreeIndex *staticvar;
    static RPTreeIndex &GetInstance()
    {
        if (!staticvar)
            staticvar = new RPTreeIndex;
        return *staticvar;
    }

    void clearTree(Node *node)
    {
        if (node == nullptr)
            return;

        // Recursively delete left and right subtrees
        clearTree(node->left);
        clearTree(node->right);

        // Delete the current node
        delete node;
    }
    double findFarthestPoint(const DataVector &point, const std::vector<DataVector> &dataset, vector<DataVector>::iterator begin, vector<DataVector>::iterator end)
    {
        double maxDistance = 0.0;
        DataVector farthestPoint;
        for (auto it = begin; it != end; ++it)
        {
            double distance = (point - *it).norm();
            if (distance > maxDistance)
            {
                maxDistance = distance;
                farthestPoint = *it;
            }
        }
        return maxDistance;
    }
    void chooserule(vector<DataVector> &dataset, vector<DataVector>::iterator begin, vector<DataVector>::iterator end, double &delta)
    {
        int size = (end - begin);
        DataVector x = *(begin + rand() % size);
        delta = findFarthestPoint(x, dataset, begin, end);
        delta *= 6;
        delta /= sqrt(dataset[0].size());
        delta *= (((rand() % 200) / 100.0) - 1);
    }
    Node *rptreevec(vector<DataVector> &dataset, vector<DataVector>::iterator begin, vector<DataVector>::iterator end)
    {
        if ((end - begin) <= kk)
        {
            return new Node(begin, end);
        }
        DataVector dir(dataset[0].size());
        for (int i = 0; i < dir.size(); i++)
        {
            dir[i] = rand();
        }
        double len = dir.norm();
        for (int i = 0; i < dir.size(); i++)
        {
            dir[i] /= len;
        }
        double delta;
        chooserule(dataset, begin, end, delta);
        auto comparator = [&dir](const DataVector &a, const DataVector &b)
        {
            return (a * dir) < (b * dir);
        };
        std::sort(begin, end, comparator);
        double median = (*(begin + (end - begin) / 2)) * dir;
        int left = 0;
        for (auto it = begin; it != end; ++it)
        {
            if ((*it) * dir <= median + delta)
            {
                left++;
            }
        }
        vector<DataVector> tmp(2);
        tmp[0] = dir;
        tmp[1].setDimension(1);
        tmp[1][0] = median + delta;
        Node *currentNode = new Node(tmp.begin(), tmp.end());
        // Recursively build the left and right subtrees
        currentNode->left = rptreevec(dataset, begin, begin + left);
        currentNode->right = rptreevec(dataset, begin + left, end);

        return currentNode;
    }
    void Maketree(vector<DataVector> &dataset)
    {
        // Clear existing data
        clearTree(root);

        // Build the tree
        root = rptreevec(dataset, dataset.begin(), dataset.end());
    }
    double calculateDistance(const DataVector &a, const DataVector &b)
    {
        double distance = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            distance += pow(a[i] - b[i], 2);
        }
        return sqrt(distance);
    }

    void searchLeafKNearest(Node *leafNode, const DataVector &target, int k, priority_queue<double> &nearestNeighbors)
    {
        if (leafNode == nullptr)
        {
            return;
        }
        // Iterate over the points in the leaf node and update nearest neighbors
        for (const DataVector &point : leafNode->splitInfo.set)
        {
            double distance = calculateDistance(target, point);

            nearestNeighbors.push(distance);

            // Keep only the top k nearest neighbors
            if (nearestNeighbors.size() > k)
            {
                nearestNeighbors.pop();
            }
        }
    }
    vector<double> searchKNearest(DataVector &target, int k)
    {
        priority_queue<double> nearestNeighbors;

        if (root == nullptr)
            return vector<double>(0);

        Node *curr = root;

        // Traverse the tree to find the leaf node containing the target
        while (curr->left != nullptr || curr->right != nullptr)
        {
            if ((target * curr->splitInfo.set[0]) > curr->splitInfo.set[1][0])
            {
                if (curr->right != nullptr)
                {
                    curr = curr->right;
                }
                else
                {
                    break;
                }
            }
            else
            {
                if (curr->left != nullptr)
                {
                    curr = curr->left;
                }
                else
                {
                    break;
                }
            }
        }

        searchLeafKNearest(curr, target, k, nearestNeighbors);

        // Extract the k-nearest neighbors from the priority queue
        vector<double> result;
        while (!nearestNeighbors.empty())
        {
            result.push_back(nearestNeighbors.top());
            nearestNeighbors.pop();
        }
        reverse(result.begin(), result.end());

        return result;
    }
};
RPTreeIndex *RPTreeIndex::staticvar = nullptr;
// global Datavectorsets to store input dataset and result dataset
VectorDataset tmp, resset;
// global datavector to store a test vector
DataVector test;
int main()
{
    srand(time(0));
    tmp.read_dataset("fmnist-train.csv");
    TreeIndex a;
    a.readdata(tmp.set);
    cout << a.set.size() << '\n';
    KDTreeIndex b;
    RPTreeIndex c;
    test = a.set[100];
    auto start_time = std::chrono::high_resolution_clock::now();
    b.Maketree(a.set);
    c.Maketree(a.set);
    vector<double> ans = b.searchKNearest(test, 10);
    vector<double> ans1 = c.searchKNearest(test, 10);
    for (int i = 0; i < ans.size(); i++)
    {
        cout << ans[i] << " " << ans[i] << '\n';
    }
    for (int i = 0; i < ans1.size(); i++)
    {
        cout << ans1[i] << " " << ans1[i] << '\n';
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() / 1000 << " milliseconds" << std::endl;
    return 0;
}