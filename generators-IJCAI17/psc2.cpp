#include <iostream>
#include <getopt.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <limits>

using namespace std;


///////////////////////////////////////////////////////
/* VARIABLES of the generator */

int n = 100;  // Number of nodes
int m = 400;  // Number of clauses
int k = 0;    // Average size of clauses (flexible part)
int K = 0;    // Rigid clause size
double b = 1; // Beta for vars
double B = 1; // Beta for clauses
double g = 2; // Gamma for vars. Notice that gamma = 1/beta + 1
double G = 2; // Gamma for clauses
double T = 0; // Temperature
bool varRename = false;  // if true varRename variables acording to angle
int seed = 0; // Random seed

vector <vector<double> > angle; // Vector representing the angle in [0,2PI)
                                // angle[0] for variables and angle[1] for clauses
vector<vector<int> > neighs;    // Vector representing neighbors of a clause node
int edges = 0;                  // Edges already created
double error = 0.001;

///////////////////////////////////////////////////////
/* Print the usage of this program */

void printUsage(char* prog){
  cerr << "c" << endl;
  cerr << "c Usage: " << prog << " [arguments]" << endl;
  cerr << "c    Arguments:" << endl;
  cerr << "c         -n <int>   : number of nodes (default=100)" << endl;
  cerr << "c         -m <int>   : number of clauses (default=400)" << endl;
  cerr << "c         -k <int>   : average clause size (default=0)" << endl;
  cerr << "c         -K <int>   : clause size for regular model (default=0)" << endl;
  cerr << "c         -b <float> : beta for variables (default=1)" << endl;
  cerr << "c         -B <float> : beta for clauses (default=1)" << endl;
  cerr << "c         -g <float> : gamma for variables (default=2) Notice that beta = 1/(gamma-1)" << endl;
  cerr << "c         -G <float> : gamma for clauses (default=2)" << endl;
  cerr << "c         -T <float> : temperature (default=0)" << endl;
  cerr << "c         -r         : varRename variables and reorder clauses so similar ones are closer (default=false)" << endl;
  cerr << "c         -s <int>   : random seed (default=0)" << endl;
  cerr << "c" << endl;
}

///////////////////////////////////////////////////////
/* Print the usage of this program and exit */
//TODO rewrite

void printUsageAndExit(char* prog, int code){
  printUsage(prog);
  exit(code);
}

///////////////////////////////////////////////////////
/* Parse the arguments given to the program */

void parseArgs(int argc, char** argv){
  int opt;
  // Parse argument
  while((opt=getopt(argc, argv, "n:m:k:K:b:B:g:G:T:s:r")) != -1){
    switch(opt){
    case 'n':
      n = atoi(optarg);
      break;
    case 'm':
      m = atoi(optarg);
      break;
    case 'k':
      k = atoi(optarg);
      break;
    case 'K':
      K = atoi(optarg);
      break;
    case 'b':
      b = atof(optarg);
      break;
    case 'B':
      B = atof(optarg);
      break;
    case 'g':
      g = atof(optarg);
      b = 1.0 / (g - 1);
      break;
    case 'G':
      G = atof(optarg);
      B = 1.0 / (G - 1);
      break;
    case 'T':
      T = atof(optarg);
      break;
    case 's':
      seed = atoi(optarg);
      break;
    case 'r':
      varRename = true;
      break;
    default:
      cerr << "c ERROR: unrecognised argument" << endl;
      printUsageAndExit(argv[0], -1);
    }
  }
	
  // Checks argument values
  // number of nodes : n
  if(n<1){
    cerr << "ERROR: n (number of nodes) must be greater than 0" << endl;
    printUsageAndExit(argv[0], -1);
  }
  // number of connections : m
  if(m<1){
    cerr << "ERROR: m (number of clauses) must be greater than 0" << endl;
    printUsageAndExit(argv[0], -1);
  }
  // temperature : T
  if(T < 0){ //TODO : Check this interval is correct
    cerr << "ERROR: T (temperature) must be in the interval [0,1]" << endl;
    printUsageAndExit(argv[0], -1);
  }
  // beta : b
  if(b < 0 || b > 1 || B < 0 || B > 1){
    cerr << "ERROR: b and B (beta) must be in the interval [0,1]" << endl;
    printUsageAndExit(argv[0], -1);
  }
  // gamma : g
  if(g < 2 || G < 2){
    cerr << "ERROR: g and G (gamma) must be greater than 2" << endl;
    printUsageAndExit(argv[0], -1);
  }
}

///////////////////////////////////////////////////////
/* Given a vector x, returns the order of its elements, or the inverse */
bool myorder(pair<int,double>x, pair<int,double>y) {
  return x.second < y.second;
}
void getorder(vector <double> x, vector <int> &y, bool inverse){
  vector <pair<int,double> >z(x.size());
  y.resize(x.size());
  for(int i=0; i<x.size(); i++) {
    z[i].first = i;
    z[i].second = x[i];
  }
  sort(z.begin(), z.end(), myorder);
  if (inverse) {
    for(int i=0; i<x.size(); i++)
      y[z[i].first] = i;
  } else {
    for(int i=0; i<x.size(); i++)
      y[i] = z[i].first;
  }
}

///////////////////////////////////////////////////////
/* Returns true if there exists an edge between variable nodes i and clause node j */

bool checkEdge(int i, int j){
  return find(neighs[j].begin(), neighs[j].end(), i) != neighs[j].end();
}

///////////////////////////////////////////////////////
/* Create an edge between variable i and clause j (or vice verse if isvar=false) */

void createEdge(int i, int j){
  //cerr << "c New edge "<<i<<" - "<<j<<endl;
  assert(i>0 && i<=n && j>0 && j<=m);
  assert(!checkEdge(i,j));
  edges++;
  neighs[j].push_back(i);
}

///////////////////////////////////////////////////////
/* Print the resulting SAT instance (including sign for literals) */
 
void printGraph(){

  cout << "c" << endl;
  cout << "c Popularity vs Similarity SAT Instance Generator" << endl;
  cout << "c Created by Jesús Giráldez Crú and Jordi Levy" << endl;
  cout << "c" << endl;
  cout << "c n=" << n << " m=" << m << " k=" << k << " K=" << K << " b=" << b << " B=" << B << " T=" << T << "size=" << edges << endl;
  cout << "c" << endl;
  cout << "p cnf " << n << " "<< neighs.size()-1 << endl;

  if (!varRename) {
    for(int j=1; j<neighs.size(); j++){
      if(neighs[j].size() > 0){  
	for(int i=0; i<neighs[j].size(); i++)
	    cout << neighs[j][i] * (rand()%2==0?1:-1) << " ";
	cout << "0" << endl;
      }
      else
	cerr << "Warning: generated empty clause" << endl;
    }
  } else {
    vector <int>indvar, indcla;
  
    getorder(angle[0],indvar,true);
    getorder(angle[1],indcla,false);
 
    for(int j=1; j<neighs.size(); j++){
      k = indcla[j];
      if(neighs[k].size() > 0){  
	for(int i=0; i<neighs[k].size(); i++)
	  cout << indvar[neighs[k][i]] * (rand()%2==0?1:-1) << " ";
	cout << "0" << endl;
      }
      else
	cerr << "Warning: generated empty clause" << endl;
    }
  }
}

///////////////////////////////////////////////////////
/* Insert an element in an ordered list, and keep ordered */

template<typename value>
void insertOrdered(vector<pair<value,double> >& l, pair<value,double> elem){	
  int pos;

  for(pos=0; pos<l.size(); pos++){
    if(elem.second < l[pos].second){
      break;
    }
  }
  l.resize(l.size()+1);

  for(int i=l.size()-1; i>pos; i--)
    l[i] = l[i-1];
  l[pos] = elem;
}

///////////////////////////////////////////////////////

double myabs(double k){
  return k >= 0 ? k : -k;
}

///////////////////////////////////////////////////////
/* Angular coordinate in PS model */

double anglePS(){
  return ((double)rand() / (double)RAND_MAX) * 2 * M_PI;
}

///////////////////////////////////////////////////////
/* Computes hyperbolic distance between variable i and clause j */

double hyperDist(int i, int j) {
  if (checkEdge(i,j)) return numeric_limits<double>::max();

  double diffang = M_PI - myabs(M_PI - myabs(angle[1][j] - angle[0][i]));
  //return b*log(i) + B*log(j) + log(diffang);
  return pow(i,b) * pow(j,B) * diffang;
}

///////////////////////////////////////////////////////

int main(int argc, char** argv){
  
  // Parse arguments
  parseArgs(argc, argv);
  
  // Initialize
  srand(seed);
  neighs.resize(m+1);
  angle.resize(2);
  angle[0].resize(n+1);
  angle[1].resize(m+1);
  
  // Compute angle location
  for(int i=1; i<=n; i++){
    angle[0][i] = anglePS();
  }
  for(int i=1; i<=m; i++){
    angle[1][i] = anglePS();
  }

  // Add edges for fixed arity (K>0)
  if (K>0) {
    for (int j=1; j<=m; j++) {
      vector<pair<int,double> > selected;
      for (int i=1; i<=n; i++) {
	if (selected.size() < K)
	  insertOrdered(selected, make_pair(i,hyperDist(i,j)));
	else {
	  double d = hyperDist(i,j);
	  if(d < selected[selected.size()-1].second){
	    insertOrdered(selected, make_pair(i,d));
	    selected.pop_back();
	  }
	}
      }
      if (T==0) {
	for(int i=0; i<selected.size(); i++)
	  createEdge(selected[i].first, j);
      } else {
	
	double R = selected[selected.size()-1].second;
	double sum;
	//cerr <<  R << endl;
	
	do {
	  double der = 0;  // Value of the derivative
	  sum = 0;         // Value of the function
	  for (int i=1; i<=n; i++) {
	    double aux1 = pow(hyperDist(i,j)/R, 1.0/T);
	    double aux2 = 1.0 / (aux1 + 1);
	    sum += aux2;
	    der += aux2 * aux2 * aux1 / R / T;
	  }
	  R = R + (K - sum) / der;     // Newtow's method to find the point where the function is K
	  if (R<0) R=1e-20;            // R is newer negative (try to fix a divergent computation)
	  //cerr <<  "R="<<R << " der="<<der << " sum="<<sum << endl;
	}
	while (K/sum < 1 - error || K/sum > 1 + error);

	int e = 0;
	while (e < K) {
	  int i = rand()%n+1;
	  double prob = 1.0 / ( 1.0 + pow(hyperDist(i,j)/R, 1.0/T) );
	  if( ((double)rand() / (double)RAND_MAX) < prob) {
	    { createEdge(i,j); e++; }
	  }
	}
      }
    }
  }
  
  // Add edges for flexible arity (k>0)
  if (k>0) {
    vector<pair<pair<int,int>,double> > selected;
    for (int i=1; i<=n; i++)
      for (int j=1; j<=m; j++) {
	if (selected.size() < k*m)
	  insertOrdered(selected, make_pair(make_pair(i,j), hyperDist(i,j)));
	else {
	  double d = hyperDist(i,j);
	  if(d < selected[selected.size()-1].second){
	    insertOrdered(selected, make_pair(make_pair(i,j),d));
	    selected.pop_back();
	  }
	}
      }
    
    if (T==0) {
      for(int i=0; i<selected.size(); i++)
	createEdge(selected[i].first.first, selected[i].first.second);
    } else {
      
      double R = selected[selected.size()-1].second;
      double sum;
      cerr << "R = " << R << endl;

      do {
	double der = 0;  // Value of the derivative
	sum = 0;         // Value of the function
	for (int i=1; i<=n; i++)
	  for (int j=1; j<=m; j++) {
	    if (!checkEdge(i,j)) {
	      double aux1 = pow(hyperDist(i,j)/R, 1.0/T);
	      double aux2 = 1.0 / (aux1 + 1);
	      sum += aux2;
	      der += aux2 * aux2 * aux1 / R / T;
	    }
	  }
	R = R + (k*m - sum) / der;   // Newtow's method to find the point where the function is k*m
	if (R<0) R=1e-100;           // R is newer negative (try to fix a divergent computation)
	cerr <<  "R = "<< R << " der = "<<der << " sum = "<<sum << endl;
      }
      while (k*m/sum < 1 - error || k*m/sum > 1 + error);

      int e = 0;
      while (e < k*m) {
	int i = rand()%n+1;
	int j = rand()%m+1;
	double prob = 1.0 / ( 1.0 + pow(hyperDist(i,j)/R, 1/T) );
	if( ((double)rand() / (double)RAND_MAX) < prob) {
	  { createEdge(i,j); e++; }
	}
      }
    }
  }
  
  // Finally print select random sign and print the graph
  printGraph();
}



///////////////////////////////////////////////////////
