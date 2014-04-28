from random import random, choice
from copy import copy
 
try:
    import psyco
    psyco.full()
except ImportError:
    pass
 
 
FLOAT_MAX = 1e100
 
 

 
 
 
if __name__ == "__main__":
    k = 4  # # clusters
    points = [Point(0.25,0),Point(0.21,0),Point(0.22,0),Point(0.23,0),Point(0.35,0),Point(0.25,0),Point(5,0)]
    cluster_centers = lloyd(points, k)
    print(points)
    ccc = []
    for c in cluster_centers:
        ccc.append(c.x)
    
    print(ccc)
