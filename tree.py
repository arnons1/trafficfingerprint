import os # For starting up Mathematica
from subprocess import call
from dpkt.pcap import Reader # Wireshark parsing
import math
import difflib # Finding closest strings
from nt import getcwd # Get current working directory
import tkinter # UI
import tkinter.scrolledtext as tkst
import numpy as np # Number methods (average, median, stdev,...)
import scipy as sp
from matplotlib.transforms import offset_copy # Graphs
import matplotlib.pyplot as pl # Graphs
import pylab # Graphs
from random import random, choice
from copy import copy
from sklearn.metrics import hamming_loss, log_loss

### TODO:
### 1. Create new training set with PCAPS with filter (tcp || dns) && !ipv6 && !(ip.addr==224.0.0.0/16)
### 2. Choose PCAP to compare with - add to UI
### 3.


lzHashmap = {}
observed_time_vector = []
pcaps_filelist_for_training = []
centroids = []
decision_boundaries = []
list_of_dirs = []
is_first_time = True
fp_db=[] # For storing the fingerprints

#===============================================================================
# node is a tree node. It contains a name, probability and a list of sons
#===============================================================================
class node:
	def __init__(self, name, prob, sons=[], size=0):
		self.prob = prob 			# Probability ('value')
		self.name = name  			# Name ('tag')
		self.sub_tree = sons  		# List of children
		self.sub_tree_size = size

#===============================================================================
# Stores a float.
# This is because we miscalculated the need for updating probabilities when we
# used a tuple. This is a bypass to the immutability of a tuple.
#===============================================================================
class edgeprob:
	def __init__(self, prob=0.0):
		self.prob = prob

class fingerprint:
	def __init__(self,tag='',tree=[]):
		self.tag=tag
		self.tree=tree

#===============================================================================
# Class for storing all UI levelElements
#===============================================================================

class tkstuff:
	def __init__(self, codebook_default="8"):
		self.window = tkinter.Tk() # Start up TK
		self.directorylb = tkinter.Listbox(self.window, height=5, width=30, selectmode=tkinter.MULTIPLE)
		self.statusString = tkinter.StringVar()
		self.var = tkinter.StringVar(self.window) # Hack begins
		self.sb = tkinter.Spinbox(self.window, from_=1, to=15,textvariable=self.var)
		self.var.set("8") # Stupid dirty hack ends. Sets the default centroids value to 8.
		self.tb = tkst.ScrolledText(
 								master = self.window,
     							wrap   = tkinter.WORD,
	 							width  = 95,
		    	 				height = 18, font=("Courier New",10)
		    	 				)

#======================================== KMEANS ++ ===============================================
class Point:
	slots__ = ["x", "y", "group"]
	def __init__(self, x=0.0, y=0.0, group=0):
		self.x, self.y, self.group = x, y, group

def nearest_cluster_center(point, cluster_centers):
	"""Distance and index of the closest cluster center"""
	def sqr_distance_2D(a, b):
		return (a.x - b.x) ** 2 + (a.y - b.y) ** 2

	min_index = point.group
	min_dist = 1e100
	for i, cc in enumerate(cluster_centers):
		d = sqr_distance_2D(cc, point)
		if min_dist > d:
			min_dist = d
			min_index = i
	return (min_index, min_dist)


def kpp(points, cluster_centers):
	cluster_centers[0] = copy(choice(points))
	d = [0.0 for _ in range(len(points))]

	for i in range(1, len(cluster_centers)):
		summation = 0
		for j, p in enumerate(points):
			d[j] = nearest_cluster_center(p, cluster_centers[:i])[1]
			summation += d[j]
		summation *= random()
		for j, di in enumerate(d):
			summation -= di
			if summation > 0:
				continue
			cluster_centers[i] = copy(points[j])
			break

	for p in points:
		p.group = nearest_cluster_center(p, cluster_centers)[0]

def lloyd(points, nclusters):
	cluster_centers = [Point() for _ in range(nclusters)]
	# call k++ init
	kpp(points, cluster_centers)
	lenpts10 = len(points) >> 10
	changed = 0
	while True:
		# group element for centroids are used as counters
		for cc in cluster_centers:
			cc.x = 0
			cc.y = 0
			cc.group = 0
		for p in points:
			cluster_centers[p.group].group += 1
			cluster_centers[p.group].x += p.x
			cluster_centers[p.group].y += p.y
		for cc in cluster_centers:
			cc.x /= cc.group
			cc.y /= cc.group
		# find closest centroid of each PointPtr
		changed = 0
		for p in points:
			min_i = nearest_cluster_center(p, cluster_centers)[0]
			if min_i != p.group:
				changed += 1
				p.group = min_i
		# stop when 99.9% of points are good
		if changed <= lenpts10:
			break
	for i, cc in enumerate(cluster_centers):
		cc.group = i
	return cluster_centers

#======================================== KMEANS ++ ===============================================

def clearGlobals():
	lzHashmap.clear()
	observed_time_vector.clear()
	#pcaps_filelist_for_training.clear()
	centroids.clear()
	decision_boundaries.clear()
	#list_of_dirs.clear()

#===============================================================================
# generateTreeFromString generates a tree from string obviously
#===============================================================================
def generateTreeFromString(string):
	global lzHashmap
	lzHashmap.clear()
	lzHashmap = getLZList(string, lzHashmap)  # Generates a hashtable for all variants, stored in 'lzHashmap'
	l = sorted(lzHashmap)  # Sort lzHashmap and get it as a nice list
	t = buildTree(l)  # Build a tree
	setProbTree(t, countLeavesTree(t))  # Set the probability for the tree nodes
	setProbEdgesTree(t)  # Set the probability for the tree edges
	# printTreeForWolfram(t, name)  # Print out the tree for export to Wolfram Mathematica
	return t

#===============================================================================
# Creates a Lempel Ziv style lzHashmap for a given string, of any alphabet
# Stores result in a global variable called lzHashmap. Recursive method
#===============================================================================
def getLZListR(head, tail):
	global lzHashmap
	if len(tail) == 0 and len(head) == 0:  # Empty list
		return
	if head in lzHashmap:
		lzHashmap[head] += 1  # This item exists. Add one to its counter
		if len(tail) == 0:
			return
		else:  # And if this is not the final letter, try the next combination (recurse)
			getLZListR(head + tail[0], tail[1:])
	else:
		lzHashmap[head] = 1  # Item wasn't in lzHashmap, so its brand new entry....
		if len(tail) == 0:
			return  # No more letters to check
		if len(tail) == 1:  # And there are still more letters to go, so recurse (only one more letter)
			getLZListR(tail[0], '')
		else:  # And there are still more letters to go, so recurse (more than one letter)
			getLZListR(tail[0], tail[1:])

#===============================================================================
# Creates a Lempel Ziv style lzHashmap for a given string, of any alphabet
# Stores result in a global variable called lzHashmap. Iterative method
#===============================================================================
def getLZList(string,hashm={}):
	flag = True
	tempstr = ''
	start = 0
	end = 1
	while(end <= len(string) and flag == True):
		tempstr = string[start:end]
		if tempstr in hashm:
			hashm[tempstr] += 1
			if end >= len(string):
				flag = False
				return hashm
			else:  # More letters to come
				end += 1
		else:  # Item isn't in lzHashmap
			hashm[tempstr] = 1  # New entry
			if end == len(string):
				flag = False
				return hashm
			else:  # More letters to check
				start = end
				end += 1


#===============================================================================
# buildTree builds a tree given a sorted list of items (from the lzHashmap usually)
#===============================================================================
def buildTree(listToInsert):
	root = node('root', 0, [])  # Create new root item with probability 0
	[ insertToTree(root, item) for item in listToInsert ]  # Insert items one by one
	return root

def insertToTree(root, tag):
	newitem = (node(tag, 0.0, []), edgeprob(0.0))  # New tuple (Node,Probability of edge)
	son = 0  # For preventing the last "if" clause

	if isLeaf(root) == True:  # Empty subtree - first tag being inserted
		root.sub_tree.append(newitem)  # Place in the node's children
		root.sub_tree_size = 1
		root.sub_tree.sort(key=lambda sub_tree_item: sub_tree_item[0].name)

	else:  # The subtree isn't empty
		son = findAppSon(root.sub_tree, tag)  # Find if there exists a child with the correct tag
		if son != -1:  # It exists, so call this method with that child as the root
			root.sub_tree_size += 1
			insertToTree(son[0], tag)

	if isLeaf(root) == False and son == -1:  # No child was found and the root isn't a leaf, so add it to the root.
		root.sub_tree.append(newitem)
		root.sub_tree_size += 1
		root.sub_tree.sort(key=lambda sub_tree_item: sub_tree_item[0].name)

#===============================================================================
# findAppSon checks if it can find a son with a specific tag.
# Returns -1 if not found, or the actual son if it was found.
#===============================================================================
def findAppSon(sonList, item):
	for son in sonList:
		if item.find(son[0].name) == 0:
			return son
	return -1

#===============================================================================
# isLeaf returns True if node is a leaf. False if not
#===============================================================================
def isLeaf(node):
	if len(node.sub_tree) == 0:
		return True
	else:
		return False

#===============================================================================
# countLeavesTree counts leaves in a tree, given its root.
#===============================================================================
def countLeavesTree(root):
	l = [ countLeavesTree1(son) for son in root.sub_tree ]
	return sum(l)

#===============================================================================
# countLeaves is used by countLeavesTree, but also uses countLeavesTree. See what
# we did there?
#===============================================================================
def countLeavesTree1(node_tuple):
	if isLeaf(node_tuple[0]) == True:
		return 1
	else:
		return countLeavesTree(node_tuple[0])

#===============================================================================
# Finds the leaves in a tree. Returns a list of the leaves.
#===============================================================================
def findLeavesTree(root):
	l = [ findLeavesTree1(son) for son in root.sub_tree ]
	return flatten(l)

#===============================================================================
# Method called by above findLeavesTree
#===============================================================================
def findLeavesTree1(node_tuple):
	if isLeaf(node_tuple[0]) == True:
		return [node_tuple[0]]
	else:
		return findLeavesTree(node_tuple[0])


#===============================================================================
# findNodeInTree finds a specific node by name in the tree and returns it
#===============================================================================
def findNodeInTree(root,tag):
	l = [ findNodeInTree1(son,tag) for son in root.sub_tree ]
	return flatten(l)

def findNodeInTree1(node_tuple,tag):
	if node_tuple[0].name == tag:
		return [node_tuple[0]]
	else:
		return findNodeInTree(node_tuple[0],tag)

#===============================================================================
# Finds the maximum depth in a tree and returns that integer
#===============================================================================
def findMaxDepthTree(root):
	l = [ findMaxDepthTree1(son) for son in root.sub_tree ]
	return max(l)

#===============================================================================
# Method used by findMaxDepthTree
#===============================================================================
def findMaxDepthTree1(node_tuple):
	if isLeaf(node_tuple[0]) == True:
		return 1
	else:
		return 1+findMaxDepthTree(node_tuple[0])

#===============================================================================
# setProbTree sets probabilities for the nodes (full probability for event)
# Accepts a tree root and the number of leaves (this is for uniform probability for the leaves)
#===============================================================================
def setProbTree(root, num_leaves):
	l = [ setProbLeaves(son, num_leaves) for son in root.sub_tree ]
	root.prob = sum(l)
	return sum(l)

#===============================================================================
# setProbLeaves sets probabilities for the rest of the nodes that aren't the root
#===============================================================================
def setProbLeaves(node_tuple, num_leaves):
	if isLeaf(node_tuple[0]) == True:
		node_tuple[0].prob = (1 / num_leaves)
		return node_tuple[0].prob
	else:
		l = [ setProbLeaves(son, num_leaves) for son in node_tuple[0].sub_tree ]
		node_tuple[0].prob = sum(l)
		return node_tuple[0].prob

#===============================================================================
# setProbEdgesTree Sets probabilities for the conditional probabilities (edges) for the root
#===============================================================================
def setProbEdgesTree(root):
	for son in root.sub_tree:
		son[1].prob = son[0].prob  # For every son of the ROOT ONLY, set probability of edge to probability of son
		setProbEdgesLeaves(son)  # Then start setting probabilities for all sons, from the top down
	return sum

#===============================================================================
# setProbEdgesLeaves Sets probabilities for the conditional probabilities (edges) for the non-root
#===============================================================================
def setProbEdgesLeaves(node_tuple):
	if isLeaf(node_tuple[0]) == True:
		return  # A leaf doesn't have any edges, so just finish up the recursion here
	else:
		for son in node_tuple[0].sub_tree:  # For every son,
			if node_tuple[0].prob != 0:
				son[1].prob = (son[0].prob / node_tuple[0].prob)  # Probability for edge is probability of son / this vertex's probability
			else:
				print("Error - Probability is 0 in node_tuple[0]. Check please!")
				return
			setProbEdgesLeaves(son)  # Now tell the son to do the same


#===============================================================================
# genListOfNodes Returns a list of all nodes in the tree including root.
#===============================================================================
def genListOfNodes(root):
	l=[]
	l.append(root)
	l.append([ genListOfNodes1(son) for son in root.sub_tree ])
	return flatten(l)

#===============================================================================
# Method called by above genListOfNodes
#===============================================================================
def genListOfNodes1(node_tuple):
	if isLeaf(node_tuple[0]) == True:
		return [node_tuple[0]]
	else:
		return genListOfNodes(node_tuple[0])

#===============================================================================
# printTreeForWolfram prints a tree nicely for WolframMathematica to display
#===============================================================================
def printTreeForWolfram(root, name, start=True, f=None):
	c = ','  # The c variable is used to prevent adding of extra commas when not necessary
	if(start == True):
		f = open(name + '.nb', 'w')
		s = "g={"
		f.write(s)
		c = ''
	for son in root.sub_tree:
		s = c + "{\"%s | %d | %f\" -> \"%s | %d | %f\" , \"%f\" }" % (root.name, root.sub_tree_size, root.prob, son[0].name, son[0].sub_tree_size, son[0].prob, son[1].prob)
		f.write(s)
		if c == '':
			c = ','
		printTreeForWolfram(son[0], name, False, f)
	if(start == True):
		s = "};\nTreePlot[g, Automatic, \"root | %d | %f\", VertexLabeling -> True]" % (root.sub_tree_size, root.prob)
		f.write(s)
		f.close()
		call(["c:\Program Files\Wolfram Research\Mathematica\9.0\Mathematica.exe", os.getcwd() + "/" + name + ".nb"])


#===============================================================================
# findAllPcapFiles creates a list of pcap files in the current directory, to
# be stored in the global variable 'pcaps_filelist_for_training'
#===============================================================================
def findAllPcapFiles(wd):
	for file in os.listdir(os.path.join(getcwd(),wd)):
		if file.endswith(".pcap"):
			pcaps_filelist_for_training.append(os.path.join(getcwd(),wd,file))

#===============================================================================
# collectAllRelativeTimestamps calls collectRelativeTimestampsForSingleFile for
# every file in the file list
#===============================================================================
def collectAllRelativeTimestamps(wd):
	for file in pcaps_filelist_for_training:
		collectRelativeTimestampsForSingleFile(file,wd)

#===============================================================================
# collectRelativeTimestampsForSingleFile generates relative time differences for a single file
# Saved in global variable observed_time_vector
#===============================================================================
def collectRelativeTimestampsForSingleFile(file,wd):
	f = open(file, "rb") # Open file
	pcapReader = Reader(f) # Parse
	frame_counter = 0
	last_time = 0
	for ts, _buf in pcapReader:
		frame_counter += 1
		if frame_counter > 1 :
			observed_time_vector.append(ts - last_time) # Calculate relative timestamp and save
		else:
			observed_time_vector.append(0) # First frame should have 0 as a timestamp
		last_time = ts
	f.close()

#===============================================================================
# createLloydMaxCodebook creates a quantized vector, with "num_codes" codes in the centroids
# based on the Lloyd Max Kmeans++ scheme. The data is based on the observed_time_vector
# observation vector.
#===============================================================================
def createLloydMaxCodebook(num_codes):
	listOfPoints = []
	for obsitem in observed_time_vector:
		listOfPoints.append(Point(obsitem,0.0)) # 1D K-Means++ clustering, so Y value is 0
	cluster_centers = lloyd(listOfPoints, num_codes)
	for value in cluster_centers:
		centroids.append(value.x) # Append centroid X value to the list


#===============================================================================
# getCodeFromCodebook gets a code for a specific value from the centroids. Returns a char
# of the code to be used, ranging from 'a' to whatever the largest code is.
#===============================================================================
def getCodeFromCodebook(value):
	for i in range(len(decision_boundaries)):
		try:
			if ((value >= decision_boundaries[i]) and (value <= decision_boundaries[i + 1])):
				return chr(i + 97)  # 0 maps to a, 1 to b and so on
		except IndexError:
			return chr(i + 97)

#===============================================================================
# parsePcapAndGetQuantizedString Parses a filename specified by 'file' and returns
# a string, that should be inserted into a tree later
#===============================================================================
def parsePcapAndGetQuantizedString(file,max_len=0,wd=''):
	if len(wd)>0:
		file = os.path.join(getcwd(),wd,file)
	f = open(file, "rb")
	pcapReader = Reader(f)
	frame_counter = 0
	last_time = 0
	vector = []
	for ts, _buf in pcapReader:
		frame_counter += 1
		if frame_counter > max_len and max_len>0:
			break
		if frame_counter > 1:
			vector.append(getCodeFromCodebook(ts - last_time))
		else:
			vector.append(getCodeFromCodebook(0))
		last_time = ts
	f.close()
	return ''.join(vector)

#===============================================================================
# flatten flattens a list of lists [[]] -> []
#===============================================================================
def flatten(lst):
	return sum( ([x] if not isinstance(x, list) else flatten(x)
		     for x in lst), [] )


#===============================================================================
# findNodeProbabilityInList Looks for an item in a list and returns its probability if found
# or -1 if not found.
#===============================================================================
def findNodeProbabilityInList(item, node_list):
	for li in node_list:
		if li.name == item.name:
			return li.prob
	return -1

# Identical to above, but just names, not nodes.
def findTagProbabilityInList(itemname, node_list):
	for li in node_list:
		if li.name == itemname:
			return li.prob
	return 0


#===============================================================================
# smoothKLContinuity attempts to find the closest match of an item (by name)
# in a list of nodes, by looking at the sum of its values. Also makes use of the
# difflib matcher.
#===============================================================================
def smoothKLContinuity(itemname, node_list):
	name_list = [ x.name for x in node_list ] # Get list of names only
	close_matches = difflib.get_close_matches(itemname, name_list,len(name_list),0.0) # Find 5 closest matches
	sums = [ (lambda x: sum([ ord(y) for y in x ]))(x) for x in close_matches] # Generate their sums
	itemname_sum = sum([ ord(x) for x in itemname ]) # Generate our search-item's sum
	closest = min(sums, key=lambda x:abs(x-itemname_sum)) # Find closest entry from the 3 found
	index_in_sums_list = sums.index(closest)  # Find the index in the original list
	return node_list[name_list.index(close_matches[index_in_sums_list])].prob # Get probability


#===============================================================================
# calculateKLDistance Calculates the KL distance between two trees. The third parameter
# (factor) is used to divide the result of the discontinuity smoother (smoothKLContinuity)
# so that it is essentially 'distancing' bad matches. Returns a float with the
# modified (smoothed) KL distance.
#===============================================================================
def calculateKLDistance(tree1, tree2, factor=0.5):
	p = genListOfNodes(tree1) #findLeavesTree(tree1) # Tree1 is the fingerprint. Only leaves.
	q = genListOfNodes(tree2) #findLeavesTree(tree2) # Second list
	# TODO: I altered the above, because I think we may need to do this on all nodes instead of just the leaves.
	kl_distance_sum = 0
	counter = 0

	for node_item in p:
		qi_prob = findNodeProbabilityInList(node_item, q)
		if qi_prob == -1:
			counter += 1
			qi_prob = smoothKLContinuity(node_item.name, q) * factor
		kl_distance_sum += (math.log(node_item.prob / qi_prob, len(centroids)) * node_item.prob) # Log with base len(centroids) (size of our alphabet)
	return kl_distance_sum

#===============================================================================
# checkTreeShapeDiff is the second of our home-made algorithms. It attempts to find
# similarity in the tree shapes. The result is a tuple. The first is a boolean stating if
# the trees are identical. The second lists the number of mismatches found.
# TODO: Normalize the number
#===============================================================================
def checkTreeShapeDiff(tree1,tree2):
	if len(tree1.sub_tree)==0 and len(tree2.sub_tree)==0:
		return (True,0)
	else:
		subtree_sum = 0
		t_or_f = True
		for i in range(max(len(tree1.sub_tree),len(tree2.sub_tree))):
			try:
				# There exists subtrees in both tree1 and tree2 that we haven't checked yet
				(returned_t_or_f, returned_subtree_sum) = checkTreeShapeDiff(tree1.sub_tree[i][0], tree2.sub_tree[i][0])
				subtree_sum += returned_subtree_sum
				t_or_f *= returned_t_or_f
			except IndexError:
				# Mismatch in sizes - either tree1 or tree2 have less subtrees, so sum up some differences
				t_or_f = False
				if len(tree1.sub_tree)<=i:
					# Root2 has more childen, so subtree_sum up all of tree2's subtrees
					subtree_sum += 1+ (tree2.sub_tree[i][0]).sub_tree_size # Get this child's subtree size
				else:
					# Root1 has more children so subtree_sum up all of tree1's subtrees
					subtree_sum += 1+ (tree1.sub_tree[i][0]).sub_tree_size
		return (t_or_f==True, subtree_sum)


#===============================================================================
# quantDist returns the quantization distance (normalized) between two centroids values
# (for instance, between 'a' and 'e')
#===============================================================================
def quantDist(left, right): # Always between 0 and 1, unless error: then -1.
	try:
		entirerange = centroids[-1]-centroids[0]
		return abs(centroids[ ord(left)-97 ] - centroids[ ord(right)-97 ])/entirerange # Quantization distance is the distance between two centroids
	except IndexError:
		print("l: %s r: %s not found" %(left,right))
		return 1



### TODO:
###  1. Normalize the values ('average' them out so that they total 1 per level
###  3. Check obviously. Verify results
###  4. Build some sort of testing framework - so that we can run many tests at once with varying vlaues of factors and length of files and so on.
###  5. UI - to make things look nice.
###  6. Find string in tree   - ????


#===============================================================================
# compare tree1 with tree2 on level
# We call this function for all the levels we want to compare
# The function computes quant distance for all the nodes in tree1 and level
#===============================================================================
def compareLevels(tree1, tree2, level):
	total = 0
	levelList = getNodeNamesInLevel(tree1, level)
	for nodeName in levelList:
		total += compareNodeToLevel(nodeName, tree2, level)
	return total

#===============================================================================
# compareNodeToLevel compare node from tree1 with all the level nodes of tree2
#===============================================================================
def compareNodeToLevel (nodeTree1, tree2, level):
	smallest = 1
	levelNodes = getNodeNamesInLevel(tree2, level)
	for node in levelNodes:
		smallest = min(smallest,quantDist(nodeTree1[-1], node[-1]))				#compare all nodes in level with node from tree1 and get the smallest
	return smallest

#===============================================================================
# getNodeNamesInLevel returns list of node names in a specific level
#===============================================================================
def getNodeNamesInLevel(root, level):
	l = [ getNodeNamesInLevel1(son, level) for son in root.sub_tree ]
	return flatten(l)

#===============================================================================
# used by getNodeNamesInLevel to find all nodes in level
#===============================================================================
def getNodeNamesInLevel1(node_tuple, level):
	if len(node_tuple[0].name) == level:
		return node_tuple[0].name
	else:
		return getNodeNamesInLevel(node_tuple[0], level)

#===============================================================================
# compare tree1 to tree2 quantization distance.
# return distance between 0 and 1
#===============================================================================
def compareTreesByLevel(tree1, tree2):
	total = 0
	level = 1
	levelNumT1 = findMaxDepthTree(tree1)
	while level <= levelNumT1:
		total += compareLevels(tree1, tree2, level)
		level += 1
	return total / levelNumT1


#===============================================================================
#  getNodesInLevel returns list of nodes name in level
#===============================================================================
def getNodesInLevel(root, level):
	l = [ getNodesInLevel1(son, level) for son in root.sub_tree ]
	return flatten(l)
#===============================================================================
# used by getNodesInLevel to find nodes in level
#===============================================================================
def getNodesInLevel1(node_tuple, level):
	if len(node_tuple[0].name) == level:
		return node_tuple[0]
	else:
		return getNodesInLevel(node_tuple[0], level)

#===============================================================================
# llfun - Calculates log-loss
#===============================================================================
def llfun(act, pred):
	epsilon = 1e-15
	pred = sp.maximum(epsilon, pred)
	pred = sp.minimum(1-epsilon, pred)
	ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
	ll = ll * -1.0/len(act)
	return ll

#===============================================================================
# compareFingerprintWithLZListLL Compares a fingerprint with a capture (lempel-ziv'd)
# and returns the Log-Loss value
#===============================================================================
def compareFingerprintWithLZListLL(fp, lzlist, window_size):
	totalLogLoss = 0
	lzlist = sorted(lzlist)
	for listElem in lzlist:
		levelElements = getNodesInLevel(fp,len(listElem)) # Pull out the level from the fingerprint tree
		nodeProb = findTagProbabilityInList(listElem,levelElements)
		#totalLogLoss += llfun(
		#					[centroids[ord(listElem)-97]]
		#					, nodeProb)
		epsilon = 1e-15
		pred = sp.maximum(epsilon, nodeProb)
		pred = sp.minimum(1-epsilon, pred)
		#ll = log_loss([listElem],[[nodeProb,1-nodeProb]])
		ll2 = - math.log(pred)
		totalLogLoss += ll2
	return -(totalLogLoss)/(window_size*math.log(epsilon))

#===============================================================================
# findMaxProbOfNode finds the node with maximum probability in the subtree
#===============================================================================
def findMaxProbOfNode(node):
	largest=0
	for son in node.sub_tree:
		if son[1].prob>largest:
			largest = son[1].prob
			tag = son[0].name
	return tag

#===============================================================================
# compareFingerprintWithLZListHL compares a fingerprint with a capture (lempel ziv'd)
# according to the Hamming Loss method.
#===============================================================================
def compareFingerprintWithLZListHL(fp, window, lzlist):
	# Find last entry inducted into the LZ List from the window
	for i in range(len(window)):
		if window[i:] in lzlist:
			# Item is contained in LZ List - we can use it
			longestMatch = window[i:]
			nodesFound = findNodeInTree(fp,longestMatch)
			# Find most probable match
			if len(nodesFound) == 0: # Nothing found
				# Nothing was found, most probable result is one of the root children
				estimateTag = findMaxProbOfNode(fp)
			else: # Found success
				# Estimate based on this node's children
				if isLeaf(nodesFound[0]):
					# Leaf, return same as root
					estimateTag = findMaxProbOfNode(fp)
				else:
					estimateTag = findMaxProbOfNode(nodesFound[0])
			return estimateTag[-1]


#===============================================================================
# compareFingerprintWithCapture wrapper function that performs two tests:
# Log loss and Hamming loss, for a sepcific fingerprint and a captured string
#===============================================================================
def compareFingerprintWithCapture(fp,capturedQuantizedString,window_size=8):
	hamming_result_vec = []
	hashmap2 = {}
	window_size = min(window_size,len(capturedQuantizedString)) # Just in case
	runTimes = len(capturedQuantizedString)-window_size+1 # Amount of times the loop will have to run
	counter=0
	for i in range(runTimes):
		window = capturedQuantizedString[i:i+window_size]
		# Algo1
		hashmap2.clear()
		hashmap2 = getLZList(window,hashmap2)  # Generates a hashtable for all variants, stored in 'lzHashmap'
		result = compareFingerprintWithLZListLL(fp, hashmap2, window_size)
		if 'smallest' not in locals():
			smallest = result
		else:
			if(result<smallest):
				smallest = result
		# Algo2
		estimatedTag = compareFingerprintWithLZListHL(fp, window, hashmap2)
		if i+window_size+1 <= len(capturedQuantizedString):
			estimatedWindow = window+estimatedTag
			realWindow = capturedQuantizedString[i:i+window_size+1]
			#if estimatedTag==realWindow[-1]:
			#	s="<--- SUCCESS"
			#	counter+=1
			#else:
			#	counter=0
			#	s=""
			#print("Counter: %d - Expected to see: %c. Actually saw %c  %s"%(counter,estimatedTag,realWindow[-1],s))
			res = hamming_loss(list(realWindow),list(estimatedWindow))
			hamming_result_vec.append(res)
		else:
			# Do nothing. We've reached the end of the window and don't want to estimate any more.
			return (smallest,minHammingWindow(hamming_result_vec,window_size))
	return (smallest,minHammingWindow(hamming_result_vec,window_size))

#===============================================================================
# minHammingWindow calculates the minimal Hamming Loss found in a sequence of hamming windows
#===============================================================================
def minHammingWindow(hammingResultVec, windowSize=8):
	minHamming = 1
	runTimes = len(hammingResultVec)-windowSize+1
	for i in range(runTimes):
		window = hammingResultVec[i:i+windowSize]
		vecSum = sum(window)
		minHamming = min(vecSum, minHamming)
	return minHamming


#===============================================================================
# training looks in a directory list (should be specified), and for each directory
# it scans the PCAP files and collects timestamps. Afterwards, it runs
# the LloydMax generator method, and sets decisiou boundaries.
# The second parameter in the function is the size of the centroids. (Default: 8)
#===============================================================================
def training(dir_list = ['training_1'], codebook_size = 8):
	for wd in dir_list:
		findAllPcapFiles(wd) # Get list of pcap files
		collectAllRelativeTimestamps(wd)

	createLloydMaxCodebook(codebook_size) # Create code book
	centroids.sort() # Sort the code book
	# Find decision boundaries
	decision_boundaries.append(0) # Insert 0 as first boundary
	i=1
	while i<len(centroids):
		decision_boundaries.append((centroids[i]+centroids[i-1])*0.5)
		i+=1
	for file in pcaps_filelist_for_training: # Add fingerprints to fp_db
		t = generateTreeFromString(parsePcapAndGetQuantizedString(file,1000))
		filename = (file.split('\\')[-1]).split('.')[0]
		fp_db.append(fingerprint(filename,t))

#===============================================================================
# showQuantizations attempts to graph the quantization steps nicely.
#===============================================================================
def showQuantizations():
	X = range(len(centroids))
	labels = [ chr(97+val) for val in X ]
	font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 16,
    }
	fig = pl.figure()
	ax = fig.add_subplot(111)
	pl.title('Codebook quantization values', fontdict=font)

	transOffset = offset_copy(ax.transData, fig=fig, x = 0.08, y=-0.20, units='inches')
	for x, y in zip(X, centroids):
		pl.plot((x,),(y,), 'ro')
		pl.text(x, y, ('%2.4f' % y), transform=transOffset)

	# You can specify a rotation for the tick labels in degrees or with keywords.
	pl.xlabel('Quantized value')
	pl.xticks(X, labels)
	# Pad margins so that markers don't get clipped by the axes
	pylab.ylabel('Time [Sec]')
	pl.margins(0.2)
	# Tweak spacing to prevent clipping of tick-labels
	pl.subplots_adjust(bottom=0.15)
	pl.show()

#===============================================================================
# trainingCallback is the method called by tkinter to start the training. It gets
# the listbox selected directories and starts the training
#===============================================================================
def trainingCallback():
	global is_first_time, tkc
	if len(tkc.directorylb.curselection())<1:
		tkc.statusString.set("You haven't picked any directories")
		return
	clearGlobals()
	items = map(int,tkc.directorylb.curselection())
	l = []
	for i in items:
		l.append(list_of_dirs[i]) #.replace('/','\\'))
	training(l,int(tkc.sb.get()))
	tkc.statusString.set("Done training!")
	if is_first_time==True:
		is_first_time=False
		tkinter.Button(tkc.window, text="  Show quantizations  ", command=showQuantizations).grid(row=5,column=0, columnspan=2, sticky=tkinter.W)
		tkinter.Button(tkc.window, text="   Show histogram  ", command=showHistogram).grid(row=5,column=2, columnspan=2, sticky=tkinter.W)
		tkinter.Button(tkc.window, text="Print tests to console ", command=testCallback).grid(row=6, column=0, columnspan=2, sticky=tkinter.W)
		tkinter.Button(tkc.window, text="Show sample graph", command=showGraphCallback).grid(row=6, column=2, columnspan=2, sticky=tkinter.W)
		tkc.tb.grid(row=7,columnspan=6,rowspan=6, sticky=tkinter.W+tkinter.E+tkinter.N+tkinter.S)

def showHistogram():
	pl.figure()
	pl.hist(observed_time_vector, centroids, normed=True, histtype='bar', rwidth=1)
	X = range(len(centroids))
	labels = [ chr(97+val) for val in X ]
	pl.xlabel('Interval')
	pl.xticks(centroids, labels)
	pl.ylabel('Probability')
	pl.title('Histogram\n Mean: '+"%.4f"%np.mean(observed_time_vector)+' Variance: '+"%.4f"%np.var(observed_time_vector))
	pl.grid(True)
	pl.autoscale(True, 'both')
	pl.show()


def showGraphCallback():
	for dbi in fp_db:
		printTreeForWolfram(dbi.tree, dbi.tag)


def testFingerprintVsOne(fingerprint, testSubject, weights=[33,33,33], kldFactor=0.5 ):
	(_tf,tsd) = checkTreeShapeDiff(fingerprint, testSubject)
	tsd=tsd/fingerprint.sub_tree_size
	kld = abs(calculateKLDistance(fingerprint, testSubject, kldFactor))
	ctbl = compareTreesByLevel(fingerprint, testSubject)
	return (weights[0]*tsd+weights[1]*kld+weights[2]*ctbl)

#===============================================================================
# findFpByTag Locate a fingerprint by its tag. Return false if not found.
#===============================================================================
def findFpByTag(tag):
	for dbi in fp_db:
		if dbi.tag==tag:
			return dbi.tree
	return False

def testCallback():
	global tkc
	capturestring = parsePcapAndGetQuantizedString("Cryptlocker_Delta.pcap",5000,'test')
	capturetree = generateTreeFromString(capturestring)
	tstr = ""
	tstr += "centroids: %s\n==========================\n"%centroids
	tstr += "Decision Boundaries: %s\n==========================\n"%decision_boundaries
	for dbi in fp_db:
		window_size = dbi.tree.sub_tree_size
		print ("Window size for %s is %d"%(dbi.tag,dbi.tree.sub_tree_size))
		f,l = compareFingerprintWithCapture(dbi.tree,capturestring,window_size)
		tstr += "Comparing %s fingerprint, resulted in:\n\tLog loss %f\n\tHamming Loss %f\n\t"%(dbi.tag,f,l)
		(tf,val) = (checkTreeShapeDiff(dbi.tree, capturetree))
		val=val/dbi.tree.sub_tree_size
		tstr += ("Tree-shape: (%s,%s)\n\t")%(tf,val)
		tstr += ("Tree-distance (k=0.05): %f\n\t")%(calculateKLDistance(dbi.tree, capturetree, 0.05))
		ctbl = compareTreesByLevel(dbi.tree, capturetree)
		tstr += ("Tree compareByLevel: %f \n\n")%(ctbl)
	tkc.tb.insert(tkinter.INSERT,tstr)

#===============================================================================
# Main entry point into the program
#===============================================================================
if __name__ == "__main__":
	global tkc
	tkc = tkstuff()
	tkc.window.title("Traffic Fingerprinting")
	tkc.window.geometry("800x600")
	tkc.window.wm_iconbitmap('fingerprint.ico')
	tkinter.Label(tkc.window,text="Pick directories to perform training on from below: ").grid(row=0, column=0, columnspan=4, sticky=tkinter.W+tkinter.E)
	i=0
	scrollbar = tkinter.Scrollbar(tkc.window)
	tkc.directorylb.config(yscrollcommand=scrollbar.set)

	for x in os.walk('.'):
		if not "git" in x[0] and not "prerequisites" in x[0] and not ".eclipse" in x[0] and not "_pycache_" in x[0]:
			list_of_dirs.append(x[0][2:])
			tkc.directorylb.insert(i,x[0][2:])
			i+=1
	tkc.directorylb.grid(row=1, column=1, columnspan=2, padx=2)#pack(side=tkinter.TOP)
	scrollbar.grid(row=1,column=0, sticky=tkinter.N+tkinter.S)
	tkinter.Label(tkc.window,text="Codebook size ",font=("Arial", 12),fg="#008000").grid(row=2,column=0, columnspan=3, sticky=tkinter.E)#pack(side=tkinter.TOP)
	tkc.sb.config(width=4)
	tkc.sb.grid(row=2, column=3, sticky=tkinter.W)#pack(side=tkinter.TOP)

	scrollbar.config(command=tkc.directorylb.yview)

	tkinter.Button(tkc.window, text="Start Training", command=trainingCallback).grid(row=3,columnspan=4)#pack(side=tkinter.BOTTOM)
	tkinter.Label(tkc.window,textvariable=tkc.statusString,font=("Arial", 16),fg="#000080").grid(row=4,columnspan=4)# pack()
	tkc.statusString.set("Ready...")
	tkinter.mainloop()
