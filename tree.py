import os # For starting up Mathematica
from subprocess import call
import dpkt # Wireshark parsing
import math
import difflib # Finding closest strings
from nt import getcwd # Get current working directory
from scipy.cluster.vq import kmeans, kmeans2, whiten # Kmeans clustering
import tkinter # UI
import numpy as np # Number methods (average, median, stdev,...
from matplotlib.transforms import offset_copy # Graphs
import matplotlib.pyplot as pl # Graphs
import pylab # Graphs
import networkx as nx # Graphs

hashmap = {}
observed_vector = []
file_list = []
codebook = []
decision_boundaries = []
list_of_dirs = []
tkwindow = tkinter.Tk() # Start up TK
statusString = tkinter.StringVar()
isFirstTime = True
lb = tkinter.Listbox(tkwindow, height=6, width=200, selectmode=tkinter.MULTIPLE)
var = tkinter.StringVar(tkwindow) # Hack begins
sb = tkinter.Spinbox(tkwindow, from_=1, to=15,textvariable=var)
var.set("8") # Stupid dirty hack ends. Sets the default codebook value to 8.
G = nx.DiGraph()


def clearGlobals():
	hashmap.clear()
	observed_vector.clear()
	#file_list.clear()
	codebook.clear()
	decision_boundaries.clear()
	#list_of_dirs.clear()
	
#===============================================================================
# generateTreeFromString generates a tree from string obviously
#===============================================================================
def generateTreeFromString(string):
	hashmap.clear()
	getLZList(string)  # Generates a hashtable for all variants, stored in 'hashmap'
	l = sorted(hashmap)  # Sort hashmap and get it as a nice list
	t = buildTree(l)  # Build a tree
	setProbTree(t, countLeavesTree(t))  # Set the probability for the tree nodes 
	setProbEdgesTree(t)  # Set the probability for the tree edges
	# printTreeForWolfram(t, name)  # Print out the tree for export to Wolfram Mathematica
	return t
	
#===============================================================================
# Creates a Lempel Ziv style hashmap for a given string, of any alphabet
# Stores result in a global variable called hashmap. Recursive method
#===============================================================================
def getLZListR(head, tail):
	global hashmap
	if len(tail) == 0 and len(head) == 0:  # Empty list
		return 
	if head in hashmap:
		hashmap[head] += 1  # This item exists. Add one to its counter
		if len(tail) == 0:
			return
		else:  # And if this is not the final letter, try the next combination (recurse)
			getLZListR(head + tail[0], tail[1:]) 
	else:
		hashmap[head] = 1  # Item wasn't in hashmap, so its brand new entry....
		if len(tail) == 0:
			return  # No more letters to check
		if len(tail) == 1:  # And there are still more letters to go, so recurse (only one more letter)
			getLZListR(tail[0], '')
		else:  # And there are still more letters to go, so recurse (more than one letter)
			getLZListR(tail[0], tail[1:])
			
#===============================================================================
# Creates a Lempel Ziv style hashmap for a given string, of any alphabet
# Stores result in a global variable called hashmap. Iterative method
#===============================================================================
def getLZList(string):
	flag = True
	tempstr = ''
	start = 0
	end = 1
	while(end <= len(string) and flag == True):
		tempstr = string[start:end]
		if tempstr in hashmap:
			hashmap[tempstr] += 1
			if end >= len(string):
				flag = False
				return
			else:  # More letters to come
				end += 1
		else:  # Item isn't in hashmap
			hashmap[tempstr] = 1  # New entry
			if end == len(string):
				flag = False
				return
			else:  # More letters to check
				start = end
				end += 1
	
#===============================================================================
# node is a tree node. It contains a name, probability and a list of sons  
#===============================================================================
class node:
	def __init__(self, name, prob, sons=[], size=0):
		self.prob = prob  # Probability ('value')
		self.name = name  # Name ('tag')
		self.sub_tree = sons  # List of children
		self.sub_tree_size = size

#===============================================================================
# Stores a float.
# This is because we miscalculated the need for updating probabilities when we
# used a tuple. This is a bypass to the immutability of a tuple. 
#===============================================================================
class edgeprob:
	def __init__(self, prob=0.0):
		self.prob = prob
	
#===============================================================================
# buildTree builds a tree given a sorted list of items (from the hashmap usually)
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
	l = [ countLeaves(son) for son in root.sub_tree ]
	return sum(l)

#===============================================================================
# countLeaves is used by countLeavesTree, but also uses countLeavesTree. See what
# we did there?
#===============================================================================
def countLeaves(node_tuple):
	if isLeaf(node_tuple[0]) == True:
		return 1
	else:
		return countLeavesTree(node_tuple[0])

#===============================================================================
# Finds the leaves in a tree. Returns a list of the leaves.
#===============================================================================
def findLeavesTree(root):

	l = [ findLeaves(son) for son in root.sub_tree ]
	return flatten(l)

#===============================================================================
# Method called by above findLeavesTree
#===============================================================================
def findLeaves(node_tuple):
	if isLeaf(node_tuple[0]) == True:
		return [node_tuple[0]]
	else:
		return findLeavesTree(node_tuple[0])

#===============================================================================
# Finds the maximum depth in a tree and returns that integer
#===============================================================================
def findMaxDepthTree(root):

	l = [ findMaxDepthLeaves(son) for son in root.sub_tree ]
	return max(l)

#===============================================================================
# Method used by findMaxDepthTree
#===============================================================================
def findMaxDepthLeaves(node_tuple):
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
# Finds the leaves in a tree. Returns a list of the leaves.
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


def printTreeWithNetworkX(root,start=True):
	for son in root.sub_tree:
		G.add_edge("%s | %d | %f" % (root.name, root.sub_tree_size, root.prob),"%s | %d | %f" % (son[0].name, son[0].sub_tree_size, son[0].prob), weight = son[1].prob)
		printTreeWithNetworkX(son[0],False)
	
	if(start == True):
		elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.1]
		esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.1]
		
		pos=nx.spectral_layout(G,2,'weight',5) # positions for all nodes
		
		# nodes
		nx.draw_networkx_nodes(G,pos,alpha=0.9,node_color='#A5A5A5')
		# edges
		#nx.draw_networkx_edges(G,pos,edgelist=elarge,width=2)
		nx.draw_networkx_edges(G,pos,alpha=0.7,edgelist=esmall+elarge,width=2,edge_color=range(len(esmall+elarge)),edge_cmap=pl.cm.Blues)
		# labels
		nx.draw_networkx_labels(G,pos,font_size=8,font_family='sans-serif',font_color='b')
		
		pl.axis('off')
		pl.savefig("weighted_graph.png") # save as png
		pl.show() # display'''
		
		'''nx.write_dot(G,'test.dot')
		pl.title("draw_networkx")
		pos = nx.graphviz_layout(G, prog='dot', args="-Grankdir=LR")
		nx.draw(G,pos,with_labels=False,arrows=False)
		pl.savefig('nx_test.png')'''
		



#===============================================================================
# findAllPcapFiles creates a list of pcap files in the current directory, to
# be stored in the global variable 'file_list'
#===============================================================================
def findAllPcapFiles(wd):
	for file in os.listdir(os.path.join(getcwd(),wd)):
		if file.endswith(".pcap"):
			file_list.append(os.path.join(getcwd(),wd,file))
		
#===============================================================================
# collectAllRelativeTimestamps calls collectRelativeTimestampsForSingleFile for
# every file in the file list
#===============================================================================
def collectAllRelativeTimestamps(wd):
	for file in file_list:
		collectRelativeTimestampsForSingleFile(file,wd)

#===============================================================================
# collectRelativeTimestampsForSingleFile generates relative time differences for a single file
# Saved in global variable observed_vector
#===============================================================================
def collectRelativeTimestampsForSingleFile(file,wd):
	f = open(file, "rb") # Open file
	pcapReader = dpkt.pcap.Reader(f) # Parse
	frame_counter = 0
	last_time = 0
	for ts, _buf in pcapReader:
		frame_counter += 1
		if frame_counter > 1 :
			observed_vector.append(ts - last_time) # Calculate relative timestamp and save
		else:
			observed_vector.append(0) # First frame should have 0 as a timestamp
		last_time = ts
	f.close()
#===============================================================================
# createLloydMaxCodebook creates a quantized vector, with "num_codes" codes in the codebook
# based on the Lloyd Max quantizing scheme. The data is based on the observed_vector
# observation vector.
#===============================================================================
def createLloydMaxCodebook(num_codes):
	whitened = np.array(observed_vector) #whiten(observed_vector)
	#(vec, _error) = kmeans2(whitened, num_codes, iter=20, thresh=1e-05, minit='random')
	(vec, _error) = kmeans(whitened, num_codes, 20, 1e-05)  # 10 is number of codes, 20 is number of iteration, 1e-05 is the error we want
	for value in vec:
		codebook.append(value) # Append centroid value to the list
	

#===============================================================================
# getCodeFromCodebook gets a code for a specific value from the codebook. Returns a char
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
def parsePcapAndGetQuantizedString(file,wd,max_len=0):
	f = open(os.path.join(getcwd(),wd,file), "rb")
	pcapReader = dpkt.pcap.Reader(f)
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

# def smoothKLContinuity(itemname, node_list):
# 	name_list = [ x.name for x in node_list ]
# 	close_matches = difflib.get_close_matches(itemname, name_list,3,0.3)
# 	# ratios = [ difflib.SequenceMatcher(None, itemname, x).ratio() for x in close_matches ]
# 	sums = [ (lambda x: sum([ ord(y) for y in x ]))(x) for x in close_matches]
# 	itemname_sum = sum([ ord(x) for x in itemname ])
# 	closest = min(sums, key=lambda x:abs(x-itemname_sum))
# 	index_in_sums_list = sums.index(closest)
# 	r_prob = node_list[name_list.index(close_matches[index_in_sums_list])].prob
# 	print("Didn't find %s, matching with %s (prob=%f) instead" %(itemname,close_matches[index_in_sums_list],r_prob))
# 	return r_prob


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
	p = genListOfNodes(tree1) #findLeavesTree(tree1) # First list
	q = genListOfNodes(tree2) #findLeavesTree(tree2) # Second list
	# TODO: I altered the above, because I think we may need to do this on all nodes instead of just the leaves.
	# 
	
	if len(p)>len(q):
		temp = p
		p = q
		q = temp

	kl_distance_sum = 0
	counter = 0
	
	for node_item in p:
		qi_prob = findNodeProbabilityInList(node_item, q)
		if qi_prob == -1:
			counter += 1
			qi_prob = smoothKLContinuity(node_item.name, q) * factor
		kl_distance_sum += (math.log(node_item.prob / qi_prob) * node_item.prob)
#		else:
			# If Q contains a zero, we want to handle it in a specific manner. Because KL distance
			# Is undefined for cases in which the distribution is not continuous. Therefore, we will define
			# a constant/variable which will allow us to give 'weight' to such cases in which one distribution/tree
			# doesn't contain the values. This will allow us to alter our misdetection / false alarm rates.
#			counter += 1
			# Factor is the error factor to multiply by the counter. This way, more errors will cause a larger deviation
			#kl_distance_sum += (counter/len(p))*factor
#			kl_distance_sum += node_item.prob * factor	
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
				
def quantDist(left, right): # Always between 0 and 1, unless error: then -1.
	try:
		entirerange = codebook[-1]-codebook[0]
		return abs(codebook[ ord(left)-97 ] - codebook[ ord(right)-97 ])/entirerange # Quantization distance is the distance between two centroids
	except IndexError:
		return -1		



### TODO:
###  1. Normalize the values ('average' them out so that they total 1 per level
###  2. Pretty sure compareTreesByLevel is redundant for now.
###  3. Check obviously. Verify results
###  4. Build some sort of testing framework - so that we can run many tests at once with varying vlaues of factors and length of files and so on.
###  5. UI - to make things look nice. 
###  6. Find string in tree

def compareChildrenOnLevel(tree1, tree2): # Tree1 should be deeper than Tree2
	smallest = 2
	total = 0
	j=0
	for tree1node in tree1.sub_tree:
		for tree2node in tree2.sub_tree:
			smallest = min(smallest, quantDist(tree1node[0].name, tree2node[0].name))
			j += compareChildrenOnLevel(tree1node[0], tree2node[0])
		j += min(smallest,1)
	total += j
	return total
	

def compareTreesByLevel(tree1, tree2):
	total = 0
	total += compareChildrenOnLevel(tree1,tree2)
	for c1 in tree1.sub_tree:
		for c2 in tree2.sub_tree:
			total += compareTreesByLevel(c1[0], c2[0])
	
	return total

def compareTwoTrees(tree1,tree2):
	if findMaxDepthTree(tree2)>findMaxDepthTree(tree1): # Ensure tree1 is deeper
		temp = tree2
		tree2 = tree1
		tree1 = temp
	return compareTreesByLevel(tree1,tree2)

#===============================================================================
# training looks in a directory list (should be specified), and for each directory
# it scans the PCAP files and collects timestamps. Afterwards, it runs
# the LloydMax generator method, and sets decisiou boundaries. 
# The second parameter in the function is the size of the codebook. (Default: 8)
#===============================================================================
def training(dir_list = ['training_1'], codebook_size = 8):
	for wd in dir_list:
		findAllPcapFiles(wd) # Get list of pcap files
		collectAllRelativeTimestamps(wd)
		
	createLloydMaxCodebook(codebook_size) # Create code book
	codebook.sort() # Sort the code book
	# Find decision boundaries
	decision_boundaries.append(0) # Insert 0 as first boundary
	last_value = 0;
	for value in codebook:
		decision_boundaries.append((value + last_value) / 2) # Boundary is between two centroids in the codebook
		last_value = value 

#===============================================================================
# showQuantizations attempts to graph the quantization steps nicely.
#===============================================================================
def showQuantizations():
	X = range(len(codebook))
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
	for x, y in zip(X, codebook):
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
	global isFirstTime
	if len(lb.curselection())<1:
		statusString.set("You haven't picked any directories")
		return
	clearGlobals()
	items = map(int,lb.curselection())	
	l = []
	for i in items:
		l.append(list_of_dirs[i]) #.replace('/','\\'))
	training(l,int(sb.get()))
	statusString.set("Done training!")
	if isFirstTime==True:
		isFirstTime=False
		tkinter.Button(tkwindow, text="Show quantizations", command=showQuantizations).pack()
		tkinter.Button(tkwindow, text="Show histogram", command=showHistogram).pack()
		tkinter.Button(tkwindow, text="Print tests to console", command=testCallback).pack()
		tkinter.Button(tkwindow, text="Show a sample graph (Godlion/Kerogod phishing virus)", command=showGraphCallback).pack()

def showHistogram():
	pl.figure()
	pl.hist(observed_vector, codebook, normed=True, histtype='bar', rwidth=1)
	X = range(len(codebook))
	labels = [ chr(97+val) for val in X ]
	pl.xlabel('Interval')
	pl.xticks(codebook, labels)
	pl.ylabel('Probability')
	pl.title('Histogram\n Mean: '+"%.4f"%np.mean(observed_vector)+' Variance: '+"%.4f"%np.var(observed_vector))
	pl.grid(True)
	pl.autoscale(True, 'both')
	pl.show()

	
def showGraphCallback():
	t = generateTreeFromString(parsePcapAndGetQuantizedString("Internet Bank Phishing - ActiveX_kerogod-godlion_FULL.pcap",'test',100))
	printTreeWithNetworkX(t)

def testCallback():
	t = []
	t.append(generateTreeFromString(parsePcapAndGetQuantizedString("Internet Bank Phishing - ActiveX_kerogod-godlion.pcap",'test',1000)))
	t.append(generateTreeFromString(parsePcapAndGetQuantizedString("Internet Bank Phishing - ActiveX_kerogod-godlion_with_more.pcap",'test',1000)))
	t.append(generateTreeFromString(parsePcapAndGetQuantizedString("Internet Bank Phishing - ActiveX_kerogod-godlion_FULL.pcap",'test',1000)))
	
	for i in range(len(t)):
		for j in range(len(t)):
			if j != i:
				print("Tree-shape: Tree %d vs. tree %d: %s" % (i,j,checkTreeShapeDiff(t[i], t[j])))
				print("Tree-distance (k=0.5): Tree %d vs. tree %d: %s" % (i,j,calculateKLDistance(t[i], t[j], 0.9)))
	

#===============================================================================
# Main entry point into the program
#===============================================================================
if __name__ == "__main__":
	tkwindow.title("Traffic Fingerprinting")
	tkwindow.geometry("640x480")
	tkwindow.wm_iconbitmap('fingerprint.ico')
	tkinter.Label(tkwindow,text="Pick directories to perform training on from below: ").pack()
	i=0
	scrollbar = tkinter.Scrollbar(tkwindow)
	scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
	lb.config(yscrollcommand=scrollbar.set)
	
	for x in os.walk('.'):
		if not "git" in x[0] and x[0] != ".":
			list_of_dirs.append(x[0][2:])
			lb.insert(i,x[0][2:])
			i+=1
	lb.pack(side=tkinter.TOP)
	tkinter.Label(tkwindow,text="Codebook size ",font=("Arial", 12),fg="#008000").pack(side=tkinter.TOP)
	sb.config(width=5)
	sb.pack(side=tkinter.TOP)

	scrollbar.config(command=lb.yview)
	
	
	tkinter.Button(tkwindow, text="Start Training", command=trainingCallback).pack(side=tkinter.BOTTOM)
	tkinter.Label(tkwindow,textvariable=statusString,font=("Arial", 16),fg="#000080").pack()
	statusString.set("Ready...")
	tkinter.mainloop()
