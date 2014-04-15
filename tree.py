import os # For starting up Mathematica
from subprocess import call
import dpkt # Wireshark parsing
import math
import difflib # Finding closest strings
from nt import getcwd # Get current working directory
from scipy.cluster.vq import kmeans, whiten # Kmeans clustering
import tkinter # UI

hashmap = {}
observed_vector = []
file_list = []
codebook = []
decision_boundaries = []
list_of_dirs = []
tkwindow = tkinter.Tk() # Start up TK
lb = tkinter.Listbox(tkwindow, height=10, width=300, selectmode=tkinter.MULTIPLE)
	
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
	tkwindow = node('tkwindow', 0, [])  # Create new tkwindow item with probability 0
	[ insertToTree(tkwindow, item) for item in listToInsert ]  # Insert items one by one
	return tkwindow
		
def insertToTree(tkwindow, tag):
	newitem = (node(tag, 0.0, []), edgeprob(0.0))  # New tuple (Node,Probability of edge)
	son = 0  # For preventing the last "if" clause
	
	if isLeaf(tkwindow) == True:  # Empty subtree - first tag being inserted
		tkwindow.sub_tree.append(newitem)  # Place in the node's children
		tkwindow.sub_tree_size = 1
		tkwindow.sub_tree.sort(key=lambda sub_tree_item: sub_tree_item[0].name)
		
	else:  # The subtree isn't empty
		son = findAppSon(tkwindow.sub_tree, tag)  # Find if there exists a child with the correct tag
		if son != -1:  # It exists, so call this method with that child as the tkwindow
			tkwindow.sub_tree_size += 1
			insertToTree(son[0], tag)
			
	if isLeaf(tkwindow) == False and son == -1:  # No child was found and the tkwindow isn't a leaf, so add it to the tkwindow.
		tkwindow.sub_tree.append(newitem)
		tkwindow.sub_tree_size += 1
		tkwindow.sub_tree.sort(key=lambda sub_tree_item: sub_tree_item[0].name)
	
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
# countLeavesTree counts leaves in a tree, given its tkwindow.
#===============================================================================
def countLeavesTree(tkwindow):
	l = [ countLeaves(son) for son in tkwindow.sub_tree ]
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
#
#===============================================================================
def findLeavesTree(tkwindow):

	l = [ findLeaves(son) for son in tkwindow.sub_tree ]
	return l

#===============================================================================

#===============================================================================
def findLeaves(node_tuple):
	if isLeaf(node_tuple[0]) == True:
		return [node_tuple[0]]
	else:
		return findLeavesTree(node_tuple[0])

#===============================================================================
#
#===============================================================================
def findMaxDepthTree(tkwindow):

	l = [ findMaxDepthLeaves(son) for son in tkwindow.sub_tree ]
	return max(l)

#===============================================================================

#===============================================================================
def findMaxDepthLeaves(node_tuple):
	if isLeaf(node_tuple[0]) == True:
		return 1
	else:
		return 1+findMaxDepthTree(node_tuple[0])
	
#===============================================================================
# setProbTree sets probabilities for the nodes (full probability for event)
# Accepts a tree tkwindow and the number of leaves (this is for uniform probability for the leaves)
#===============================================================================
def setProbTree(tkwindow, num_leaves):
	l = [ setProbLeaves(son, num_leaves) for son in tkwindow.sub_tree ]
	tkwindow.prob = sum(l)
	return sum(l)

#===============================================================================
# setProbLeaves sets probabilities for the rest of the nodes that aren't the tkwindow
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
# setProbEdgesTree Sets probabilities for the conditional probabilities (edges) for the tkwindow
#===============================================================================
def setProbEdgesTree(tkwindow):
	for son in tkwindow.sub_tree:
		son[1].prob = son[0].prob  # For every son of the ROOT ONLY, set probability of edge to probability of son
		setProbEdgesLeaves(son)  # Then start setting probabilities for all sons, from the top down
	return sum

#===============================================================================
# setProbEdgesLeaves Sets probabilities for the conditional probabilities (edges) for the non-tkwindow
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
# printTreeForWolfram prints a tree nicely for WolframMathematica to display
#===============================================================================
def printTreeForWolfram(tkwindow, name, start=True, f=None):
	c = ','  # The c variable is used to prevent adding of extra commas when not necessary
	if(start == True):
		f = open(name + '.nb', 'w')
		s = "g={"
		f.write(s)
		c = ''
	for son in tkwindow.sub_tree:
		s = c + "{\"%s | %d | %f\" -> \"%s | %d | %f\" , \"%f\" }" % (tkwindow.name, tkwindow.sub_tree_size, tkwindow.prob, son[0].name, son[0].sub_tree_size, son[0].prob, son[1].prob)
		f.write(s)
		if c == '':
			c = ','
		printTreeForWolfram(son[0], name, False, f)
	if(start == True):
		s = "};\nTreePlot[g, Automatic, \"tkwindow | %d | %f\", VertexLabeling -> True]" % (tkwindow.sub_tree_size, tkwindow.prob)
		f.write(s)
		f.close()
		call(["c:\Program Files\Wolfram Research\Mathematica\9.0\Mathematica.exe", os.getcwd() + "/" + name + ".nb"])


#===============================================================================
# findAllPcapFiles creates a list of pcap files in the current directory, to
# be stored in the global variable 'file_list'
#===============================================================================
def findAllPcapFiles(wd):
	for file in os.listdir(getcwd()+"/"+wd+"/"):
		if file.endswith(".pcap"):
			file_list.append(file)
		
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
	f = open(getcwd() + "/"+wd+"/" + file, "rb") # Open file
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
	whitened = whiten(observed_vector)
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
	f = open(getcwd() + "/"+wd+"/" + file, "rb")
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

def flatten(lst):
	return sum( ([x] if not isinstance(x, list) else flatten(x)
		     for x in lst), [] )

def findNodeProbabilityInList(item, node_list):
	for li in node_list:
		if li.name == item.name:
			return li.prob
	return -1

# def findClosestProbabilityInList(itemname, node_list):
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


def findClosestProbabilityInList(itemname, node_list):
	name_list = [ x.name for x in node_list ] # Get list of names only
	close_matches = difflib.get_close_matches(itemname, name_list,len(name_list),0.0) # Find 5 closest matches
	sums = [ (lambda x: sum([ ord(y) for y in x ]))(x) for x in close_matches] # Generate their sums
	itemname_sum = sum([ ord(x) for x in itemname ]) # Generate our search-item's sum
	closest = min(sums, key=lambda x:abs(x-itemname_sum)) # Find closest entry from the 3 found
	index_in_sums_list = sums.index(closest)  # Find the index in the original list
	return node_list[name_list.index(close_matches[index_in_sums_list])].prob # Get probability
	
	
def calculateKLDistance(tree1, tree2, factor=0.5):
	p = flatten(findLeavesTree(tree1)) # First list
	q = flatten(findLeavesTree(tree2))
	
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
			qi_prob = findClosestProbabilityInList(node_item.name, q) * factor
		
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
			
	
	# print("The distance between the two trees with %d errors is:\t%f" %(counter,kl_distance_sum))
	return kl_distance_sum

def checkTreeShapeDiff(tree1,tree2):
	# diff = tree1.sub_tree_size - tree2.sub_tree_size
	if len(tree1.sub_tree)==0 and len(tree2.sub_tree)==0:
		return (True,0)
	else:
		subtree_sum = 0
		t_or_f = True
		for i in range(max(len(tree1.sub_tree),len(tree2.sub_tree))):
			try:
				(returned_t_or_f, returned_subtree_sum) = checkTreeShapeDiff(tree1.sub_tree[i][0], tree2.sub_tree[i][0])
				subtree_sum += returned_subtree_sum
				t_or_f *= returned_t_or_f
			except IndexError:
				t_or_f = False
				if len(tree1.sub_tree)<=i:
					# Root2 has more childen, so subtree_sum up all of tree2's subtrees
					subtree_sum += 1+ (tree2.sub_tree[i][0]).sub_tree_size # Get this child's subtree size
				else:
					# Root1 has more children so subtree_sum up all of tree1's subtrees
					subtree_sum += 1+ (tree1.sub_tree[i][0]).sub_tree_size
		return (t_or_f, subtree_sum)
				
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

def training(dir_list = ['training_1']):
	for wd in dir_list:
		print("Gonna scan dir %s now" % wd)
		findAllPcapFiles(wd) # Get list of pcap files
		collectAllRelativeTimestamps(wd)
		
	createLloydMaxCodebook(8) # Create code book
	codebook.sort() # Sort the code book
	# Find decision boundaries
	decision_boundaries.append(0) # Insert 0 as first boundary
	last_value = 0;
	for value in codebook:
		decision_boundaries.append((value + last_value) / 2) # Boundary is between two centroids in the codebook
		last_value = value 

def trainingCallback():
	items = map(int,lb.curselection())
	l = []
	for i in items:
		l.append(list_of_dirs[i])
	training(l)

if __name__ == "__main__":
	tkwindow.title("Traffic Fingerprinting")
	tkwindow.geometry("640x480")
	tkwindow.wm_iconbitmap('fingerprint.ico')
	
	i=0
	for x in os.walk('.'):
		list_of_dirs.append(x[0])
		lb.insert(i,x[0])
		i+=1
	lb.pack()
	tkinter.Button(tkwindow, text="Training", command=trainingCallback).pack()
	tkinter.mainloop()

	#t = []
	#t.append(generateTreeFromString(parsePcapAndGetQuantizedString("bbc1.pcap",'training_1',100)))
	#t.append(generateTreeFromString(parsePcapAndGetQuantizedString("bbc2.pcap",'pcaps',100)))
#	
	#t.append(generateTreeFromString(parsePcapAndGetQuantizedString("skyp.pcap",'pcaps',100)))
	#t.append(generateTreeFromString(parsePcapAndGetQuantizedString("skyp2.pcap",'pcaps',100)))
#	
	#for i in range(4):
		#for j in range(4):
			#print("Tree-shape: Tree %d vs. tree %d: %s" % (i,j,checkTreeShapeDiff(t[i], t[j])))
			#print("Tree-distance (k=0.5): Tree %d vs. tree %d: %s" % (i,j,calculateKLDistance(t[i], t[j], 0.5)))
	# print(findMaxDepthTree(t))
	
	
	
	
