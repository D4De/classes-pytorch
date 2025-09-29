#https://manalelaidouni.github.io/manalelaidouni.github.io/Evaluating-Object-Detection-Models-Guide-to-Performance-Metrics.html

import sys
import os
import math

#threshold for the Jaccard index
JACCARD_THR = 0.5

#compute area. it takes 4 int values being the coordinates of x_min, y_min, x_max, y_max
def computeArea(coords):
  if coords[2] >= coords[0] and coords[3] >= coords[1]:
    return (coords[2]-coords[0]+1)*(coords[3]-coords[1]+1)
  return 0

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python " + sys.argv[0] + " exp_bb.txt golden_bb.txt")
    sys.exit(1)
    
  #True positives, False positives and False negatives  
  TP=0 #correctly matched bb
  FP=0 #"new" wrong bb or bb not having a match with a golden bb with Jaccard value < JACCARD_THR
  FN=0 #golden bb not present in the corrupted image
  
  #read bb file of the current exp
  bb_exp = []
  f=open(sys.argv[1], 'r')
  for l in f:
    l=l[:-1] #discard '\n'
    if l!='':
      bb_exp.append(l)
  f.close()
  
  #read bb file of the golden run and start count exact matches with bb_exp. corresponding bbs are discarded
  bb_golden = []
  f=open(sys.argv[2], 'r')
  for l in f:
    l=l[:-1] #discard '\n'
    if l!='':
      #if there is a 1:1 match with a box for the current exp, 
      #then discard the pair from the two sets and TP++.
      #otherwise add it to the golden list
      if bb_exp.count(l) > 0:
        TP = TP + 1
        bb_exp.remove(l)
      else:
        bb_golden.append(l)
  f.close()
    
  #cast the string of each golden bb in a list of 4 integer values (x_min, y_min, x_max, y_max)
  bb_golden1 = []
  for l in bb_golden:
    bb = l.split(' ')
    bb.pop(2) #remove the arrow from the line
    bb_golden1.append([int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])])

  #check all Jaccard indexes of all bb_exp x bb_golden1
  matched_bbe = 0
  for bbe in bb_exp:
    bbe = bbe.split(' ')
    bbe.pop(2)
    #cast the string of the bb in a list of 4 integer values (x_min, y_min, x_max, y_max)
    bbe = [int(bbe[0]), int(bbe[1]), int(bbe[2]), int(bbe[3])]
    #we need to find the pair of bbs one of the golder run and one of the current exp maximising the Jaccard index
    max_jaccard = None
    max_bbg = None
    for bbg in bb_golden1:
      #compute intersection coords
      inters_coord=[]
      if bbe[0]>bbg[0]:
        inters_coord.append(bbe[0])
      else:
        inters_coord.append(bbg[0])
      if bbe[1]>bbg[1]:
        inters_coord.append(bbe[1])
      else:
        inters_coord.append(bbg[1])
      if bbe[2]<bbg[2]:
        inters_coord.append(bbe[2])
      else:
        inters_coord.append(bbg[2])
      if bbe[3]<bbg[3]:
        inters_coord.append(bbe[3])
      else:
        inters_coord.append(bbg[3])
      #compute intersect and union areas, then the JAccard index
      intersArea = computeArea(inters_coord)
      unionArea = computeArea(bbe) + computeArea(bbg) - intersArea
      jaccard = float(intersArea)/unionArea
      #take the maximum Jaccard index
      if max_jaccard == None or jaccard > max_jaccard:
        max_jaccard = jaccard
        max_bbg = bbg
    #if we have found a pair and the Jaccard index is higher than JACCARD_THR we have a match 
    #(i.e. TP++ - we use another variable to count just for the sake of clarity. we add to TP later)
    if max_jaccard != None and max_jaccard >= JACCARD_THR:
      bb_golden1.remove(max_bbg) #it is not necessary to remove from bb_exp since we are scanning such list
      matched_bbe = matched_bbe + 1
  #compute final stats
  TP = TP + matched_bbe # we add to TP the count of bb pairs having jaccard value >= JACCARD_THR
  FP = len(bb_exp) - matched_bbe #the number of "new" wrong bbs is equal to the remaining bbs in bb_exp 
  FN = len(bb_golden1) #the number of not detected golden bbs is equal to the size of bb_golden1
  print("TP: " + str(TP) + " FP: " + str(FP) + " FN: " + str(FN))
