import sys
sys.path.append("../")
from BNReasoner import BNReasoner

if __name__ == "__main__": 
  bn = BNReasoner("testing/lecture_example2.BIFXML")
    # Test 1: Prune a network
    print("Test 1: Prune a network")
    