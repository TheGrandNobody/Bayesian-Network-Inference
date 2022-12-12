import sys, os
sys.path.append(os.path.abspath(os.path.join('tools')))
from BNReasoner import BNReasoner
import pandas as pd
from pandas.util.testing import assert_frame_equal

if __name__ == "__main__": 
    bn = BNReasoner("testing/lecture_example.BIFXML")

    # Test: Network Pruning
    graph = bn.network_prune("Wet Grass?", {"Winter?": True, "Rain?": False})

    # Test 1: Correct Node+Edge Pruning
    print("Test 1 (Lecture 3 examples): Edge+Node Pruning with Query: Wet Grass and evidence: (Winter = true, Rain = false)\n")
    # Test 1a: Check the graph's nodes
    print("Should only leave 4 nodes: Winter, Sprinkler, Rain, and Wet Grass: ")
    assert list(graph.structure.nodes()) == ["Winter?", "Sprinkler?", "Rain?", "Wet Grass?"], "Incorrect nodes"
    print("Correct nodes\n")
    # Test 1b: Check the graph's edges
    print("Should only leave one edge: Sprinkler -> Wet Grass: ")
    assert list(graph.structure.edges()) == [("Sprinkler?", "Wet Grass?")], "Incorrect edges"
    print("Correct edges\n")
    graph.draw_structure()
    print("Test 1 Complete\n")
    input("Press Enter to continue...")

    # Test: d-Separation/Independence
    bn1 = BNReasoner("testing/earthquake.BIFXML")

    # Test 2a: Correct independence inference (using d-Separation) with list of Y/X
    print("Test 2 (Lecture 2 examples): d-Separation/Independence \n")
    print("Call should be independent/d-separated of/by {Burglary, Earthquake, Radio} given Alarm: ")
    assert bn1.independent('Call', ['Burglary', 'Earthquake', 'Radio'], 'Alarm'), "Incorrect independence with multiple X/Y items"
    print("Correct")
    print("Radio should be independent of {Alarm, Burglary, Call} given Earthquake: ")
    assert bn1.independent('Radio', ['Burglary', 'Alarm', 'Call'], 'Earthquake'), "Incorrect independence with multiple X/Y items"
    print("Correct Independence/d-Separation with multiple items in either X or Y \n")
    # Test 2b: Correct independence inference (using d-Separation) with list of Z
    print("Alarm should be independent of {Radio} given {Burglary, Earthquake}: ")
    assert bn1.independent('Alarm', 'Radio', ['Burglary', 'Earthquake']), "Incorrect independence with multiple Z items"
    print("Correct Independence/d-Separation multiple items in Z\n")
    # Test 2c: Correct independence inference (using d-Separation) with empty set
    print("Burglary should be independent of {Earthquake, Radio} given the empty set: ")
    assert bn1.independent('Burglary', ['Earthquake', 'Radio'], []), "Incorrect independence with empty set"
    print("Correct")
    print("Earthquake should be independent of {Burglary} given the empty set: ")
    assert bn1.independent('Earthquake', 'Burglary', []), "Incorrect independence with empty set"
    print("Correct Independence/d-Separation with empty set\n")
    print("Test 2 Complete\n")
    input("Press Enter to continue...")

    # Test: Marginalization

    # Test 3: Correct marginalization
    print("Test 3 (Lecture 3 examples): Marginalization \n")
    result = bn.marginalize("Sprinkler?", bn.bn.get_cpt("Wet Grass?"))
    result = bn.marginalize("Wet Grass?", result)
    # Test 3a: Correct marginalization with multiple variables
    print("Marginalizing Wet Grass and Sprinkler from (Wet Grass | Rain and Sprinkler) should yield\n a table with both True/False probabilities equal to 2 for Rain: ")
    assert result['p'][1] == 2 and result['p'][0] == 2, "Incorrect marginalization with multiple variables"
    print("Correct marginalization with multiple variables\n")
    result = bn.marginalize("Rain?", result)
    # Test 3b: Correct marginalization with a single variable
    print("Marginalizing Rain, Wet Grass and Sprinkler from (Wet Grass | Rain and Sprinkler) should yield\n a table with the Tautology sign and a probability of 4: ")
    assert result['p'][0] == 4, "Incorrect marginalization with a single variable"
    print(result)
    print("Correct marginalization with a single variable\n")
    print("Test 3 Complete\n")
    input("Press Enter to continue...")

    # Test: Maxing-out

    # Test 4: Correct maxing-out
    print("Test 4 (Lecture 3 examples): Maximizing-out \n")
    result = bn.maximize("Wet Grass?", bn.bn.get_cpt("Wet Grass?"))
    # Test 4a: Correct maximizing-out one variable
    print("Maximizing out Wet Grass from (Wet Grass | Rain and Sprinkler) should yield\n a table with .95, .9, .8 and .1")
    assert_frame_equal(result.drop(columns='ext. factor Wet Grass?'), pd.DataFrame({'Sprinkler?': pd.Series([False, False, True, True]),\
       'Rain?': pd.Series([False, True, False, True]), 'p': pd.Series([1, 0.8, 0.9, 0.95])}))
    print('Correct maximizing-out with one variable\n')
    # Test 4b: Correct maximizing-out with multiple variables
    result = bn.maximize("Sprinkler?", result)
    print("Maximizing out Wet Grass and Sprinkler from (Wet Grass | Rain and Sprinkler) should yield\n a table with .95 and .8")
    assert_frame_equal(result.drop(columns=['ext. factor Sprinkler?', 'ext. factor Wet Grass?']), pd.DataFrame({'Rain?': pd.Series([False, True]), 'p': pd.Series([1, 0.95])}))
    print('Correct maximizing-out with multiple variables\n')
    # Test 4c: Maximizing-out keeps track of the extended factors
    print("Extended factors should have been tracked for previous tests: ")
    assert all(i in result.columns for i in ["ext. factor Wet Grass?", "ext. factor Sprinkler?"]), "Incorrect extended factors"
    print(result)
    print("Correct extended factor tracking/updating\n")
    # Test 4d: Correct maximizing out with only one variable left
    result = bn.maximize("Rain?", result)
    print("Maximizing out Wet Grass, Sprinkler and Rain from (Wet Grass | Rain and Sprinkler) should yield\n an empty table with a probability 1")
    assert result['p'][0] == 1, "Incorrect maximizing-out with only one variable left"
    print(result)
    print("Correct maximizing-out with only one variable left\n")
    print("Test 4 Complete\n")
    input("Press Enter to continue...")

    # Test: Factor Multiplication and Variable Elimination (...+ some more marginalization)

    # Test 5: Correct factor multiplication and variable elimination
    print("Test 5 (Lecture 3 examples): Factor Multiplication/Variable Elimination \n")
    bn2 = BNReasoner("testing/abc.BIFXML")
    # Test 5a: Correct variable elimination + factor multiplication with tables containing similar variables
    result = bn2.elim_var(['A', 'B'])
    print("Eliminating A and B to get C should yield a table with .624, .376")
    assert_frame_equal(result, pd.DataFrame({'C': pd.Series([False, True]), 'p': pd.Series([0.624, 0.376])}))
    print("Correct variable elimination/factor multiplication with similar variables\n")
    # Test 5b: Correct factor multiplication with tables containing different variables
    result = bn2.f_multiply(bn.bn.get_cpt("Winter?"), bn.bn.get_cpt("Slippery Road?"))
    print("Multiplying different variables together should still give a table with the correct probabilities: ")
    assert_frame_equal(result, pd.DataFrame({'Rain?': pd.Series([False, False, False, False, True, True, True, True]),\
       'Slippery Road?': pd.Series([False, False, True, True, False, False, True, True]),\
        'Winter?': pd.Series([False, True, False, True, False, True, False, True]), 'p': pd.Series([0.4, 0.6, 0.00, 0.00, 0.12, 0.18, 0.28, 0.42])}))
    print("Correct factor multiplication with different variables\n")
    print("Test 5 Complete\n")
    input("Press Enter to continue...")

    # Test: Ordering

    # Test 6: Correct Min-degree/Min-fill ordering heuristics
    print("Test 6 (Lecture 3/4 examples): Ordering \n")
    bn3 = BNReasoner("testing/lecture_example2.BIFXML")
    res1 = bn3.ordering("d", ['X', 'Y', 'O', 'J'])
    res2 = bn.ordering("f", ['Winter?', 'Wet Grass?', 'Rain?', 'Sprinkler?'])
    # Test 6a: Correct min-degree ordering
    print("Min-degree ordering for (X, Y, O, J) in the graph of lecture 4 should yield ['O', 'Y', 'X', 'J']: ")
    assert res1 == ['O', 'Y', 'X', 'J'], "Incorrect min-degree ordering"
    print("Correct min-degree ordering\n")
    # Test 6b: Correct min-fill ordering
    print("Min-fill ordering for (Winter, Wet Grass, Rain, Sprinkler) should yield\n ['Winter?', 'Wet Grass?', 'Sprinkler?', 'Rain?']:")
    assert res2 == ['Winter?', 'Wet Grass?', 'Sprinkler?', 'Rain?'], "Incorrect min-fill ordering"
    print("Correct min-fill ordering\n")
    print("Test 6 Complete\n")

    # Test: Marginal Distribution

    # Test 7: Correct marginal distribution computation

    # Test: MAP

    # Test 8: Correct MAP application
    print("Test 8 (Lecture 4 examples): MAP \n")
    m_a_p = bn3.m_a_p(['I', 'J'], {'O': True})
    m_a_p2 = bn3.m_a_p(['O', 'J'], {})
    # Test 8a: Correct MAP computation + extended factors with evidence
    print("MAP for (I, J) given O = true should yield a probability of 0.24272 with ext. factors {'I': True/False, 'J': False}:")
    assert_frame_equal(m_a_p, pd.DataFrame({'p': pd.Series(0.24272), 'ext. factor I': pd.Series(False), 'ext. factor J': pd.Series(False)}))
    print("Correct MAP computation and extended factors when evidence is not empty\n")
    # Test 8b: Correct MAP computation + extended factors without evidence
    print("MAP for (O, J) given no evidence should yield a probability of 0.48544 with ext. factors {'O': True, 'J': False}:")
    assert_frame_equal(m_a_p2, pd.DataFrame({'p': pd.Series(0.48544), 'ext. factor O': pd.Series(True), 'ext. factor J': pd.Series(False)}))
    print("Correct MAP computation and extended factors when evidence is empty\n")
    print("Test 8 Complete\n")
    input("Press Enter to continue...")

    # Test: MPE

    # Test 9: Correct MPE application
    print("Test 9 (Lecture 4 examples): MPE \n")
    m_p_e = bn3.m_p_e({'O': False, 'J': True})
    # Test 9a: Correct MPE computation
    print("MPE for O = false, J = true should yield a probability of 0.2304225:")
    assert m_p_e['p'][0] == 0.2304225, "Incorrect MPE computation"
    print("Correct MPE computation\n")
    # Test 9b: Correct MPE extended factors
    print("MPE for O = false, J = true should yield ext. factors {'I': False, 'Y': False, 'X': False, 'O': False, J: 'True}:")
    assert_frame_equal(m_p_e[['ext. factor I', 'ext. factor Y', 'ext. factor X', 'ext. factor O', 'ext. factor J']], pd.DataFrame({'ext. factor I': pd.Series(False), 'ext. factor Y': pd.Series(False), 'ext. factor X': pd.Series(False), 'ext. factor O': pd.Series(False), 'ext. factor J': pd.Series(True)}))
    print("Correct MPE extended factors\n")
    print(m_p_e)
    print("Test 9 Complete\n")

    print("All tests passed.")
    print("Closing unit test program...")
    exit()