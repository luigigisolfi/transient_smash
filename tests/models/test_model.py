from transient_smash.models.model import *
import unittest
import numpy as np
from scipy.stats import norm

class TestSimpleModel(unittest.TestCase):
    
    def test_simulator(self):
        """Test the simple model and its simulator function."""
        simple_model = SimpleModel()

        ### Test in the case of an integer input or an np array
        self.assertEqual(simple_model.simulator(x=1,a=1,b=1),2)
        self.assertTrue((simple_model.simulator(x=np.array([1,2,3]),a=1,b=1)==np.array([2,3,4])).all())
    
class TestNoisySimpleModel(unittest.TestCase):

    def test_simulator_with_noise(self):
        """Test the simple model and its simulator function."""
        simple_model = SimpleModel_PlusSimpleNoise()

        ### Test in the case of an integer input or an np array
        self.assertEqual(simple_model.simulator(np.array([1]),1,1,0.2),2+norm.rvs(loc=0,scale=0.2,size=1,random_state=42))
        # self.assertTrue((simple_model.simulator(x=np.array([1,2,3]),a=1,b=1)==np.array([2,3,4])).all())

if __name__=='__main__':
    unittest.main()