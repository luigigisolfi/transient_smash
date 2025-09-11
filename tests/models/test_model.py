from transient_smash.models.model import *
import unittest
import numpy as np
from scipy.stats import norm

class TestModel(unittest.TestCase):
    def test_abstract_model(self):
        """Test that the abstract Model class cannot be instantiated."""
        with self.assertRaises(TypeError):
            model = Model()  # This should raise a TypeError since Model is abstract

class TestSimpleModel(unittest.TestCase):
    
    def test_evaluate(self):
        """Test the simple model and its simulator function."""
        simple_model = SimpleModel()
        ### Test in the case of an integer input or an np array
        self.assertEqual(simple_model.evaluate(x=1,a=1,b=1),2)
        self.assertTrue((simple_model.evaluate(x=np.array([1,2,3]),a=1,b=1)==np.array([2,3,4])).all())

    def test_set_input_data(self):
        """Test setting input data for the model."""
        simple_model = SimpleModel()
        x_data = np.array([0, 1, 2, 3])
        simple_model.set_input_data(x_data)
        self.assertTrue((simple_model.x == x_data).all())

    def test_simulator(self):
        """Test the simple model and its simulator function."""
        simple_model = SimpleModel()
        a = 1
        b = 1
        ### Test that simulator raises an error if input data not set
        with self.assertRaises(ValueError):
            simple_model.simulator(a,b)
        ### Test that simulator works correctly after setting input data
        simple_model.set_input_data(np.array([1,2,3]))
        self.assertTrue((simple_model.simulator(a,b)==np.array([2,3,4])).all())
    
class TestNoisySimpleModel(unittest.TestCase):

    def test_evaluate_with_noise(self):
        """Test the simple model and its simulator function."""
        simple_model = SimpleModel_PlusSimpleNoise()

        ### Test in the case of an np array with a fixed random seed, 42
        self.assertEqual(simple_model.evaluate(np.array([1]),1,1,0.2),2+norm.rvs(loc=0,scale=0.2,size=1,random_state=42))
        # self.assertTrue((simple_model.simulator(x=np.array([1,2,3]),a=1,b=1)==np.array([2,3,4])).all())

if __name__=='__main__':
    unittest.main()