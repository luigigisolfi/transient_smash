from transient_smash.models.model import Model,SimpleModel
import unittest
import numpy as np

class TestSimpleModel(unittest.TestCase):
    
    def test_simulator(self):
        """Test the simple model and its simulator function."""
        simple_model = SimpleModel()

        ### Test in the case of an integer input or an np array
        self.assertEqual(simple_model.simulator(x=1,a=1,b=1),2)
        self.assertTrue((simple_model.simulator(x=np.array([1,2,3]),a=1,b=1)==np.array([2,3,4])).all())

if __name__=='__main__':
    unittest.main()