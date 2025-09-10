from transient_smash.models.model import Model,SimpleModel
import unittest

class TestSimpleModel(unittest.TestCase):
    
    def test_simulator(self):
        simple_model = SimpleModel()
        self.assertEqual(simple_model.simulator(x=1,a=1,b=1),2)

if __name__=='__main__':
    unittest.main()