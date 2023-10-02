import unittest
import os
import sys
import shutil
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from feature_extraction import VideoFeatureExtractor

class TestVideoFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.input_path = os.path.abspath('tests')
        self.output_path = os.path.abspath('tests/output/feature_map')
        self.openface_exe = os.path.abspath('openface2/FeatureExtraction.exe')
        
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        
    def tearDown(self):
        shutil.rmtree(self.output_path)
        
    def test_extract_features(self):
        extractor = VideoFeatureExtractor(self.openface_exe, self.input_path, self.output_path)
        extractor.extract_features()
        
        self.assertTrue(os.path.isdir(self.output_path))