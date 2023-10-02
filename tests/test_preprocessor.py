import unittest
import os
import sys
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from unittest.mock import patch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from preprocessor import VideoPreprocessor

class TestVideoPreprocessor(unittest.TestCase):
    def setUp(self):
        self.test_csv = os.path.abspath('tests')
        self.test_out = os.path.abspath('tests/output')
        self.test_out_file = self.test_out + '/test.csv'
        
    def tearDown(self):
        os.remove(self.test_out_file)
        
    
    def test_preprocess_csv(self):
        test_preprocess = VideoPreprocessor(self.test_csv, self.test_out)
        
        test_preprocess.preprocess_csv()
        
        self.assertTrue(os.path.isfile(self.test_out_file))
        
if __name__ == "__main__":
    unittest.main()
