import tensorflow as tf
import tensorflow_hub as hub

from preprocessing import *
from model import get_model_parts


def train_and_evaluate_model():
