# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 20:50:19 2025

@author: sanja
"""

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Quick check: can it build a simple model?
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(10, input_shape=(5,), activation='relu')])
print("Model built successfully ✅")
