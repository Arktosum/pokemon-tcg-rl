import sys
import os

# Remove the kaggle_libs path so we use local Windows onnxruntime for the local test
# Actually, since main.py adds it dynamically in its try block, we can't easily prevent it unless we temporarily patch main.py.

with open('main.py', 'r') as f:
    content = f.read()

# Temporarily remove sys.path.insert for the test
patched_content = content.replace("sys.path.insert(0", "#sys.path.insert(0")

with open('main_test.py', 'w') as f:
    f.write(patched_content)

import main_test as main

class MockSelect:
    def __init__(self, maxCount, option):
        self.maxCount = maxCount
        self.option = option

class MockObs:
    def __init__(self, step, select):
        self.step = step
        self.select = select

# Step 0 test
obs0 = MockObs(0, None)
res0 = main.agent(obs0)
print(f"Step 0 Result length: {len(res0)}, First 5: {res0[:5]}")

# Step 1 test
obs1 = MockObs(1, MockSelect(2, [0, 1, 2, 3, 4]))
res1 = main.agent(obs1)
print(f"Step 1 Result: {res1}")
