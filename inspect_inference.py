with open('pupil_tracking/ml/inference.py', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'class SegmentationInference' in line or (66 <= i <= 160):
        print("{:4d} | {}".format(i+1, line), end='')