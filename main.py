import os

for folder in ["data", "configs", "figures", "logs"]:
    os.makedirs(folder, exist_ok=True)

os.system("python3 data_setup.py")
os.system("python3 question_1.py | tee logs/q1.log")
os.system("python3 question_2.py | tee logs/q2.log")
os.system("python3 question_3.py | tee logs/q3.log")