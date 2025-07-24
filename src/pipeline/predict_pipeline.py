import os
import sys
from src.exception import CustomException
df = os.path.join('notebook','data','housing.csv')

de=os.path
print(de)

raw_file_path = os.path.join('notebook\data\housing.csv')
if not os.path.exists(raw_file_path):
    raise CustomException(f"File not found: {raw_file_path}", sys)

print(df)