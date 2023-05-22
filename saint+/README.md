# Structure
```bash
─ saint+
  ├── README.md
  ├── after_data # 전처리 후 데이터 grouping
  ├── data # 기본 데이터
  │    ├── test_data.csv
  │    ├── total_data.csv # test + train 에서 answerCode가 -1인 제외
  │    └── train_data.csv
  ├── model
  ├── output
  ├── args.py 
  ├── data_generator.py
  ├── kfold.py
  ├── model.py
  ├── pre_process.py
  ├── submission.py
  ├── train.py
  └── utils.py
```
# Run
```
1.
python pre_process.py
python train.py
python submission.py

2. kfold
python kfold.py
```
