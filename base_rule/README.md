# DKT Base_Rule 기반 정답 예측

---

```
base_rule/
|-- README.md
|-- base_rule.ipynb
|-- data
|   |-- KnowledgeTag_dict.pkl
|   |-- assessmentItemID_dict.pkl
|   `-- userID_dict.pkl
|-- requirements.txt
`-- submit
    |-- KnowledgeTag.csv
    |-- assemssmentItemID.csv [0.7638, 0.6909]
    |-- cut_20_result.csv [0.7641, 0.6909]
    |-- cut_30_result.csv [0.7613, 0.6909]
    |-- cut_50_result.csv [0.6930, 0.6909]
    |-- ensemble_ItemID_Tag.csv [0.7598, 0.6613]
    |-- ensemble_all.csv [0.7785, 0.6747]
    |-- ensemble_all_cut_50_result.csv
    |-- ensemble_userID_ItemID.csv
    `-- userID.csv
```

---

## - result [auroc, accuracy]
> assessmentItemID [0.7638, 0.6909]
>
> assessmentItemID, KnowledgeTag - 1:1 앙상블 [0.7598, 0.6613]
>
> userID, assessmentItemID, KnowledgeTag - 앙상블 [0.7785, 0.6747]
>
> cut 20% [0.7641, 0.6909]
>
> cut 30% [0.7613, 0.6909]
>
> cut 50% [0.6930, 0.6909]