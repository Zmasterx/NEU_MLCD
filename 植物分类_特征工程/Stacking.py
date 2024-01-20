# 集成分类器
from sklearn.ensemble import StackingClassifier
# XGBooster分类器
from xgboost import XGBClassifier
# 支持向量机
from sklearn.svm import SVC, SVR
# 随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# 逻辑回归模型
from sklearn.linear_model import LogisticRegression

def Stacking(data,label):
    base_classifiers = [
        ('rf', RandomForestClassifier(n_estimators=400, max_features='sqrt', max_depth=40,
                                 bootstrap=False, oob_score=False, random_state=10)),
        ('SVC', SVC(C=1, probability=True)),
        ('Et', ExtraTreesClassifier(n_estimators=400, max_features='sqrt', max_depth=50)),
        ('XG', XGBClassifier(learning_rate=0.13, max_depth=5, n_estimators=300, nthread=10,
                        use_label_encoder=False, eval_metric='mlogloss'))        ]
    meta_classifier = LogisticRegression(solver='liblinear')  #输入维度为40*N
    # meta_classifier = SVC(C=1 , probability=True)  # 使用SVC作为元分类器
    stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)
    stacking_classifier.fit(data, label)  # 训练堆叠分类器
    return stacking_classifier  # 进行预测

# def Stacking(data,label):
#     base_classifiers = [
#         ('rf', RandomForestClassifier(n_estimators=400, max_features='sqrt', max_depth=40,
#                                  bootstrap=False, oob_score=False, random_state=10)),
#         ('SVC', SVC(C=1, probability=True)),
#         ('Et', ExtraTreesClassifier(n_estimators=400, max_features='sqrt', max_depth=50)),
#         ('XG', XGBClassifier(learning_rate=0.13, max_depth=5, n_estimators=300, nthread=10,
#                         use_label_encoder=False, eval_metric='mlogloss'))        ]
#     # meta_classifier = LogisticRegression()
#     meta_classifier = SVC(C=1 , probability=True)  # 使用SVC作为元分类器
#     stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)
#     stacking_classifier.fit(data, label)  # 训练堆叠分类器
#     return stacking_classifier  # 进行预测


