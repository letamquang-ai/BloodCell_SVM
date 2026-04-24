import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

np.random.seed(76976)

w = pd.read_csv('dataset/wbc_SHAPES.txt', sep=',')
r = pd.read_csv('dataset/rbc_SHAPES.txt', sep=',')
p = pd.read_csv('dataset/plates_SHAPES.txt', sep=',')

data = pd.concat([w, r, p], ignore_index=True)

ei_w = pd.read_csv('dataset/wbc.txt', sep=',')
ei_r = pd.read_csv('dataset/rbc.txt', sep=',')
ei_p = pd.read_csv('dataset/plates.txt', sep=',')

ei = pd.concat([ei_p, ei_w, ei_r], ignore_index=True)

sp = ei['white'] / (ei['white'] + ei['black'])

labs = np.concatenate([
    np.repeat(1, len(w)),
    np.repeat(2, len(r)),
    np.repeat(3, len(p))
])

temp = pd.concat([sp.rename('sp'), ei, data], axis=1)

test = pd.concat([
    pd.DataFrame(labs, columns=['labs_svm']),
    temp.reset_index(drop=True)
], axis=1)

keep = test.columns[1:]

keep1 = test.index[test['labs_svm'] == 1].to_numpy()
keep2 = test.index[test['labs_svm'] == 2].to_numpy()
keep3 = test.index[test['labs_svm'] == 3].to_numpy()

valid_1 = np.random.choice(keep1, size=int(len(keep1) * 0.30), replace=False)
valid_2 = np.random.choice(keep2, size=int(len(keep2) * 0.30), replace=False)
valid_3 = np.random.choice(keep3, size=int(len(keep3) * 0.30), replace=False)

valid = np.concatenate([valid_1, valid_2, valid_3])

train_data = test.drop(valid)
valid_data = test.loc[valid]

X_train = train_data[keep]
y_train = train_data['labs_svm']

X_valid = valid_data[keep]
y_valid = valid_data['labs_svm']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='sigmoid'))
])

param_grid = {
    'svm__C': [1, 2, 3, 4, 5],
    'svm__coef0': [0, 1, 2, 3, 4, 5, 50],
    'svm__gamma': [0.0001, 0.001, 0.01, 0.1, 1, 2, 3]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)

best_model = grid.best_estimator_

ypred_train = best_model.predict(X_train)

print("\nTRAIN CONFUSION MATRIX:")
print(confusion_matrix(y_train, ypred_train))

print("TRAIN ACC:", accuracy_score(y_train, ypred_train))

ypred_valid = best_model.predict(X_valid)

print("\nVALID CONFUSION MATRIX:")
print(confusion_matrix(y_valid, ypred_valid))

print("VALID ACC:", accuracy_score(y_valid, ypred_valid))