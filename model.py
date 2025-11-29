import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH = "rf_model.joblib"

def prepare_X_y(df):
    X = df[['r_1','r_2','sma_diff','vol_ema']].values
    # target: 1 if next return > 0, else 0
    y = (df['return'].shift(-1) > 0).astype(int).iloc[:-1]
    X = X[:-1]
    return X, y

def train_and_save(df):
    X, y = prepare_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    joblib.dump(clf, MODEL_PATH)
    return clf

def load_model():
    return joblib.load(MODEL_PATH)