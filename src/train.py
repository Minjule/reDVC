from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import dvc.api
import pandas as pd
from dvclive import Live
from IPython.display import HTML

X, y = make_circles(noise=0.3, factor=0.5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


for n_estimators in (10, 50, 100):

  with Live() as live:

    live.log_param("n_estimators", n_estimators)

    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)

    live.log_metric("train/f1", f1_score(y_train, y_train_pred, average="weighted"), plot=False)
    live.log_sklearn_plot(
      "confusion_matrix", y_train, y_train_pred, name="train/confusion_matrix",
      title="Train Confusion Matrix")

    y_test_pred = clf.predict(X_test)

    live.log_metric("test/f1", f1_score(y_test, y_test_pred, average="weighted"), plot=False)
    live.log_sklearn_plot(
      "confusion_matrix", y_test, y_test_pred, name="test/confusion_matrix",
      title="Test Confusion Matrix")
