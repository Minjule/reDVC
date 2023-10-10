import random
from pathlib import Path

from dvclive import Live
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

NUM_EPOCHS = 3

X, y = make_circles(noise=0.3, factor=0.5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=42)
    
with Live() as live:

    live.log_param("epochs", NUM_EPOCHS)
    live.log_param("n_estimators", 30)

    clf = RandomForestClassifier(n_estimators=30)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)

    for i in range(NUM_EPOCHS):
        train_model(...)
        live.log_metric("metric", i + random.random())
        live.log_metric("nested/metric", i + random.random())
        live.log_image(f"img/{live.step}.png", Image.new("RGB", (50, 50), (i, i, i)))
        Path("model.pt").write_text(str(random.random()))
        live.next_step()

    live.log_artifact("model.pt", type="model", name="mymodel")
    live.log_sklearn_plot("confusion_matrix", [0, 0, 1, 1], [0, 1, 0, 1])
    live.log_metric("summary_metric", 1.0, plot=False)