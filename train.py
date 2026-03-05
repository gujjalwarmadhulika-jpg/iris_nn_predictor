import pickle
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    # 1️⃣ Load Iris dataset automatically
    iris = load_iris()
    X = iris.data
    y = iris.target

    print("Iris dataset loaded successfully!")

    # 2️⃣ Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3️⃣ Create model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 4️⃣ Train model
    model.fit(X_train, y_train)

    # 5️⃣ Evaluate model
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"Training Accuracy: {train_acc:.2f}")
    print(f"Testing Accuracy: {test_acc:.2f}")

    # 6️⃣ Save model
    os.makedirs("models", exist_ok=True)
    with open("models/iris_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved as models/iris_model.pkl")


if __name__ == "__main__":
    main()