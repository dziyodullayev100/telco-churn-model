import joblib
import pandas as pd

from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def load_data(path):
    df = pd.read_csv(path)

    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)


    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


    df.dropna(inplace=True)


    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    return df


def build_preprocessor(X):
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_cols),
        ('categorical', categorical_transformer, categorical_cols)
    ])

    return preprocessor


def train_models(X, y, preprocessor):
    models = [
        LogisticRegression(class_weight='balanced', solver='liblinear'),
        RandomForestClassifier(),
        GradientBoostingClassifier()
    ]

    base_score = 0
    best_pipeline = None

    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > base_score:
            best_pipeline = pipe
            base_score = score.mean()

    best_pipeline.fit(X, y)
    print(f'\nbest_model: {type(best_pipeline.named_steps["classifier"]).__name__}, accuracy: {base_score:.4f}')

    return best_pipeline, base_score


def save_model(pipeline, base_score, path):
    joblib.dump({
        'model': pipeline,
        'metadata': {
            'name': 'churn prediction pipeline',
            'version': '1.0',
            'date': str(datetime.now()),
            'type': type(pipeline.named_steps['classifier']).__name__,
            'accuracy': base_score
        }
    }, path)
    print(f' Model saqlandi: {path}')


def main():
    print('Churn Prediction Pipeline\n')


    df = load_data('model/data/dataset.csv')
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    preprocessor = build_preprocessor(X)
    best_pipeline, base_score = train_models(X, y, preprocessor)
    save_model(best_pipeline, base_score, 'model/data/churn_pipeline.pkl')


if __name__ == '__main__':
    main()