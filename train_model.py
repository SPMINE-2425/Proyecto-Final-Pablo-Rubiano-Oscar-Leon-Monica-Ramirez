import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

def train_and_save_model():
    # 1. Cargar y limpiar datos
    df = pd.read_csv('data.csv', sep=';', encoding='utf-8')
    df.columns = df.columns.str.replace(r'[\t";]', '', regex=True).str.strip()

    # 2. Selección de características
    feature_columns = [
        'Marital status',
        'Application mode',
        'Application order',
        'Daytime/evening attendance',
        'Previous qualification',
        'Previous qualification (grade)',
        'Admission grade',
        'Debtor',
        'Tuition fees up to date',
        'Gender',
        'Scholarship holder',
        'Age at enrollment'
    ]
    target_column = 'Target'

    X = df[feature_columns]
    y = df[target_column]

    # 3. Codificar target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. Preprocesamiento
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 5. Modelo
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42
        ))
    ])

    # 6. Entrenamiento
    model.fit(X, y_encoded)

    # 7. Guardar componentes por separado
    joblib.dump({
        'pipeline': model,
        'label_encoder': le,
        'feature_columns': feature_columns,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }, 'model.joblib')
    
    # Metadata para la API
    metadata = {
        'feature_columns': feature_columns,
        'target_classes': le.classes_.tolist(),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f)

    print("✅ Modelo entrenado y guardado correctamente")

if __name__ == '__main__':
    train_and_save_model()