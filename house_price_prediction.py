
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import widgets
from IPython.display import display

class HousePricePredictor:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.preprocessor = None
        self.trained_models = {}

    def generate_sample_data(self, num_samples=1000):
        locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville', 'San Francisco', 'Columbus', 'Fort Worth']
        property_types = ['House', 'Apartment', 'Townhouse']
        conditions = ['Poor', 'Fair', 'Good', 'Excellent']

        data = {
            'location': np.random.choice(locations, num_samples),
            'property_type': np.random.choice(property_types, num_samples),
            'size': np.random.randint(800, 5000, num_samples), # sq ft
            'bedrooms': np.random.randint(1, 6, num_samples),
            'bathrooms': np.random.randint(1, 5, num_samples),
            'age': np.random.randint(0, 80, num_samples), # years
            'condition': np.random.choice(conditions, num_samples),
            'has_garage': np.random.randint(0, 2, num_samples),
            'has_garden': np.random.randint(0, 2, num_samples)
        }

        df = pd.DataFrame(data)

        # Introduce location-based price multipliers
        location_multipliers = {loc: np.random.uniform(0.8, 2.0) for loc in locations}
        df['price_multiplier'] = df['location'].map(location_multipliers)

        # Base price calculation (simplified)
        df['base_price'] = (df['size'] * 200 +
                            df['bedrooms'] * 50000 +
                            df['bathrooms'] * 30000 -
                            df['age'] * 1000)

        # Add noise and apply multipliers
        df['price'] = (df['base_price'] * df['price_multiplier'] +
                       np.random.normal(0, 50000, num_samples))

        # Ensure prices are not negative
        df['price'] = df['price'].apply(lambda x: max(x, 50000))

        return df[['location', 'property_type', 'size', 'bedrooms', 'bathrooms', 'age', 'condition', 'has_garage', 'has_garden', 'price']]

    def prepare_data(self, df):
        X = df.drop('price', axis=1)
        y = df['price']

        categorical_features = ['location', 'property_type', 'condition']
        numerical_features = ['size', 'bedrooms', 'bathrooms', 'age', 'has_garage', 'has_garden']

        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough' # Keep other columns (like house_id if added)
        )

        return X, y

    def train_models(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}
        for name, model in self.models.items():
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                       ('regressor', model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {'MSE': mse, 'R²': r2, 'model': pipeline}
            self.trained_models[name] = pipeline

        return results, X_test, y_test

    def plot_results(self, results):
        model_names = list(results.keys())
        mse_values = [results[name]['MSE'] for name in model_names]
        r2_values = [results[name]['R²'] for name in model_names]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].bar(model_names, mse_values, color='skyblue')
        axes[0].set_ylabel('Mean Squared Error (MSE)')
        axes[0].set_title('Model Performance: MSE')
        axes[0].tick_params(axis='x', rotation=45)

        axes[1].bar(model_names, r2_values, color='lightgreen')
        axes[1].set_ylabel('R-squared (R²)')
        axes[1].set_title('Model Performance: R²')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, df):
        if 'Random Forest' in self.trained_models:
            rf_model = self.trained_models['Random Forest']
            # The feature importances are from the regressor step in the pipeline
            importances = rf_model.named_steps['regressor'].feature_importances_

            # Get feature names after preprocessing
            # This part can be tricky depending on how the preprocessor handles columns
            # For simplicity, let's get original numerical features and try to map one-hot encoded ones
            categorical_features = ['location', 'property_type', 'condition']
            numerical_features = ['size', 'bedrooms', 'bathrooms', 'age', 'has_garage', 'has_garden']

            # Get names of one-hot encoded features
            ohe_feature_names = list(self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

            feature_names = numerical_features + ohe_feature_names

            # Ensure the number of importances matches the number of feature names
            if len(importances) == len(feature_names):
                feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

                plt.figure(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20)) # Top 20 features
                plt.title('Top 20 Feature Importance (Random Forest)')
                plt.show()
            else:
                print("Warning: Could not match feature importances to feature names correctly.")
                print(f"Number of importances: {len(importances)}")
                print(f"Number of feature names: {len(feature_names)}")
        else:
            print("Random Forest model not trained. Cannot plot feature importance.")


    def predict_single_house(self, house_features):
        # Convert dictionary to DataFrame
        house_df = pd.DataFrame([house_features])

        predictions = {}
        for name, model in self.trained_models.items():
            try:
                prediction = model.predict(house_df)
                predictions[name] = prediction[0]
            except Exception as e:
                predictions[name] = f"Error predicting: {e}"
                print(f"Error predicting with {name}: {e}")

        return predictions

    def interactive_prediction(self):
        locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville', 'San Francisco', 'Columbus', 'Fort Worth']
        property_types = ['House', 'Apartment', 'Townhouse']
        conditions = ['Poor', 'Fair', 'Good', 'Excellent']

        location_widget = widgets.Dropdown(options=locations, description='Location:')
        property_type_widget = widgets.Dropdown(options=property_types, description='Property Type:')
        size_widget = widgets.IntSlider(min=500, max=6000, step=50, value=2000, description='Size (sq ft):')
        bedrooms_widget = widgets.IntSlider(min=1, max=10, step=1, value=3, description='Bedrooms:')
        bathrooms_widget = widgets.IntSlider(min=1, max=10, step=0.5, value=2, description='Bathrooms:')
        age_widget = widgets.IntSlider(min=0, max=100, step=1, value=10, description='Age (years):')
        condition_widget = widgets.Dropdown(options=conditions, description='Condition:')
        garage_widget = widgets.Checkbox(value=True, description='Has Garage:')
        garden_widget = widgets.Checkbox(value=False, description='Has Garden:')

        output_widget = widgets.Output()
        predict_button = widgets.Button(description="Predict Price")

        def on_predict_button_clicked(b):
            house_features = {
                'location': location_widget.value,
                'property_type': property_type_widget.value,
                'size': size_widget.value,
                'bedrooms': bedrooms_widget.value,
                'bathrooms': bathrooms_widget.value,
                'age': age_widget.value,
                'condition': condition_widget.value,
                'has_garage': int(garage_widget.value), # Convert boolean to int
                'has_garden': int(garden_widget.value)   # Convert boolean to int
            }
            predictions = self.predict_single_house(house_features)

            with output_widget:
                output_widget.clear_output()
                print("Predicted Prices:")
                for model, price in predictions.items():
                    print(f"- {model}: ${price:,.2f}")

        predict_button.on_click(on_predict_button_clicked)

        display(widgets.VBox([
            location_widget,
            property_type_widget,
            size_widget,
            bedrooms_widget,
            bathrooms_widget,
            age_widget,
            condition_widget,
            garage_widget,
            garden_widget,
            predict_button,
            output_widget
        ]))

if __name__ == '__main__':
    predictor = HousePricePredictor()
    df = predictor.generate_sample_data(1000)
    X, y = predictor.prepare_data(df.copy())
    results, X_test, y_test = predictor.train_models(X, y)

    print("Model Training Results:")
    for name, res in results.items():
        print(f"{name}: MSE={res['MSE']:.2f}, R²={res['R²']:.2f}")

    predictor.plot_results(results)
    predictor.plot_feature_importance(df)

    print("\nInteractive Prediction Tool:")
    predictor.interactive_prediction()

    # Example of predicting a single house
    house_features = {
        'location': 'San Francisco',
        'property_type': 'House',
        'size': 2500,
        'bedrooms': 4,
        'bathrooms': 3,
        'age': 15,
        'condition': 'Good',
        'has_garage': 1,
        'has_garden': 1
    }
    predictions = predictor.predict_single_house(house_features)
    print("\nPrediction for a specific house:")
    print(predictions)
