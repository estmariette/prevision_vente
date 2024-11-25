import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from io import BytesIO

# Charger les données
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

# Modèle de prévision
def forecast_sales(data, target, forecast_steps=12, frequency='M'):
    sales_data = data[target].dropna()

    # Modèle Holt-Winters pour la prévision
    model = ExponentialSmoothing(
        sales_data,
        seasonal='add',
        trend='add',
        seasonal_periods=12
    ).fit()

    forecast = model.forecast(forecast_steps)
    return sales_data, forecast

# Graphique interactif ou PNG
def plot_forecast(sales_data, forecast, title, output_png=False):
    plt.figure(figsize=(10, 6))
    plt.plot(sales_data, label='Ventes Historiques')
    plt.plot(forecast, label='Prévisions', linestyle='--', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Ventes')
    plt.legend()
    if output_png:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf
    else:
        st.pyplot(plt)

# Interface utilisateur avec Streamlit
def main():
    st.title("Application de Prévision des Ventes")
    st.subheader("Analyse des Catégories, Sous-Catégories et Références Produits")

    # Charger les données
    file_path = "historical_sales_2020_2024.csv"  # Modifier avec le chemin réel
    data = load_data(file_path)

    # Options d'analyse
    categories = data['Catégorie'].unique()
    selected_category = st.selectbox("Choisissez une catégorie :", ['Toutes'] + list(categories))

    if selected_category != 'Toutes':
        subcategories = data[data['Catégorie'] == selected_category]['Référence'].unique()
        selected_reference = st.selectbox("Choisissez une référence :", ['Toutes'] + list(subcategories))
    else:
        selected_reference = 'Toutes'

    # Période de prévision
    forecast_steps = st.slider("Nombre de mois à prévoir :", 1, 24, 12)
    frequency = st.selectbox("Fréquence de prévision :", ['Mensuelle', 'Semestrielle'])

    if frequency == 'Semestrielle':
        forecast_steps = forecast_steps // 6

    # Lancer l'analyse
    if st.button("Prédire"):
        if selected_category == 'Toutes':
            target_data = data.groupby('Date')['Ventes'].sum()
            title = "Prévisions pour Toutes les Catégories"
        elif selected_reference == 'Toutes':
            target_data = data[data['Catégorie'] == selected_category].groupby('Date')['Ventes'].sum()
            title = f"Prévisions pour la Catégorie : {selected_category}"
        else:
            target_data = data[data['Référence'] == selected_reference].groupby('Date')['Ventes'].sum()
            title = f"Prévisions pour la Référence : {selected_reference}"

        sales_data, forecast = forecast_sales(target_data, target_data.index, forecast_steps, frequency)

        # Afficher les résultats
        st.write(f"Prédictions pour les {forecast_steps} prochains {frequency.lower()}(s)")
        plot_forecast(sales_data, forecast, title)

        # Option d'export en PNG
        if st.button("Télécharger le graphique en PNG"):
            png_file = plot_forecast(sales_data, forecast, title, output_png=True)
            st.download_button(
                label="Télécharger l'image",
                data=png_file,
                file_name=f"{title.replace(' ', '_')}.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
