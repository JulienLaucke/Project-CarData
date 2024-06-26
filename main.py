import base64
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="centered")

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("autoscout.jpg")

# Erstellen des Hintergrundbildes
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: 100%;
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

[data-testid="element-container"]{{
style:"color-scheme: gray";
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Home", "Marken", "Datenanalyse", "ML-Modelle"],
    icons=["house", "bar-chart", "bar-chart", "calculator-fill"],
    menu_icon="cast",
    orientation="horizontal")

@st.cache_data
def load_data():
    df = pd.read_csv("final_autoscout24.csv")
    return df

df = load_data()

# Home Section
if selected == "Home":
    st.title("Willkommen zur Autoscout24 Datenanalyse")
    st.write("Diese Anwendung bietet umfassende Einblicke in die Daten von Autoscout24. Mit interaktiven Visualisierungen und modernsten maschinellen Lernmodellen können Sie die wichtigsten Merkmale von Fahrzeugen analysieren und zukünftige Autopreise vorhersagen. Nutzen Sie die verschiedenen Funktionen, um tiefere Einblicke zu gewinnen und fundierte Entscheidungen zu treffen. Egal, ob Sie Autohändler, Käufer oder einfach nur ein Datenenthusiast sind, diese Plattform bietet Ihnen wertvolle Werkzeuge zur Analyse und Vorhersage von Fahrzeugpreisen.")

# Marken Section
elif selected == "Marken":
    st.title("Auto Vergleichssystem")
    brands = df['make'].unique()
    selected_brand = st.selectbox("Wählen Sie eine Marke", brands)

    if selected_brand:
        brand_df = df[df['make'] == selected_brand]
        models = brand_df['model'].unique()
        selected_models = st.multiselect("Wählen Sie ein oder mehrere Modelle", models)

        if selected_models:
            radar_data = []
            for selected_model in selected_models:
                model_entries = brand_df[brand_df['model'] == selected_model]
                
                avg_price = model_entries['price'].mean()
                avg_hp = model_entries['hp'].mean()
                avg_year = model_entries['year'].mean()
                avg_mileage = model_entries['mileage'].mean()

                def format_value(value, average, lower_is_better=True):
                    if lower_is_better:
                        color = "green" if value < average else "red"
                    else:
                        color = "green" if value > average else "red"
                    return f'<span style="color:{color}">{value:.2f}</span>'

                st.write(f"### Modell: {selected_model}")
                st.write(f"Durchschnittlicher Preis des Modells: {avg_price:.2f}")
                st.write(f"Durchschnittliche PS des Modells: {avg_hp:.2f}")
                st.write(f"Durchschnittlicher Baujahr des Modells: {avg_year:.2f}")
                st.write(f"Durchschnittlicher Kilometerstand des Modells: {avg_mileage:.2f}")

                def manual_normalize(data, min_value, max_value):
                    normalized_data = (data - min_value) / (max_value - min_value)
                    return normalized_data

                def manual_inverse_normalize(data, min_value, max_value):
                    inverted_data = 1 - (data - min_value) / (max_value - min_value)
                    return inverted_data

                normalized_avg_price = manual_inverse_normalize(avg_price, 1000, 200000)
                normalized_avg_hp = manual_normalize(avg_hp, 1, 800)
                normalized_avg_mileage = manual_inverse_normalize(avg_mileage, 0, 150000)
                normalized_avg_year = manual_normalize(avg_year, 1980, 2024)

                radar_data.append({
                    'model': selected_model,
                    'values': [normalized_avg_price, normalized_avg_hp, normalized_avg_mileage, normalized_avg_year],
                    'score': (normalized_avg_price + normalized_avg_hp + normalized_avg_mileage + normalized_avg_year) / 4
                })

            if radar_data:
                categories = ['Preis', 'PS', 'Kilometerstand', 'Jahr']

                fig = go.Figure()

                colors = ['red', 'blue', 'green', 'purple', 'orange']

                for i, data in enumerate(radar_data):
                    fig.add_trace(go.Scatterpolar(
                          r=data['values'],
                          theta=categories,
                          fill='toself',
                          name=f'{data["model"]}',
                          line_color=colors[i % len(colors)]
                    ))

                fig.update_layout(
                  polar=dict(
                    radialaxis=dict(
                      visible=True,
                      range=[0, 1]
                    )),
                  showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Gesamtscores")
                for data in radar_data:
                    st.metric(label=f"Gesamtscore für {data['model']}", value=f"{data['score']:.2f}")

                # Zusatz: Detailansicht der Scores
                left_chart, right_indicator = st.columns([1.5, 1])


# Datenanalyse Section
elif selected == "Datenanalyse":
    st.title("Datenanalyse")

    plot_options = [
        "Mileage vs Price",
        "Mileage vs HP",
        "Make-Model Counts",
        "Top 20 Makes Boxplot",
        "Price Histogram",
        "Price vs. Year Boxplot",
        "Correlation Heatmap",
        "Verteilung der Getriebetypen",
        "Price per Manufacturer Barplot by Gear",
        "Mean Price per Manufacturer"
    ]
    
    selected_plot = st.selectbox("Wählen Sie eine Grafik aus:", plot_options)

    sns.set(style="darkgrid")
    plt.style.use('default')

    
    if selected_plot == "Mileage vs Price":
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.set(style="darkgrid")
        sns.scatterplot(x='mileage', y='price', hue='year', palette='viridis', data=df)
        ax.set_title('Mileage vs Price', color='white')
        ax.set_xlabel('Mileage', color='white')
        ax.set_ylabel('Price', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.set_facecolor('#333333')  
        fig.patch.set_facecolor('#333333')  
        st.pyplot(fig)
        st.write("Erkenntnis: Mit zunehmender Laufleistung eines Autos sinkt der Preis im Allgemeinen. Neuere Autos (in einer anderen Farbe dargestellt) haben tendenziell höhere Preise, selbst bei höherer Laufleistung.")

    elif selected_plot == "Mileage vs HP":
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.set(style="darkgrid")
        sns.scatterplot(x='mileage', y='hp', hue='year', palette='viridis', data=df, ax=ax, alpha=0.8)
        ax.set_title('Mileage vs HP', color='white')
        ax.set_xlabel('Mileage', color='white')
        ax.set_ylabel('HP', color='white')
        ax.tick_params(colors='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        ax.grid(False)  
        st.pyplot(fig)
        st.write("Erkenntnis: Es gibt keine klare Korrelation zwischen Laufleistung und PS (HP). Neuere Autos (in einer anderen Farbe dargestellt) haben jedoch tendenziell eine breitere Palette von PS-Werten.")


    elif selected_plot == "Make-Model Counts":
        make_model_counts = df[['make', 'model']].value_counts().reset_index(name='count')
        make_model_counts['make+model'] = make_model_counts['make'] + make_model_counts['model']
        df['make_model'] = df['make'] + df['model']
        df = df[df['make_model'].isin(make_model_counts['make+model'])]

        top_make_model_counts = make_model_counts.nlargest(20, 'count')
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.set(style="darkgrid")
        sns.barplot(x='count', y='model', hue='make', data=top_make_model_counts, palette='viridis', ax=ax)
        ax.set_xlabel('Count', color='white')
        ax.set_ylabel('Model', color='white')
        ax.set_title('Counts of Make-Model Combinations', color='white')
        ax.legend(title='Make', bbox_to_anchor=(1.05, 1), loc='upper left', title_fontsize='13', fontsize='11', frameon=False, labelcolor='white')
        ax.tick_params(colors='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        ax.grid(False) 
        st.pyplot(fig)
        st.write("Erkenntnis: Die Top 20 der Make-Model-Kombinationen zeigen, welche Automodelle in den Daten am häufigsten vorkommen. Bestimmte Marken dominieren die Liste mit mehreren beliebten Modellen.")

    elif selected_plot == "Top 20 Makes Boxplot":
        make_counts = df['make'].value_counts()
        top_20_makes = make_counts.nlargest(20).index.tolist()
        df_filtered = df[(df['make'].isin(top_20_makes)) & (df['price'] >= 0) & (df['price'] <= 250000)]
        
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.set(style="darkgrid")
        palette = sns.color_palette("husl", len(top_20_makes))
        sns.boxplot(x='make', y='price', data=df_filtered, ax=ax, palette=palette,
                    flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red', 'markeredgecolor': 'black'})  
        ax.set_xlabel('Marke', fontsize=15, color='white')
        ax.set_ylabel('Price', fontsize=15, color='white')
        ax.set_title('Boxplot of the Price vs. Make (Top 20, Price range: 0-250,000)', fontsize=18, color='white')
        ax.tick_params(axis='both', which='major', labelsize=10, colors='white')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, color='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        ax.grid(False)
        st.pyplot(fig)
        st.write("Erkenntnis: Das Boxplot zeigt die Preisverteilung für die Top 20 Automarken. Es gibt erhebliche Preisunterschiede innerhalb und zwischen den verschiedenen Marken.")

    elif selected_plot == "Price Histogram":
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.set(style="darkgrid")
        sns.histplot(df['price'], bins=50, ax=ax)
        ax.set_xlabel('Price', fontsize=16, color='white')
        ax.set_ylabel('Count', fontsize=16, color='white')
        ax.set_title('Histogram of the Price Variable', fontsize=20, color='white')
        ax.tick_params(axis='both', which='major', labelsize=20, colors='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        ax.grid(False) 
        st.pyplot(fig)
        st.write("Erkenntnis: Das Histogramm zeigt die Verteilung der Autopreise. Die meisten Autopreise sind am unteren Ende konzentriert, was darauf hinweist, dass es weniger hochpreisige Autos in den Daten gibt.")

    elif selected_plot == "Price vs. Year Boxplot":
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.set(style="darkgrid")
        sns.boxplot(x=df['year'], y=df['price'], ax=ax)
        ax.set_xlabel('Year', fontsize=16, color='white')
        ax.set_ylabel('Price', fontsize=16, color='white')
        ax.set_title('Boxplot of the Price vs. Year', fontsize=20, color='white')
        ax.tick_params(axis='both', which='major', labelsize=20, colors='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        ax.grid(False)
        st.pyplot(fig)
        st.write("Erkenntnis: Das Boxplot zeigt die Preisverteilung im Laufe der Jahre. Neuere Autos haben generell höhere Preise, und es gibt einen deutlichen Preisrückgang, wenn die Autos älter werden.")

    elif selected_plot == "Correlation Heatmap":
        corr_matrix = df[['price', 'mileage', 'hp', 'year']].corr()
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.set(style="darkgrid")
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', annot_kws={'size': 15}, ax=ax)
        ax.set_title('Correlations between the Numerical Variables', fontsize=20, color='white')
        ax.tick_params(axis='both', which='major', labelsize=12, colors='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        ax.grid(False) 
        st.pyplot(fig)
        st.write("Erkenntnis: Die Heatmap zeigt die Korrelationen zwischen den numerischen Variablen. Es gibt eine starke negative Korrelation zwischen Laufleistung und Preis sowie eine starke positive Korrelation zwischen Jahr und Preis.")

    elif selected_plot == "Verteilung der Getriebetypen":
        plt.figure(figsize=(10, 6))
        sns.set(style="darkgrid")
        gear_counts = df['gear'].value_counts()
        colors = sns.color_palette('viridis', len(gear_counts))
        wedges, texts, autotexts = plt.pie(gear_counts, labels=gear_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)

        for text in texts:
            text.set_color('white')
        for autotext in autotexts:
            autotext.set_color('white')

        plt.title('Verteilung der Anzahl der Autos pro Getriebetyp', color='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        st.pyplot(plt)
        st.write("Erkenntnis: Das Tortendiagramm zeigt die Verteilung der Anzahl der Autos für jeden Getriebetyp. Es ist erkennbar, dass bestimmte Getriebetypen häufiger vorkommen als andere.")


    elif selected_plot == "Price per Manufacturer Barplot by Gear":
        fig, ax = plt.subplots(figsize=(22, 14))
        sns.set(style="darkgrid")
        sns.barplot(x=df['make'], y=df['price'], hue=df['gear'], ax=ax)
        ax.set_xlabel('Manufacturer', fontsize=16, color='white')
        ax.set_ylabel('Price', fontsize=16, color='white')
        ax.set_title('Barplot of the Price per Manufacturer for Different Gear Categories', fontsize=20, color='white')
        ax.tick_params(axis='both', which='major', labelsize=20, colors='white')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, color='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        ax.grid(False) 
        st.pyplot(fig)
        st.write("Erkenntnis: Das Barplot zeigt die Preisunterschiede pro Hersteller für verschiedene Getriebearten. Bei einigen Herstellern gibt es einen signifikanten Preisunterschied basierend auf der Getriebeart.")

    elif selected_plot == "Mean Price per Manufacturer":
        mean_price_per_manuf = df.groupby('make')[['price']].mean().sort_values(by='price')
        mean_price_per_manuf.reset_index(inplace=True)
        mean_price_per_manuf.rename(columns={'make':'Make', 'price':'Avg_Price'}, inplace=True)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.set(style="darkgrid")
        sns.barplot(data=mean_price_per_manuf, x='Make', y='Avg_Price', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, color='white')
        ax.set_xlabel('Make', fontsize=16, color='white')
        ax.set_ylabel('Price', fontsize=16, color='white')
        ax.set_title('Mean Price Per Manufacturer', fontsize=20, color='white')
        ax.tick_params(axis='both', which='major', labelsize=14, colors='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        ax.grid(False)  
        st.pyplot(fig)
        st.write("Erkenntnis: Das Barplot zeigt den durchschnittlichen Preis pro Hersteller. Einige Hersteller haben deutlich höhere Durchschnittspreise, was auf eine Präsenz im Premiumsegment hinweist.")





# ML-Modell Section
elif selected == "ML-Modelle":
    st.title("Preisvorhersage für Autos mit Random Forest")

    st.header("Geben Sie die Fahrzeugmerkmale ein:")
    mileage = st.number_input("Kilometerstand", min_value=0)
    hp = st.number_input("PS", min_value=0)
    year = st.number_input("Baujahr", min_value=1900, max_value=2024)
    make = st.selectbox("Marke", df['make'].unique())
    model = st.selectbox("Modell", df[df['make'] == make]['model'].unique())
    fuel = st.selectbox("Kraftstoffart", df['fuel'].unique())
    gear = st.selectbox("Getriebeart", df['gear'].unique())

    if st.button("Preis berechnen"):
        with st.spinner('Berechne Preis...'):
            top_5_manufacturers = df['make'].value_counts().nlargest(5).index.tolist()
            df_filtered = df[df['make'].isin(top_5_manufacturers)]

            X = df_filtered[['mileage', 'hp', 'year', 'make', 'model', 'fuel', 'gear']]
            y = df_filtered['price']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Pipeline für numerische Attribute
            num_attribs = ['mileage', 'hp', 'year']
            final_num_pipeline = Pipeline([
                ('min_max_scaler', MinMaxScaler()),
                ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
            ])

            # Pipeline für kategorische Attribute
            cat_attribs = ['make', 'model', 'fuel', 'gear']
            final_pipeline = ColumnTransformer([
                ('num', final_num_pipeline, num_attribs),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs),
            ])

            X_train_prepared = final_pipeline.fit_transform(X_train)

            # Random Forest Modell
            random_forest_model = RandomForestRegressor(random_state=42)
            random_forest_model.fit(X_train_prepared, y_train)

            input_data = pd.DataFrame([[mileage, hp, year, make, model, fuel, gear]], 
                                      columns=['mileage', 'hp', 'year', 'make', 'model', 'fuel', 'gear'])
            input_prepared = final_pipeline.transform(input_data)
            predicted_price = random_forest_model.predict(input_prepared)

            st.subheader("Vorhergesagter Preis:")
            st.write(f"{predicted_price[0]:,.2f} €")

            model_entries = df[(df['make'] == make) & (df['model'] == model)]
            avg_price = model_entries['price'].mean()

            st.subheader("Durchschnittspreis des Modells:")
            st.write(f"{avg_price:,.2f} €")

            if predicted_price[0] < avg_price:
                comparison_text = f"Der vorhergesagte Preis liegt unter dem Durchschnittspreis des Modells."
                comparison_color = "green"
            else:
                comparison_text = f"Der vorhergesagte Preis liegt über dem Durchschnittspreis des Modells."
                comparison_color = "red"

            st.markdown(f"<span style='color:{comparison_color}'>{comparison_text}</span>", unsafe_allow_html=True)

            # Diagramm erstellen
            fig, ax = plt.subplots(figsize=(10, 6))
            prices = [predicted_price[0], avg_price]
            labels = ['Vorhergesagter Preis', 'Durchschnittspreis']
            colors = ['blue', 'red']

            ax.bar(labels, prices, color=colors)
            ax.set_xlabel('Preistyp', fontsize=15, color='white')
            ax.set_ylabel('Preis (€)', fontsize=15, color='white')
            ax.set_title('Vorhergesagter Preis vs. Durchschnittspreis', fontsize=18, color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            plt.gca().set_facecolor('#333333')
            plt.gcf().set_facecolor('#333333')
            ax.grid(False)

            st.pyplot(fig)

            # Modellevaluierung
            y_train_predictions_rf = random_forest_model.predict(X_train_prepared)
            X_test_prepared = final_pipeline.transform(X_test)
            y_test_predictions_rf = random_forest_model.predict(X_test_prepared)

            train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_train_predictions_rf))
            test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_predictions_rf))

            r2_score_rf_train = random_forest_model.score(X_train_prepared, y_train)
            r2_score_rf_test = random_forest_model.score(X_test_prepared, y_test)

            st.subheader("Modellevaluierung für die 5 am häufigsten verkauften Hersteller:")
            st.write(f"RMSE für das Random Forest-Modell auf den Trainingsdaten: {train_rmse_rf:.2f}")
            st.write(f"Koeffizient auf dem Trainingssatz: {r2_score_rf_train:.2f}")
            st.write(f"RMSE für das Random Forest-Modell auf den Testdaten: {test_rmse_rf:.2f}")
            st.write(f"Koeffizient auf dem Testsatz: {r2_score_rf_test:.2f}")




