
# Autoscout24 Datenanalyse und Preisvorhersage

## Übersicht

Diese Anwendung bietet umfassende Einblicke in die Daten von Autoscout24. Mit interaktiven Visualisierungen und modernen maschinellen Lernmodellen können Sie die wichtigsten Merkmale von Fahrzeugen analysieren und zukünftige Autopreise vorhersagen. Nutzen Sie die verschiedenen Funktionen, um tiefere Einblicke zu gewinnen und fundierte Entscheidungen zu treffen. Egal, ob Sie Autohändler, Käufer oder einfach nur ein Datenenthusiast sind, diese Plattform bietet Ihnen wertvolle Werkzeuge zur Analyse und Vorhersage von Fahrzeugpreisen.

## Funktionen

- **Home:** Überblick und Einführung in die Anwendung.
- **Marken:** Vergleich von Automarken und Modellen mit Radar-Diagrammen und Gesamtscores.
- **Datenanalyse:** Verschiedene grafische Analysen der Autoscout24-Daten, z.B. Mileage vs Price, Price Histogram, Correlation Heatmap und mehr.
- **ML-Modelle:** Preisvorhersage für Autos basierend auf verschiedenen Merkmalen mit Random Forest.

## Installation

1. Klone das Repository:

   ```sh
   git clone https://github.com/dein-benutzername/autoscout24-datenanalyse.git
   cd autoscout24-datenanalyse
   ```

2. Installiere die erforderlichen Python-Pakete:

   ```sh
   pip install -r requirements.txt
   ```

3. Starte die Anwendung:

   ```sh
   streamlit run main.py
   ```

## Verwendung

### Home

- Überblick über die Anwendung und ihre Funktionen.

### Marken

- Wählen Sie eine Automarke und ein oder mehrere Modelle aus, um deren Merkmale zu vergleichen.
- Die Radar-Diagramme und Gesamtscores helfen, die Unterschiede zwischen den Modellen zu visualisieren.

### Datenanalyse

- Wählen Sie eine der verfügbaren Grafiken aus, um verschiedene Aspekte der Autoscout24-Daten zu analysieren.
- Erhalten Sie Erkenntnisse über die Beziehungen zwischen verschiedenen Merkmalen wie Preis, Kilometerstand, PS und mehr.

### ML-Modelle

- Geben Sie die Merkmale eines Fahrzeugs ein, um eine Preisvorhersage zu erhalten.
- Vergleichen Sie den vorhergesagten Preis mit dem Durchschnittspreis des Modells.
- Erhalten Sie eine Modellevaluierung für die 5 am häufigsten verkauften Hersteller.

## Projektstruktur

```plaintext
.
├── main.py                  # Hauptanwendungscode
├── autoscout.jpg            # Hintergrundbild
├── final_autoscout24.csv    # Datensatz
├── requirements.txt         # Abhängigkeiten
└── README.md                # Diese README-Datei
```

## Abhängigkeiten

- Python 3.8 oder höher
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Seaborn
- Matplotlib

## Autor

- Julien Laucke
- Kontakt: julienlaucke@web.de
```

