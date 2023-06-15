# Air Pollution Predictor
Projekt wykorzystujący uczenie maszynowe do przewidywania zanieczyszczenia powietrza.

A project that implements machine learning to predict air pollution.

## Project structure
```
data
├── data_pipeline
│   ├── api_key_example.txt - klucz api OpenWeatherMap
│   └── 3h
│       └── 3h.py - plik do pozyskiwania danych
├── model/3h
│   ├── NeuralNetwork
│   │   ├── neural_network.py - implementacja sieci neuronowych
│   │   └── nntest.py - metody testowe dla sieci neuronowych
│   ├── RandomForestRegressor
│   └── forst.py - implementacja sieci neuronowych oraz testowanie
│       └── choosingParams
│           ├── findBestParams.py - znajdowanie najlepszych hiperparametrów
│           └── testParams.py - sprawdzanie zestawów parametrów z najmniejszym marginesem błędu
├── statistics/3h
│   ├── statistics.py - analiza regresji liniowej dla sieci neuronowych
│   └── timeutils.py - funkcje do zamiany daty i czasu na inny format
├── dane_nauka.csv
├── dane_predykcja.csv
├── gui.py - główny plik do interfejsu graficznego prognozy
├── logo.png
└── rqeuirements.txt - zależności python projektu
```

## How to run
Interfejs graficzny do prognozy można uruchomić poprzez plik gui.py. Wykonywanie innych predykcji możliwe jest poprzez wykorzystanie metod dostarczanych przez implementacje modeli.

The graphical interface to the forecast can be run through the gui.py file. Performing other predictions is possible by using methods provided by model implementations.

## How to use graphical interface
W interfejsie graficznym należy załączyć pliki csv do nauki i predykcji w odpowiednich polach i następnie nacisnąć przycisk Generate chart.

In the GUI, attach csv files for learning and prediction in the appropriate fields and then press the Generate chart button.
