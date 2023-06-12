import csv
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLineEdit, QMessageBox, QLabel, QLabel
from PyQt5.QtCore import Qt, QFileInfo
from PyQt5.QtGui import QIcon, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

sys.path.append('./model/3h/RandomForestRegressor')
import forest

sys.path.append('./model/3h/NeuralNetwork')
import nntest as nnTest
import neural_network as nn
import timeutils as ts

#TODO dodac okno z marginesem bledow

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Przewidywanie pogody")
        self.setFixedSize(400, 250)  # Ustawienie stałego rozmiaru okna

        # ikonka programu
        logo_icon = QIcon("logo.png")
        self.setWindowIcon(logo_icon)
    
        # napis glowny 
        self.start = QLabel("Predykcja wartości PM2,5", self)
        self.start.setGeometry(0, 10, 400, 30)
        font = QFont("Arial", 12, QFont.Bold)
        self.start.setFont(font)
        self.start.setAlignment(Qt.AlignCenter)

        # wczytanie 1 pliku
        self.tekst1plik = QLabel('Wczytaj plik csv z danymi do nauki:', self)
        self.tekst1plik.setGeometry(20, 50, 400, 30)

        self.line_edit = QLineEdit(self)
        self.line_edit.setGeometry(20, 80, 160, 20)
        self.line_edit.setReadOnly(True)
        self.line_edit.setStyleSheet("border: 0.5px solid black;")

        self.button = QPushButton("Wybierz plik", self)
        self.button.setGeometry(190, 75, 100, 30)
        self.button.clicked.connect(lambda: self.select_file('train'))

        # wczytanie 2 pliku
        self.tekst2plik = QLabel('Wczytaj plik csv z danymi do predykcji:', self)
        self.tekst2plik.setGeometry(20, 110, 400, 30)

        self.line_edit2 = QLineEdit(self)
        self.line_edit2.setGeometry(20, 140, 160, 20)
        self.line_edit2.setReadOnly(True)
        self.line_edit2.setStyleSheet("border: 0.5px solid black;")

        self.button2 = QPushButton("Wybierz plik", self)
        self.button2.setGeometry(190, 135, 100, 30)
        self.button2.clicked.connect(lambda: self.select_file('pred'))

        # generowanie wykresow
        self.button_wykres = QPushButton("Generuj wykres", self)
        self.button_wykres.setGeometry(120, 190, 150, 30)
        self.button_wykres.clicked.connect(self.generate_plot)

    def select_file(self, data_type):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Wybierz plik", "", "Wszystkie pliki (*)", options=options)
        if file_name:
            if file_name.lower().endswith('.csv'):
                file_info = QFileInfo(file_name)
                if data_type == 'train':
                    self.line_edit.setText(file_info.fileName())
                    self.data_train = forest.read_data(file_name)
                    self.data_train_path = file_name  # Ścieżka do pliku
                elif data_type == 'pred':
                    self.line_edit2.setText(file_info.fileName())
                    self.data_pred = forest.read_data(file_name)
                    self.data_pred_path = file_name  # Ścieżka do pliku
            else:
                QMessageBox.critical(self, "Błąd", "Wybrano nieprawidłowy plik. Proszę wybrać plik CSV.")

    def generate_plot(self):
        if self.line_edit.text() == '' or self.line_edit2.text() == '':
            QMessageBox.warning(self, "Błąd", "Nie wybrano obu plików. Proszę wybrać pliki CSV.")
            return

        # Wybór wybranych cech
        selected_features = ['pm2_5', 'wind_speed', 'wind_deg', 'temp', 'humidity']

        # Podział danych na cechy i wartości oczekiwane
        X_train = forest.select_features(self.data_train, selected_features)
        y_train = self.data_train['pm2_5_label']
        X_pred = forest.select_features(self.data_pred, selected_features)

        # Parametry modelu
        params = {
            'criterion': 'absolute_error',
            'max_depth': 15,
            'max_features': 'sqrt',
            'min_impurity_decrease': 0.0,
            'min_samples_leaf': 4,
            'min_samples_split': 10,
            'n_estimators': 150
        }

        # Tworzenie nowej instancji obiektu Figure i FigureCanvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setGeometry(10, 50, 780, 540)
        self.ax = self.figure.add_subplot(111)

        # Trening modelu
        model = forest.train_model(X_train, y_train, params)

        # Przewidywanie wartości
        predictions = forest.predict(model, X_pred)

        # SIECI NEURONOWE
        with open(self.data_train_path, 'r') as file:
            csv_reader = csv.reader(file)
            data = []
            labels = []
            for row in csv_reader:
                data.append([
                    ts.timestamp_to_day_in_week_number(row[8]),
                    ts.timestamp_to_hour(row[8]),
                    (1 / float(row[17])) * 10 * (float(row[17]) - float(row[3])),  # temp_change importance
                    float(row[6]),  # wind_speed
                    float(row[9]),  # pm2_5
                ])
                labels.append(float(row[22]))  # pm2_5_label

        # Load evaluate data from CSV
        with open(self.data_pred_path, 'r') as file:
            csv_reader = csv.reader(file)
            data2 = []
            labels2 = []
            for row in csv_reader:
                data2.append([
                    ts.timestamp_to_day_in_week_number(row[8]),
                    ts.timestamp_to_hour(row[8]),
                    (1 / float(row[17])) * 10 * (float(row[17]) - float(row[3])),  # temp_change importance
                    float(row[6]),  # wind_speed
                    float(row[9]),  # pm2_5
                ])
                labels2.append(float(row[22]))  # pm2_5_label

        # Create a neural network object
        neural_network = nn.NeuralNetwork(data, labels)

        # Tutaj dodac bar loading
        # Build and train the neural network
        neural_network.build_and_train()
        
        # Create a test object
        test = nnTest.NNTest(neural_network, data2, labels2)

        #print(data)

        # Print predictions
        predictions_neural = test.forecast(12)
        #print(predictions)

        # Save the neural network
        #neural_network.save()

        # Load the neural network
        #neural_network.load('model.h5')

        # Use the neural network to make predictions
        # neural_network.predict()

        # Convert predictions to a list of pm2.5 values
        pm25_predictions = [pred[0] for pred in predictions_neural]
        # KONIEC SIECI NEURONOWYCH

        # Generowanie wykresu
        fig, ax = plt.subplots(figsize=(8, 6))

        line1, = ax.plot(self.data_pred['pm2_5'], label='Wartości prawdziwe')
        line2, = ax.plot(predictions, label='Las losowy')
        line3, = ax.plot(pm25_predictions, label='Sieci neuronowe')
        lines = [line1, line2, line3]

        prawdziwe_wartosci_pm2_5 = self.data_pred['pm2_5']

        correct_forest = 0
        incorrect_forest = 0

        for i in range(1, len(prawdziwe_wartosci_pm2_5)):
            predicted_change = (predictions[i] - predictions[i-1]) > 0
            actual_change = (prawdziwe_wartosci_pm2_5[i] - prawdziwe_wartosci_pm2_5[i-1]) > 0
            if predicted_change == actual_change:
                correct_forest += 1
            else:
                incorrect_forest += 1

        legend = ax.legend(loc='upper right')
        legends = legend.get_lines()

        for line, leg in zip(lines, legends):
            leg.set_picker(True)
            leg.set_pickradius(10)

        def on_pick(event):
            leg = event.artist
            index = legends.index(leg)
            line = lines[index]
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
            leg.set_visible(not isVisible)
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', on_pick)
        plt.xlabel('Indeks próbki')
        plt.ylabel('Wartość pm2_5')
        plt.title('Porównanie wartości aktualnych i przewidywanych')

        plt.show()

        correct, incorrect = test.pobierz_trendy()
        text = f"Correct: {correct}, Incorrect: {incorrect}"

        text2 = f"Correct: {correct_forest}, Incorrect: {incorrect_forest}"

        def oblicz_margines_bledu(wartosci_rzeczywiste, przewidywania):
            if len(wartosci_rzeczywiste) != len(przewidywania):
                raise ValueError("Liczba wartości rzeczywistych i przewidywanych musi być taka sama.")
            
            liczba_wartosci = len(wartosci_rzeczywiste)
            margines_bledu = sum(abs(przewidywania[i] - wartosci_rzeczywiste[i]) / wartosci_rzeczywiste[i] for i in range(liczba_wartosci))
            procentowy_margines_bledu = (margines_bledu / liczba_wartosci) * 100
            
            tekst_procentowy_margines_bledu = "{:.2f}%".format(procentowy_margines_bledu)

            return tekst_procentowy_margines_bledu

        # obliczanie marginesu bledu
        margines_bledu_sieci_neuronowe = oblicz_margines_bledu(prawdziwe_wartosci_pm2_5, pm25_predictions)
        margines_bledu_las_losowy = oblicz_margines_bledu(prawdziwe_wartosci_pm2_5, predictions)

        # OKNO TRENDOW
        self.oknoTrendow = QMainWindow()
        self.oknoTrendow.setWindowTitle("Trendy")
        self.oknoTrendow.setFixedSize(400, 300)
        self.oknoTrendow.move(100, 100)
        logo_icon = QIcon("logo.png")
        self.oknoTrendow.setWindowIcon(logo_icon)

        font = QFont("Arial", 12, QFont.Bold)

        labelGlownyTrendy = QLabel("TRENDY", self.oknoTrendow)
        labelGlownyTrendy.setFont(font)
        labelGlownyTrendy.setGeometry(0, -60, 400, 300)
        labelGlownyTrendy.setAlignment(Qt.AlignCenter)

        label = QLabel("Sieci neuronowe", self.oknoTrendow)
        label.setGeometry(0, -30, 400, 300)
        label.setAlignment(Qt.AlignCenter)

        label2 = QLabel(text, self.oknoTrendow)
        label2.setGeometry(0, 0, 400, 300)
        label2.setAlignment(Qt.AlignCenter)

        label3 = QLabel("Las losowy", self.oknoTrendow)
        label3.setGeometry(0, 30, 400, 300)
        label3.setAlignment(Qt.AlignCenter)

        label4 = QLabel(text2, self.oknoTrendow)
        label4.setGeometry(0, 60, 400, 300)
        label4.setAlignment(Qt.AlignCenter)

        self.oknoTrendow.show()

        # OKNO MARGINESU BLEDU
        self.oknoMarginesuBledu = QMainWindow()
        self.oknoMarginesuBledu.setWindowTitle("Marginesy błędu")
        self.oknoMarginesuBledu.setFixedSize(400, 300)
        self.oknoMarginesuBledu.move(100, 500)
        logo_icon = QIcon("logo.png")
        self.oknoMarginesuBledu.setWindowIcon(logo_icon)

        labelGlownyMarginesBledu = QLabel("MARGINESY BŁĘDU", self.oknoMarginesuBledu)
        labelGlownyMarginesBledu.setFont(font)
        labelGlownyMarginesBledu.setGeometry(0, -60, 400, 300)
        labelGlownyMarginesBledu.setAlignment(Qt.AlignCenter)

        label5 = QLabel("Sieci neuronowe", self.oknoMarginesuBledu)
        label5.setGeometry(0, -30, 400, 300)
        label5.setAlignment(Qt.AlignCenter)

        label6 = QLabel(margines_bledu_sieci_neuronowe, self.oknoMarginesuBledu)
        label6.setGeometry(0, 0, 400, 300)
        label6.setAlignment(Qt.AlignCenter)

        label7 = QLabel("Las losowy", self.oknoMarginesuBledu)
        label7.setGeometry(0, 30, 400, 300)
        label7.setAlignment(Qt.AlignCenter)

        label8 = QLabel(margines_bledu_las_losowy, self.oknoMarginesuBledu)
        label8.setGeometry(0, 60, 400, 300)
        label8.setAlignment(Qt.AlignCenter)

        self.oknoMarginesuBledu.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())
