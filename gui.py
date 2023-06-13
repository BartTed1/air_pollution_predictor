import csv
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLineEdit, QMessageBox, QLabel, QLabel
from PyQt5.QtCore import Qt, QFileInfo
from PyQt5.QtGui import QIcon, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

sys.path.append('./model/3h/RandomForestRegressor')
from forest import RandomForestRegressorModel
forest = RandomForestRegressorModel()

sys.path.append('./model/3h/NeuralNetwork')
import nntest as nnTest
import neural_network as nn
import timeutils as ts

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pollution prediction")
        self.setFixedSize(400, 250)  # Set fixed size

        # program icon
        logo_icon = QIcon("logo.png")
        self.setWindowIcon(logo_icon)
    
        # main label
        self.start = QLabel("PM 2.5 value prediction", self)
        self.start.setGeometry(0, 10, 400, 30)
        font = QFont("Arial", 12, QFont.Bold)
        self.start.setFont(font)
        self.start.setAlignment(Qt.AlignCenter)

        # load first file
        self.tekst1plik = QLabel("Load a csv file with study data: ", self)
        self.tekst1plik.setGeometry(20, 50, 400, 30)

        self.line_edit = QLineEdit(self)
        self.line_edit.setGeometry(20, 80, 160, 20)
        self.line_edit.setReadOnly(True)
        self.line_edit.setStyleSheet("border: 0.5px solid black;")

        self.button = QPushButton("Choose a file", self)
        self.button.setGeometry(190, 75, 100, 30)
        self.button.clicked.connect(lambda: self.select_file('train'))

        # load second file
        self.tekst2plik = QLabel("Load a csv file with prediction data: ", self)
        self.tekst2plik.setGeometry(20, 110, 400, 30)

        self.line_edit2 = QLineEdit(self)
        self.line_edit2.setGeometry(20, 140, 160, 20)
        self.line_edit2.setReadOnly(True)
        self.line_edit2.setStyleSheet("border: 0.5px solid black;")

        self.button2 = QPushButton("Choose a file", self)
        self.button2.setGeometry(190, 135, 100, 30)
        self.button2.clicked.connect(lambda: self.select_file('pred'))

        # Generate chart
        self.button_wykres = QPushButton("Generate chart", self)
        self.button_wykres.setGeometry(120, 190, 150, 30)
        self.button_wykres.clicked.connect(self.generate_plot)

    def select_file(self, data_type):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose a file", "", "All files (*)", options=options)
        if file_name:
            if file_name.lower().endswith('.csv'):
                file_info = QFileInfo(file_name)
                if data_type == 'train':
                    self.line_edit.setText(file_info.fileName())
                    self.data_train = forest.read_data(file_name)
                    self.data_train_path = file_name  # path to file
                elif data_type == 'pred':
                    self.line_edit2.setText(file_info.fileName())
                    self.data_pred = forest.read_data(file_name)
                    self.data_pred_path = file_name  # path to file
            else:
                QMessageBox.critical(self, "ERROR!", "Invalid file selected. Please select a CSV file.")

    def generate_plot(self):
        if self.line_edit.text() == '' or self.line_edit2.text() == '':
            QMessageBox.warning(self, "ERROR!", "Both files are not selected. Please select CSV files.")
            return

        # Selection of selected features
        selected_features = ['pm2_5', 'wind_speed', 'wind_deg', 'temp', 'humidity']

        # Division of data into features and expected values
        X_train = forest.select_features(self.data_train, selected_features)
        y_train = self.data_train['pm2_5_label']
        X_pred = forest.select_features(self.data_pred, selected_features)

        # Model parameters
        params = {
            'criterion': 'absolute_error',
            'max_depth': 15,
            'max_features': 'sqrt',
            'min_impurity_decrease': 0.0,
            'min_samples_leaf': 4,
            'min_samples_split': 10,
            'n_estimators': 150
        }

        # Create a new instance of Figure and FigureCanvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setGeometry(10, 50, 780, 540)
        self.ax = self.figure.add_subplot(111)

        # Model training
        model = forest.train_model(X_train, y_train, params)

        # Value prediction
        predictions = forest.predict(model, X_pred)

        # Neural network
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

        #TODO: Add loading bar

        # Build and train the neural network
        neural_network.build_and_train()
        
        # Create a test object
        test = nnTest.NNTest(neural_network, data2, labels2)

        # Print predictions
        predictions_neural = test.forecast(12)

        # Convert predictions to a list of pm2.5 values
        pm25_predictions = [pred[0] for pred in predictions_neural]
        # Neural network END

        # Chart generation
        fig, ax = plt.subplots(figsize=(8, 6))
        line1, = ax.plot(self.data_pred['pm2_5'], label='Real values')
        line2, = ax.plot(predictions, label='Random forest')
        line3, = ax.plot(pm25_predictions, label='Neutal network')
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
        plt.xlabel('Sample index')
        plt.ylabel('pm2_5 value')
        plt.title('Comparison of current and predicted values')
        plt.show()

        correct, incorrect = test.pobierz_trendy()
        text = f"Correct: {correct}, Incorrect: {incorrect}"
        text2 = f"Correct: {correct_forest}, Incorrect: {incorrect_forest}"

        def calculate_error_margin(real_values, predicted_values):
            if len(real_values) != len(predicted_values):
                raise ValueError("Liczba wartości rzeczywistych i przewidywanych musi być taka sama.")
            
            number_values = len(real_values)
            margin_of_error = sum(abs(predicted_values[i] - real_values[i]) / real_values[i] for i in range(number_values))
            percentage_margin_of_error = (margin_of_error / number_values) * 100
            tekst_percentage_margin_of_error = "{:.2f}%".format(percentage_margin_of_error)
            return tekst_percentage_margin_of_error

        # Calculating the margin of error
        margin_of_error_sieci_neuronowe = calculate_error_margin(prawdziwe_wartosci_pm2_5, pm25_predictions)
        margin_of_error_las_losowy = calculate_error_margin(prawdziwe_wartosci_pm2_5, predictions)

        # Trend Window
        self.TrendWindow = QMainWindow()
        self.TrendWindow.setWindowTitle("Trends")
        self.TrendWindow.setFixedSize(400, 300)
        self.TrendWindow.move(100, 100)
        logo_icon = QIcon("logo.png")
        self.TrendWindow.setWindowIcon(logo_icon)

        font = QFont("Arial", 12, QFont.Bold)

        labelGlownyTrendy = QLabel("TRENDS", self.TrendWindow)
        labelGlownyTrendy.setFont(font)
        labelGlownyTrendy.setGeometry(0, -60, 400, 300)
        labelGlownyTrendy.setAlignment(Qt.AlignCenter)

        label = QLabel("Neural Network", self.TrendWindow)
        label.setGeometry(0, -30, 400, 300)
        label.setAlignment(Qt.AlignCenter)

        label2 = QLabel(text, self.TrendWindow)
        label2.setGeometry(0, 0, 400, 300)
        label2.setAlignment(Qt.AlignCenter)

        label3 = QLabel("Random forest", self.TrendWindow)
        label3.setGeometry(0, 30, 400, 300)
        label3.setAlignment(Qt.AlignCenter)

        label4 = QLabel(text2, self.TrendWindow)
        label4.setGeometry(0, 60, 400, 300)
        label4.setAlignment(Qt.AlignCenter)

        self.TrendWindow.show()

        # ERROR MARGIN WINDOW
        self.oknoMarginesuBledu = QMainWindow()
        self.oknoMarginesuBledu.setWindowTitle("Margin of error")
        self.oknoMarginesuBledu.setFixedSize(400, 300)
        self.oknoMarginesuBledu.move(100, 500)
        logo_icon = QIcon("logo.png")
        self.oknoMarginesuBledu.setWindowIcon(logo_icon)

        labelGlownyMarginesBledu = QLabel("MARGIN OF ERROR", self.oknoMarginesuBledu)
        labelGlownyMarginesBledu.setFont(font)
        labelGlownyMarginesBledu.setGeometry(0, -60, 400, 300)
        labelGlownyMarginesBledu.setAlignment(Qt.AlignCenter)

        label5 = QLabel("Neural Network", self.oknoMarginesuBledu)
        label5.setGeometry(0, -30, 400, 300)
        label5.setAlignment(Qt.AlignCenter)

        label6 = QLabel(margin_of_error_sieci_neuronowe, self.oknoMarginesuBledu)
        label6.setGeometry(0, 0, 400, 300)
        label6.setAlignment(Qt.AlignCenter)

        label7 = QLabel("Random Forest", self.oknoMarginesuBledu)
        label7.setGeometry(0, 30, 400, 300)
        label7.setAlignment(Qt.AlignCenter)

        label8 = QLabel(margin_of_error_las_losowy, self.oknoMarginesuBledu)
        label8.setGeometry(0, 60, 400, 300)
        label8.setAlignment(Qt.AlignCenter)

        self.oknoMarginesuBledu.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())
