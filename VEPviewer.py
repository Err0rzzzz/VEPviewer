import sys
import random
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
                             QComboBox, QHBoxLayout, QLineEdit, QMessageBox, QFileDialog)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, title=""):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.title = title
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

    def plot(self, data, global_avg=None, class_avg=None, title=None):
        if title is None:
            self.axes.clear()
            for waveform in data:
                self.axes.plot(waveform)
            if global_avg is not None:
                self.axes.plot(global_avg, 'k--', label="Global Average")
            if class_avg is not None:
                self.axes.plot(class_avg, 'r--', label=f"{self.title} Class Average")
            self.axes.set_title(f'{self.title} Waveforms')
            self.axes.legend()
            self.draw()
        else:
            self.axes.clear()
            for waveform in data:
                self.axes.plot(waveform)
            if global_avg is not None:
                self.axes.plot(global_avg, 'k--', label="Global Average")
            if class_avg is not None:
                self.axes.plot(class_avg, 'r--', label=f"{title} Class Average")
            self.axes.set_title(f'{title} Waveforms')
            self.axes.legend()
            self.draw()


class EEGAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.global_average_waveform = None
        self.initUI()
        self.mse = None

    def initUI(self):
        self.setGeometry(100, 100, 1200, 900)
        self.setWindowTitle('EEG Data Analyzer')

        # Main layout and central widget
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        # File loading and processing controls
        loadButton = QPushButton('Load EEG Data', self)
        loadButton.clicked.connect(self.loadFile)
        layout.addWidget(loadButton)

        self.fileNameLabel = QLabel('No file loaded', self)
        layout.addWidget(self.fileNameLabel)

        # Inputs for EEG and Stimuli channels, sigma, baseline points
        self.eegChannelInput = QLineEdit(self)
        self.eegChannelInput.setPlaceholderText("Enter EEG channel index")
        layout.addWidget(self.eegChannelInput)

        self.stimChannelInput = QLineEdit(self)
        self.stimChannelInput.setPlaceholderText("Enter Stimuli channel index")
        layout.addWidget(self.stimChannelInput)

        self.sigmaInput = QLineEdit(self)
        self.sigmaInput.setPlaceholderText("Enter Gaussian smoothing sigma")
        layout.addWidget(self.sigmaInput)

        self.baselinePtsInput = QLineEdit(self)
        self.baselinePtsInput.setPlaceholderText("Enter baseline correction points")
        layout.addWidget(self.baselinePtsInput)

        processButton = QPushButton('Process Data', self)
        processButton.clicked.connect(self.processData)
        layout.addWidget(processButton)

        self.resultsLabel = QLabel('Results will be displayed here', self)
        layout.addWidget(self.resultsLabel)

        # Plot canvases for L, M, S
        self.plotL = PlotCanvas(self, width=5, height=4, dpi=100, title="Large (L)")
        self.plotM = PlotCanvas(self, width=5, height=4, dpi=100, title="Medium (M)")
        self.plotS = PlotCanvas(self, width=5, height=4, dpi=100, title="Small (S)")
        layout.addWidget(self.plotL)
        layout.addWidget(self.plotM)
        layout.addWidget(self.plotS)

        # Dropdown to select number of waveforms
        self.numWaveformSelector = QComboBox(self)
        self.numWaveformSelector.addItems(['5', '6', '7', '8', '9', '10'])
        layout.addWidget(self.numWaveformSelector)

        # Zoom controls for L, M, S plots
        self.zoom_controls_layout = QHBoxLayout()
        self.zoomInputsL = QLineEdit(self)
        self.zoomInputsM = QLineEdit(self)
        self.zoomInputsS = QLineEdit(self)
        self.applyZoomButtonL = QPushButton('Apply Zoom L', self)
        self.applyZoomButtonM = QPushButton('Apply Zoom M', self)
        self.applyZoomButtonS = QPushButton('Apply Zoom S', self)
        self.zoom_controls_layout.addWidget(self.zoomInputsL)
        self.zoom_controls_layout.addWidget(self.applyZoomButtonL)
        self.zoom_controls_layout.addWidget(self.zoomInputsM)
        self.zoom_controls_layout.addWidget(self.applyZoomButtonM)
        self.zoom_controls_layout.addWidget(self.zoomInputsS)
        self.zoom_controls_layout.addWidget(self.applyZoomButtonS)
        layout.addLayout(self.zoom_controls_layout)

        self.applyZoomButtonL.clicked.connect(lambda: self.applyZoom('L'))
        self.applyZoomButtonM.clicked.connect(lambda: self.applyZoom('M'))
        self.applyZoomButtonS.clicked.connect(lambda: self.applyZoom('S'))

    def loadFile(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open EEG Data File", "", "EEG Files (*.bin);;All Files (*)",
                                                  options=options)
        if fileName:
            try:
                self.fileNameLabel.setText(f'Loaded File: {fileName}')
                # Assuming the file is binary and needs to be processed as such
                self.df = self.load_bin_data(fileName)
                QMessageBox.information(self, "File Loaded", "EEG data file has been loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "File Loading Error", f"An error occurred while loading the file: {str(e)}")
                self.fileNameLabel.setText('File loading failed!')

    def load_bin_data(self, filepath, num_channels=8):
        try:
            # Change dtype according to your data specifics
            data = np.fromfile(filepath, dtype=np.float64)
            data = data.reshape((-1, num_channels))
            return pd.DataFrame(data)
        except Exception as e:
            raise Exception(f"Error processing the binary file: {str(e)}")

    def processData(self):
        try:
            eeg_channel_index = int(self.eegChannelInput.text())
            stim_channel_index = int(self.stimChannelInput.text())
            sigma = float(self.sigmaInput.text())
            baseline_pts = int(self.baselinePtsInput.text())

            stim_channel = self.df.iloc[:, stim_channel_index - 1].values
            eeg_channel = self.df.iloc[:, eeg_channel_index - 1].values

            # Find stimulation epochs
            epoch_idx = np.where(stim_channel > 1)[0]
            start_idx = [epoch_idx[0]]
            close_idx = []
            for i in range(len(epoch_idx) - 1):
                if epoch_idx[i + 1] - epoch_idx[i] > 100:
                    start_idx.append(epoch_idx[i + 1])
                    close_idx.append(epoch_idx[i])
            close_idx.append(epoch_idx[-1])

            # Flop the last epoch of each period of stimuli
            for c in range(len(close_idx) + 1):
                if c % 200 == 0:
                    close_idx[c - 1] = close_idx[c - 1] - 500

            # Smooth the EEG data
            eeg_channel = gaussian_filter(eeg_channel, sigma=sigma)

            # Collect the waveforms
            waveforms = []
            for i in range(len(close_idx)):
                waveforms.append(eeg_channel[start_idx[i]:start_idx[i] + 500])
                waveforms.append(eeg_channel[close_idx[i]:close_idx[i] + 500])
            waveforms = np.array(waveforms)

            # Baseline correction
            baseline = waveforms[:, :baseline_pts].mean(axis=1, keepdims=True)
            waveforms -= baseline

            global_average_waveform = np.mean(waveforms, axis=0)
            mse = np.mean((waveforms - global_average_waveform) ** 2, axis=1)
            thresholds = np.quantile(mse, [1 / 3, 2 / 3])
            self.mse = mse
            # Store data for plotting
            self.dataL = waveforms[mse <= thresholds[0]]
            self.dataM = waveforms[(mse > thresholds[0]) & (mse <= thresholds[1])]
            self.dataS = waveforms[mse > thresholds[1]]
            self.global_average_waveform = np.mean(waveforms, axis=0)
            self.updatePlots()
            self.resultsLabel.setText(f"Processed Data: {len(waveforms)} epochs detected, MSE thresholds: {thresholds}")


        except Exception as e:
            QMessageBox.critical(self, "Processing Error", str(e))

    def updatePlots(self):
        # Get the number of waveforms to show from the dropdown
        num_waveforms = int(self.numWaveformSelector.currentText())
        # Update each plot with the default number of waveforms
        self.updatePlot([self.dataL[i] for i in random.sample(range(len(self.dataL)), min(num_waveforms, len(self.dataL)))], self.global_average_waveform, self.plotL, "Large (L)")
        self.updatePlot([self.dataM[i] for i in random.sample(range(len(self.dataL)), min(num_waveforms, len(self.dataL)))], self.global_average_waveform, self.plotM, "Medium (M)")
        self.updatePlot([self.dataL[i] for i in random.sample(range(len(self.dataL)), min(num_waveforms, len(self.dataL)))], self.global_average_waveform, self.plotS, "Small (S)")

    def updatePlot(self, data, global_avg, plot_canvas, title):
        class_avg = np.mean(data, axis=0)
        plot_canvas.plot(data, global_avg, class_avg, title)

    def applyZoom(self, plot_class):
        try:
            if plot_class == 'L':
                zoom_text = self.zoomInputsL.text()
                plot_canvas = self.plotL
                data = self.dataL
            elif plot_class == 'M':
                zoom_text = self.zoomInputsM.text()
                plot_canvas = self.plotM
                data = self.dataM
            elif plot_class == 'S':
                zoom_text = self.zoomInputsS.text()
                plot_canvas = self.plotS
                data = self.dataS
            else:
                raise ValueError("Invalid plot class specified")

            x, y = [int(p) for p in zoom_text.split(',')]
            x_index = int(len(data) * (x / 100))
            y_index = int(len(data) * (y / 100))
            mse = np.mean((data - self.global_average_waveform) ** 2, axis=1)
            sorted_indices = np.argsort(mse)  # Sort based on MSE
            selected_indices = sorted_indices[x_index:y_index]
            sorted_data = data[selected_indices]

            # Assuming class averages are already calculated
            class_avg = np.mean(sorted_data, axis=0)
            plot_canvas.plot(sorted_data, self.global_average_waveform, class_avg)
        except Exception as e:
            QMessageBox.critical(self, "Zoom Error", f"An error occurred while applying zoom: {str(e)}")

    def updatePlotZoom(self, data, class_avg, zoom_range, plot_canvas):
        try:
            x, y = map(int, zoom_range.split('-'))
            sorted_data = sorted(data, key=lambda w: np.mean(w[x:y]))
            plot_canvas.plot(sorted_data, self.global_average_waveform, class_avg)
        except Exception as e:
            QMessageBox.critical(self, "Zoom Error", f"An error occurred while applying zoom: {str(e)}")


def main():
    app = QApplication(sys.argv)
    ex = EEGAnalyzer()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()