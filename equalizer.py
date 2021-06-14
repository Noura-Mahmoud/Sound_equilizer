from PyQt5 import QtWidgets, QtCore, uic, QtGui
from pyqtgraph import PlotWidget, plot
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import *   
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from scipy import signal
from os import path
import pyqtgraph as pg
import numpy as np
import sys
import os
from scipy.io import wavfile
from scipy import signal
import simpleaudio as sa
from scipy.fft import rfft, rfftfreq, fft, fftfreq, ifft, irfft
from matplotlib import pyplot as plt
import librosa
import sounddevice as sd
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from fpdf import FPDF
import pyqtgraph.exporters

MAIN_WINDOW,_=loadUiType(path.join(path.dirname(__file__),"sigview.ui"))
MAIN_WINDOW2,_=loadUiType(path.join(path.dirname(__file__),"fft2.ui"))

class MainApp(QMainWindow,MAIN_WINDOW):
    
    def __init__(self,parent=None):
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.timer= QtCore.QTimer()
        self.speed = 150

        ## for max and min sliders of spectrogram
        self.freq_min=0 
        self.freq_max=0

        global gainArray, sliderArray
        sliderArray=[self.verticalSlider,self.verticalSlider_2,self.verticalSlider_3,self.verticalSlider_4,self.verticalSlider_5,self.verticalSlider_6,self.verticalSlider_7,self.verticalSlider_8,self.verticalSlider_9,self.verticalSlider_10]
        self.graphWidgets=[self.graphWidget,self.graphWidget2]
        self.spectroWidgets=[self.spectroWidget2,self.spectroWidget]
        self.spectro_sliders=[self.min_value_slider,self.max_value_slider]
 
        # design graphWidgets
        for i in range(2):
            self.graphWidgets[i].plotItem.showGrid(True, True, alpha=0.8)
            self.graphWidgets[i].setBackground('w')
        self.graphWidgets[0].plotItem.setTitle("Before Equalization")
        self.graphWidgets[1].plotItem.setTitle("After Equalization")

        self.comboBox.currentIndexChanged.connect(self.colorPallete)
        self.checkBox.stateChanged.connect(self.showSpectro)
        self.Menubar()
        self.Toolbar()
        self.showSpectro()
        # ConnectSliders function
        for i in range(10):     
            sliderArray[i].valueChanged.connect(self.changeslidervalue)
        self.newWindows = []  
        self.comap = 0
        
    def Menubar(self):
        self.actionOpen_signal.triggered.connect(self.BrowseSignal)
        self.actionSave_signal_as.triggered.connect(self.saveFile)
        self.actionExit.triggered.connect(self.close)
        self.Add_New_window.triggered.connect(self.addNewWindow)

    def Toolbar(self):
        self.OpenSignalBtn.triggered.connect(self.BrowseSignal)
        self.DrawSig.triggered.connect(self.PlottingTimer)
        self.actionSpeed_Up.triggered.connect(self.speed_up)
        self.actionSpeed_down.triggered.connect(self.speed_down)
        self.AddPanel.triggered.connect(self.addNewWindow)
        self.PlayBtn.triggered.connect(self.play_audio)
        self.Stop.triggered.connect(self.stop_audio)
        self.ZoomIn.triggered.connect(self.zoomIn) 
        self.ZoomOut.triggered.connect(self.zoomOut) 
        self.LeftScroll.triggered.connect(self.ScrollLeft) 
        self.RightScroll.triggered.connect(self.ScrollRight)
        self.PDF.triggered.connect(self.printPDF) 
        self.Save_signal.triggered.connect(self.saveFile) 
        self.ShowFftButton.triggered.connect(self.showFFT) 

    def BrowseSignal(self):
        global fileName, sampling_rate, audioData, length
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","WAV Files (*.wav)")
        if fileName:
            self.graphWidgets[0].plotItem.clear()
            audioData, sampling_rate = librosa.load(fileName, sr=None, duration=20.0)
            length = len(audioData)
            self.changeslidervalue()
            self.plotAudio(audioData, length)
            self.graphWidgets[0].plotItem.getViewBox().setLimits(xMin=0,xMax=length)
        else: 
            pass
    
    def plotAudio(self,file,length):
        self.graphWidgets[0].plot(file[0:length],pen="b")

    def PlottingTimer(self):
        for i in range(2):
            self.timer = QtCore.QTimer()
            self.timer.setInterval(self.speed)    
            self.timer.timeout.connect(self.PlottingTimer)
            self.timer.start()
            xrange, yrange = self.graphWidgets[i].viewRange()
            scrollvalue = (xrange[1] - xrange[0])/50
            self.graphWidgets[i].setXRange(xrange[0]+scrollvalue, xrange[1]+scrollvalue, padding=0)

    def speed_up(self):
        if self.speed == 10:
            self.speed = 0
        elif self.speed > 0:
            self.speed -= 20
        self.PlottingTimer()     

    def speed_down(self):
        self.speed += 20
        self.PlottingTimer()

    def play_audio(self):
        sd.play(adjusted_audio, sampling_rate)

    def stop_audio(self):
        sd.stop()
        self.timer.stop()

    def changeslidervalue(self):
        gainArray = []
        for i in range(10):
            gainArray.append(sliderArray[i].value())
        self.processAudio(audioData, sampling_rate, gainArray)
        return gainArray

    def processAudio(self, audioData, sampling_rate, gainArray):
        window_length = length
        sample_spacing = 1/sampling_rate
        global complex_fourier, bandwidth, bands, new_fft_signal
        bands = []
        complex_fourier = rfft(audioData)
        frequencies = rfftfreq(window_length, sample_spacing)  ## sample freq
        bandwidth1=np.where(frequencies==((sampling_rate)/20))
        bandwidth=bandwidth1[0][0]
        for i in range(10):
            bands.append(complex_fourier[i*bandwidth:(i+1)*bandwidth]*gainArray[i])
        new_fft_signal=np.concatenate((bands))

        self.PlotNewSignal(new_fft_signal)
        self.colorPallete()
        self.play_audio()
        
        for i in range(2):
            self.spectro_sliders[i].setMinimum(self.min_freq)
            self.spectro_sliders[i].setMaximum(self.max_freq)
            self.spectro_sliders[i].setSingleStep(10)
            self.spectro_sliders[i].valueChanged.connect(self.spectroAfter)
        self.min_value_slider.setValue(self.min_freq)
        self.max_value_slider.setValue(self.max_freq)

    def PlotNewSignal(self,new_fft_signal):
        global adjusted_audio
        adjusted_audio = irfft(new_fft_signal)
        self.graphWidgets[1].plotItem.clear()
        self.graphWidgets[1].plot(adjusted_audio,pen = "r")
        self.graphWidgets[1].plotItem.getViewBox().setLimits(xMin=0,xMax=length,yMax=np.amax(adjusted_audio)+2,yMin=np.amin(adjusted_audio)-2)
    
    def addNewWindow(self):
        New_window=MainApp()
        New_window.show()
        self.newWindows.append(New_window)
        
    def zoomIn(self):
        self.timer.stop()
        for i in range(2):
           self.graphWidgets[i].plotItem.getViewBox().scaleBy(x=0.5, y=1) #Increases the scale of X axis and Y axis

    def zoomOut(self):
        self.timer.stop()
        for i in range(2):
           self.graphWidgets[i].plotItem.getViewBox().scaleBy(x=2, y=1) #Decreases scale of X axis and Y axis 

    def ScrollLeft(self):
        self.timer.stop()
        for i in range(2):
           self.graphWidgets[i].plotItem.getViewBox().translateBy(x=-(length/1000), y=0)

    def ScrollRight(self):
        self.timer.stop()
        for i in range(2):
           self.graphWidgets[i].plotItem.getViewBox().translateBy(x=(length/1000), y=0)
    
    def colorPallete(self):
        if self.comboBox.currentText()=='Palette 1':      
            self.comap = cm.get_cmap('viridis') 
        elif self.comboBox.currentText()=='Palette 2':
            self.comap = cm.get_cmap('plasma')
        elif self.comboBox.currentText()=='Palette 3':
            self.comap = cm.get_cmap('cool')
        elif self.comboBox.currentText()=='Palette 4':
            self.comap = cm.get_cmap('rainbow')
        else:
            self.comap = cm.get_cmap('GnBu')
        self.spectroBefore()
        self.spectroAfter()
    
    def spectroBefore(self):
        fig = plt.figure()
        plt.subplot(111)
        self.powerSpectrum, self.freqenciesFound, self.time, self.imageAxis = plt.specgram(audioData, Fs=sampling_rate, cmap=self.comap)
        self.min_freq = int(10* (np.log10 (self.powerSpectrum.min())))
        self.max_freq = int(10* (np.log10 (self.powerSpectrum.max())))
        plt.colorbar()
        plt.title('Before Equalization')
        fig.savefig('plot_before.png')
        self.upload()       
        plt.clf()
 
    def spectroAfter(self):
        # print("min" , self.min_value_slider.value())
        # print("max" , self.max_value_slider.value())    
        fig = plt.figure()
        plt.subplot(111)
        self.spectrogram = plt.specgram(adjusted_audio, Fs=sampling_rate, cmap=self.comap, vmin=self.min_value_slider.value(), vmax=self.max_value_slider.value())
        plt.colorbar()
        plt.title('After Equalization')
        fig.savefig('plot_after.png')
        self.upload()
        plt.clf()

    def upload(self):
        for i in range(2):
            self.spectroWidgets[i].setScaledContents(True)
        self.spectroWidgets[0].setPixmap(QtGui.QPixmap("plot_before.png"))
        self.spectroWidgets[1].setPixmap(QtGui.QPixmap("plot_after.png"))

    def showSpectro(self):
        if self.checkBox.isChecked()==True:
            self.verticalWidget.show()
        else:
            self.verticalWidget.hide()

    def generatePDF(self, filename):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 15)
        pdf.set_xy(0,0)
        for i in range(2):
            pdf.cell(0, 10,ln=1,align='C')
            exporter = pg.exporters.ImageExporter(self.graphWidgets[i].plotItem)               
            exporter.parameters()['width'] = 250  
            exporter.parameters()['height'] = 250         
            exporter.export('fileName'+str(i+1)+'.png')
            pdf.image(('fileName'+str(i+1)+'.png'),x=None,y=None, w=180,h=70)

        pdf.cell(0, 10,ln=1,align='C')
        pdf.image('plot_before.png',x=None,y=None, w=200,h=100)

        pdf.cell(0, 10,ln=1,align='C')
        pdf.image('plot_after.png',x=None,y=None, w=200,h=100)

        pdf.output(filename)

    def printPDF(self):
        # allows the user to save the file and name it as they like
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PDF", None, "PDF files (.pdf);;All Files()"
        )
        if filename:
            if QtCore.QFileInfo(filename).suffix() == "":
                filename += ".pdf"
            self.generatePDF(filename)

    def generate_WavFile(self, filename):
        maximum = np.max(np.abs(adjusted_audio))
        data = (adjusted_audio / maximum).astype(np.float32)
        save = wavfile.write(filename, int(sampling_rate), data)
        plt.subplot(211)
        plot(adjusted_audio)

    def saveFile(self):
        # allows the user to save the file and name it as they like
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export WAV", None, "WAV files (.wav);;All Files()"
        )
        if filename:
            if QtCore.QFileInfo(filename).suffix() == "":
                filename += ".wav"
            self.generate_WavFile(filename)
    
    def showFFT(self):
        fft_window.show()

class MainApp2(QMainWindow,MAIN_WINDOW2):
    def __init__(self,parent=None):
        super(MainApp2,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)    
        self.pushButton.clicked.connect(self.fftt)
        
    def fftt(self):
        window_length = length
        sample_spacing = 1/sampling_rate  
        global complex_fourier
        complex_fourier = rfft(audioData)
        frequencies = rfftfreq(window_length, sample_spacing)
        self.fourWidget.plot(frequencies,np.abs(complex_fourier),pen = "b")
        print(len(new_fft_signal))
        self.fourWidget2.plot(frequencies[1:],np.abs(new_fft_signal)[ : len(frequencies)], pen='r')
        
def main():
    app = QApplication(sys.argv)
    window = MainApp()
    global fft_window
    fft_window = MainApp2()
    window.show()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()