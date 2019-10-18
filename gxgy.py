# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 12:45:05 2018

@author: MK
"""

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import numpy as np
from scipy.linalg import *
import matplotlib.pyplot as plt

class GUIWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.gxLQG = 0
        self.gyLQG = 0
        self.gxCrit = 0
        self.gyCrit = 0
        self.gxGy = 0
        self.gyGy = 1
        self.fignum = 1

        centralWidget = QWidget(self)
        layout = QGridLayout(centralWidget)
        layout.setSpacing(10)
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.setLabel = QLabel(self)
        self.setLabel.setText("Set parameters")
        layout.addWidget(self.setLabel, 0, 1)
        self.tau_xLabel = QLabel(self)
        self.tau_xLabel.setText("tau_x (d): ")
        layout.addWidget(self.tau_xLabel, 1, 0)
        self.tau_xEdit = QLineEdit(self)
        self.tau_xEdit.editingFinished.connect(self.to_tau_x)
        layout.addWidget(self.tau_xEdit, 1, 1)
        self.tau_xEdit.setText("1")
        self.tau_x = 1

        self.dtLabel = QLabel(self)
        self.dtLabel.setText("dt (h): ")
        layout.addWidget(self.dtLabel, 2, 0)
        self.dtEdit = QLineEdit(self)
        self.dtEdit.editingFinished.connect(self.to_dt)
        layout.addWidget(self.dtEdit, 2, 1)
        self.dtEdit.setText("1")
        self.dt = 1
        
        self.pLabel = QLabel(self)
        self.pLabel.setText("p: ")
        layout.addWidget(self.pLabel, 3, 0)
        self.pEdit = QLineEdit(self)
        self.pEdit.editingFinished.connect(self.to_p)
        layout.addWidget(self.pEdit, 3, 1)
        self.pEdit.setText("1")
        self.p = 1
        
        self.txTLabel = QLabel(self)
        self.txTLabel.setText("alpha_x: ")
        layout.addWidget(self.txTLabel, 4, 0)
        self.txTEdit = QLineEdit(self)
        self.txTEdit.editingFinished.connect(self.to_txT)
        layout.addWidget(self.txTEdit, 4, 1)
        self.txTEdit.setText("0.3")
        self.txT = 0.3
        
        self.tcLabel = QLabel(self)
        self.tcLabel.setText("T(d)=alpha_x*tau_x: ")
        layout.addWidget(self.tcLabel, 5, 0)
        self.tcEdit = QLineEdit(self)
        layout.addWidget(self.tcEdit, 5, 1)
        tc = self.tau_x*self.txT
        self.tcEdit.setText(str(tc))
        
        
        self.gxLQGLabel = QLabel(self)
        self.gxLQGLabel.setText("gx: ")
        layout.addWidget(self.gxLQGLabel, 1, 2)
        self.gyLQGLabel = QLabel(self)
        self.gyLQGLabel.setText("gy: ")
        layout.addWidget(self.gyLQGLabel, 2, 2)
        self.eigReLQGLabel = QLabel(self)
        self.eigReLQGLabel.setText("Re eig: ")
        layout.addWidget(self.eigReLQGLabel, 3, 2)
        self.eigImLQGLabel = QLabel(self)
        self.eigImLQGLabel.setText("Im eig: ")
        layout.addWidget(self.eigImLQGLabel, 4, 2)
        self.tLabel = QLabel(self)
        self.tLabel.setText("T (d): ")
        layout.addWidget(self.tLabel, 5, 2)
        
        self.lqgLabel = QLabel(self)
        self.lqgLabel.setText("LQG")
        layout.addWidget(self.lqgLabel, 0, 3)
        self.gxLQGEdit = QLineEdit(self)
        layout.addWidget(self.gxLQGEdit, 1, 3)
        self.gyLQGEdit = QLineEdit(self)
        layout.addWidget(self.gyLQGEdit, 2, 3)
        self.eigReLQGEdit = QLineEdit(self)
        layout.addWidget(self.eigReLQGEdit, 3, 3)
        self.eigImLQGEdit = QLineEdit(self)
        layout.addWidget(self.eigImLQGEdit, 4, 3)
        
        
        self.lqgLabel = QLabel(self)
        self.lqgLabel.setText("Critical damping")
        layout.addWidget(self.lqgLabel, 0, 4)
        self.gxCritEdit = QLineEdit(self)
        layout.addWidget(self.gxCritEdit, 1, 4)
        self.gyCritEdit = QLineEdit(self)
        layout.addWidget(self.gyCritEdit, 2, 4)
        self.eigReCritEdit = QLineEdit(self)
        layout.addWidget(self.eigReCritEdit, 3, 4)
        self.eigImCritEdit = QLineEdit(self)
        layout.addWidget(self.eigImCritEdit, 4, 4)
        
        self.lqgLabel = QLabel(self)
        self.lqgLabel.setText("gy = 1")
        layout.addWidget(self.lqgLabel, 0, 5)
        self.gxGyEdit = QLineEdit(self)
        layout.addWidget(self.gxGyEdit, 1, 5)
        self.eigReGyEdit = QLineEdit(self)
        layout.addWidget(self.eigReGyEdit, 3, 5)
        self.eigImGyEdit = QLineEdit(self)
        layout.addWidget(self.eigImGyEdit, 4, 5)
        
        self.icLabel = QLabel(self)
        self.icLabel.setText("(x0, y0): ")
        layout.addWidget(self.icLabel, 6, 2)
        self.icXEdit = QLineEdit(self)
        self.icXEdit.editingFinished.connect(self.to_X0)
        self.icXEdit.setText("1e-9")
        self.x0 = 1e-9
        layout.addWidget(self.icXEdit, 6, 3)
        self.icYEdit = QLineEdit(self)
        self.icYEdit.editingFinished.connect(self.to_Y0)
        self.icYEdit.setText("0")
        self.y0 = 0
        layout.addWidget(self.icYEdit, 6, 4)
        
        self.tEdit = QLineEdit(self)
        layout.addWidget(self.tEdit, 5, 3)
        self.t2Edit = QLineEdit(self)
        layout.addWidget(self.t2Edit, 5, 4)
        self.t3Edit = QLineEdit(self)
        layout.addWidget(self.t3Edit, 5, 5)
        
        self.calcBtn = QPushButton(self)
        self.calcBtn.setText("Calculate")
        self.calcBtn.clicked.connect(self.calc)
        layout.addWidget(self.calcBtn, 6, 1)
        self.plot1Btn = QPushButton(self)
        self.plot1Btn.setText("Plot")
        self.plot1Btn.clicked.connect(self.plot)
        layout.addWidget(self.plot1Btn, 6, 5)


        self.resize(800,180)

    def to_tau_x(self):
        self.tau_x = float(self.tau_xEdit.text())
        tc = self.tau_x*self.txT
        self.tcEdit.setText(str(tc))

    def to_dt(self):
        self.dt = float(self.dtEdit.text())

    def to_p(self):
        self.p = float(self.pEdit.text())

    def to_txT(self):
        self.txT = float(self.txTEdit.text())
        tc = self.tau_x*self.txT
        self.tcEdit.setText(str(tc))

    def to_X0(self):
        self.x0 = float(self.icXEdit.text())

    def to_Y0(self):
        self.y0 = float(self.icYEdit.text())

    def calc(self):
        tc = self.tau_x*self.txT*86400
        dt = self.dt*3600
        gx = (1-np.exp(-dt/tc))**2/dt
        gy = 1-np.exp(-2*dt/tc)
        self.gxCrit = gx
        self.gyCrit = gy
        self.gxCritEdit.setText(str(gx))
        self.gyCritEdit.setText(str(gy))
        F_ = np.array([[1-dt*gx, dt-dt*gy], [-gx, 1-gy]])
        ei = eig(F_)[0]
        self.eigReCritEdit.setText(str(np.real(ei)))
        self.eigImCritEdit.setText(str(np.imag(ei)))
        self.t2Edit.setText(str(tc/86400))

        gx = (1-np.exp(-self.dt/tc))/self.dt
        gy = 1
        self.gxGy = gx
        self.gyGy = 1
        self.gxGyEdit.setText(str(gx))
        F_ = np.array([[1-dt*gx, dt-dt*gy], [-gx, 1-gy]])
        ei = eig(F_)[0]
        self.eigReGyEdit.setText(str(np.real(ei)))
        self.eigImGyEdit.setText(str(np.imag(ei)))
        #T = -self.dt/np.log(1-gx*dt)/24
        self.t3Edit.setText(str(tc/86400))

        rpq = 0.25*(tc/dt)**2*tc**2
        q = 1.e-3
        r = rpq*q*self.p

        F = np.array([[1, dt], [0, 1]])
        B = np.array([[dt], [1]])
        Wq = q*np.array([[self.p, 0], [0, 1]])
        Wr = np.array([[r]])
        Gamma = solve_discrete_are(F, B, Wq, Wr)
        G = inv(B.T@Gamma@B+Wr)@(B.T@Gamma@F)
        gx = G[0][0]
        gy = G[0][1]
        self.gxLQG = gx
        self.gyLQG = gy
        self.gxLQGEdit.setText(str(gx))
        self.gyLQGEdit.setText(str(gy))
        F_ = np.array([[1-dt*gx, dt-dt*gy], [-gx, 1-gy]])
        ei = eig(F_)[0]
        self.eigReLQGEdit.setText(str(np.real(ei)))
        self.eigImLQGEdit.setText(str(np.imag(ei)))
        T = -self.dt/np.log(np.max(np.abs(ei)))/24
        self.tEdit.setText(str(T))

    def plot(self):
        gx = self.gxLQG
        gy = self.gyLQG
        self.gyLQG
        N = int(self.tau_x*2*24/self.dt)
        dt = self.dt*3600
        x=np.zeros(N)
        y=np.zeros(N)
        t=np.zeros(N)
        u=np.zeros(N-1)
        x[0] = self.x0
        y[0] = self.y0
        t[0] = 0
        for n in range(N-1):
            u[n] = -gx*x[n]-gy*y[n]
            x[n+1] = x[n]+y[n]*dt+u[n]*dt
            y[n+1] = y[n]+u[n]
            t[n+1] = t[n]+dt

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(t/86400,x,'-',)
        ax2.plot(t/86400,y,'-')
        gx = self.gxCrit
        gy = self.gyCrit
        N = int(self.tau_x*2*24/self.dt)
        dt = self.dt*3600
        x = np.zeros(N)
        y = np.zeros(N)
        t = np.zeros(N)
        u = np.zeros(N-1)
        x[0] = self.x0
        y[0] = self.y0
        t[0] = 0
        for n in range(N-1):
            u[n] = -gx*x[n]-gy*y[n]
            x[n+1] = x[n]+y[n]*dt+u[n]*dt
            y[n+1] = y[n]+u[n]
            t[n+1] = t[n]+dt
        ax1.plot(t/86400,x,'-g')
        ax2.plot(t/86400,y,'-g')
        gx = self.gxGy
        gy = 1
        N = int(self.tau_x*2*24/self.dt)
        dt = self.dt*3600
        x = np.zeros(N)
        y = np.zeros(N)
        t = np.zeros(N)
        u = np.zeros(N-1)
        x[0] = self.x0
        y[0] = self.y0
        t[0] = 0
        for n in range(N-1):
            u[n] = -gx*x[n]-gy*y[n]
            x[n+1] = x[n]+y[n]*dt+u[n]*dt
            y[n+1] = y[n]+u[n]
            t[n+1] = t[n]+dt
        ax1.plot(t/86400,x,'-r')
        ax2.plot(t/86400,y,'-r')
        ax1.set(ylabel='phase offset, s')
        ax2.set(xlabel='time in days', ylabel='frequency offset')
        ax1.legend(['LQG','Crit. damping','g_y=1'])
        ax2.legend(['LQG','Crit. damping','g_y=1'])
        plt.show()

app = QCoreApplication.instance()
if app is None:
    app = QApplication(sys.argv)

win = GUIWindow()
win.show()

app.exec()