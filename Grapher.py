import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Quantlib as ql




class Grapher:
    def __init__(self, marketData):
        self.dataObj = marketData
        self.returns = self.dataObj.table.pct_change()

    def displayRendementActif(self):
        table = self.dataObj.table
        plt.figure(figsize=(14, 7))
        for c in table.columns.values:
            plt.plot(table.index, table[c], lw=3, alpha=0.8,label=c)
        plt.legend(loc='upper left', fontsize=12)
        plt.ylabel('price in $')
        plt.show()

    def displayRendementJournalier(self):
        returns = self.returns
        plt.figure(figsize=(14, 7))
        for c in returns.columns.values:
            plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
        plt.legend(loc='upper right', fontsize=12)
        plt.ylabel('daily returns')
        plt.show()

    def displayFrontiereEfficiente(self, mean_returns, cov_matrix, num_portfolios, risk_free_rate):
        ql.display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
        return 0

    def displayImpactDiversification(self, mean_returns, cov_matrix, risk_free_rate):
        ql.display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, self.returns)
        return 0
