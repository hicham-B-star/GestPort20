#!/usr/bin/python
# -*- coding: latin-1 -*-
import Grapher as gr
import MarketData as md

def main():
    while True:
        print("Options Disponibles:\n")
        print("1. Graphe des cours de l'action\n")
        print("2. Graphe de rendement journalier\n")
        print("3. Graphe de frontiere efficiente pour un portefeuille d'actions:\n")
        print("4. Graphe de l'impact de la diversification\n")
        choixGraph = int(input("Saisir 1, 2, 3 ou 4:\n"))

        # demer le
        mesDonnes = md.MarketData()
        monGrapher = gr.Grapher(mesDonnes)

        if choixGraph == 1:
            monGrapher.displayRendementActif()

        elif choixGraph == 2:
            monGrapher.displayRendementJournalier()

        elif choixGraph == 3 or choixGraph == 4:
            if choixGraph == 3:
                num_portfolios = int(input("Nombre de portefeuils simules:\n"))
            risk_free_rate = float(input("Taux sans risque\n"))

            # rendement couts(t) - cours(t-1)
            rendements = mesDonnes.table.pct_change()
            # Rendement moyens
            mean_returns = rendements.mean()
            # Matrice de variance covariance
            cov_matrix = rendements.cov()
            if choixGraph == 3:
                monGrapher.displayFrontiereEfficiente(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
            else:
                monGrapher.displayImpactDiversification(mean_returns, cov_matrix, risk_free_rate)

        else:
            print('Saisie non reconnue\n')

if __name__ == '__main__':
    main()

