# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sco
import MarketData as md

# *********************************************** #
# Inputs
# *********************************************** #
# weights : Vecteur des poids de chaque actif
# mean_returns : Vecteur des rendrements moyens de chaque actif
# cov_matrix : Matrice de variance covariance
# *********************************************** #
# Outputs
# *********************************************** #
# returns : rendement annualisé du portefeuille
# std : Volatilité du portefeuille
# *********************************************** #
AnnuBase = 252


def perfo_annualisee_portefeuille(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * AnnuBase
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(AnnuBase)  # Volatilité
    return std, returns


# *********************************************** #
# Inputs
# *********************************************** #
# num_portfolios : Nombre de portefeuilles
# mean_returns : rendements moyens
# cov_matrix : matrice de variance covariance
# risk_free_rate : taux sans risque
# *********************************************** #
# Outputs
# *********************************************** #
# results : matrice des volatilités, rendements et ratio de sharpe par portefeuille
# weights_record : matrice des poids
# *********************************************** #
def portefeuilles_alea(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    numberOfAssets = len(cov_matrix)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(numberOfAssets)  # generation de numberOfAssets nombres aléatoirs
        weights /= np.sum(weights)  # reduction (sommes des poids =1)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = perfo_annualisee_portefeuille(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record
    

# to do ......................
def reportPrint(rp,sdp,max_sharpe_allocation,rp_min,sdp_min,min_vol_allocation):
    print ( "*"*100)
    print ("Portefeuille ayant le meilleur ratio de sharp\n")
    print ("Rendement Annualisé:", round(rp,2))
    print ("Volatilité annualisée:", round(sdp,2))
    print ("\n")
    print (max_sharpe_allocation)
    print ("*"*100)
    print ("Portefeuille le moins volatile\n")
    print ("Rendement Annualisé:", round(rp_min,2))
    print ("Volatilité annualisée:", round(sdp_min,2))
    print ("\n")
    print (min_vol_allocation)

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    mesDonnes = md.MarketData()

    results, _ = portefeuilles_alea(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = perfo_annualisee_portefeuille(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=mesDonnes.table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = perfo_annualisee_portefeuille(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=mesDonnes.table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    reportPrint(rp,sdp,max_sharpe_allocation,rp_min,sdp_min,min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Meilleur ratio de Sharpe')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Volatilité minimum')

    target = np.linspace(rp_min, 0.63, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    plt.title('Simulation de portefeuilles aléatoirs')
    plt.xlabel('Volatilité annualisée')
    plt.ylabel('Rendement annualisé')
    plt.legend(labelspacing=0.8)
    plt.show()


# Les fonction d'optimisation sont toujours des fonctions de minimisation (exemple solver excel). 
# Donc pour maximiser le ration de sharp nous devons calculer un ration de sharpe 
# negatif pour pouvoir applique une fonction min
# minimize -ratioSharp equivalent to maximize ratioSharp

# negative sharpe ratio, qui est la focntion à minimiser
# *********************************************** #
# Inputs 
# *********************************************** #
# weights : Vecteur des poids de chaque actif 
# mean_returns : Vecteur des rendrements moyens de chaque actif
# cov_matrix : Matrice de variance covariance
# risk_free_rate : Taux sans risque
# *********************************************** #
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = perfo_annualisee_portefeuille(weights, mean_returns, cov_matrix) # vol, perfo annualisée du portefeuils
    return -(p_ret - risk_free_rate) / p_var # ratio de sharpe

# Solver ou fonction d'optimisation scipy minimize
def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)#nombre d'actifs
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # la contrainte de la fonction d'optimisation somme des poids = 1
    bound = (0.0,1.0) # seuil 0<= poids <= 1
    bounds = tuple(bound for asset in range(num_assets)) # tous les poids ont le même seuil ici
    # scipy.optimize.minimize ici le point de depart est [0.25, 0.25, 0.25, 0.25],(mean_returns, cov_matrix, risk_free_rate)
    result = sco.minimize(negative_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# *********************************************** #
# Vol
# *********************************************** #
# Inputs 
# *********************************************** #
# weights : Vecteur des poids de chaque actif 
# mean_returns : Vecteur des rendrements moyens de chaque actif
# cov_matrix : Matrice de variance covariance
# *********************************************** #
# *********************************************** #
# Outputs 
# *********************************************** #
# std : Volatilité du portefeuille
# *********************************************** #
def portfolio_volatility(weights, mean_returns, cov_matrix):
    return perfo_annualisee_portefeuille(weights, mean_returns, cov_matrix)[0]

# Solver scipy minimize variance
def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})# la contrainte de la fonction d'optimisation somme des poids = 1
    bound = (0.0,1.0)# seuil 0<= poid <= 1
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

# portefeuil le moins risqué pour un rendement donné
def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return perfo_annualisee_portefeuille(weights, mean_returns, cov_matrix)[1] # renvoie le rendement annualisé seulement pour un portefeuil aléatoire

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, returns):
    mesDonnees = md.MarketData()

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = perfo_annualisee_portefeuille(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=mesDonnees.table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = perfo_annualisee_portefeuille(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=mesDonnees.table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252
    
    print ("-"*80)
    print ("Maximum Sharpe Ratio Portfolio Allocation\n")
    print ("Annualised Return:", round(rp,2))
    print ("Annualised Volatility:", round(sdp,2))
    print ("\n")
    print (max_sharpe_allocation)
    print ("-"*80)
    print ("Minimum Volatility Portfolio Allocation\n")
    print ("Annualised Return:", round(rp_min,2))
    print ("Annualised Volatility:", round(sdp_min,2))
    print ("\n")
    print (min_vol_allocation)
    print ("-"*80)
    print ("Individual Stock Returns and Volatility\n")
    for i, txt in enumerate(mesDonnees.table.columns):
        print (txt,":","annuaised return",round(an_rt[i],2),", annualised volatility:",round(an_vol[i],2))
    print ("-"*80)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(an_vol,an_rt,marker='o',s=200)

    for i, txt in enumerate(mesDonnees.table.columns):
        ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, rp, 100)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Portfolio Optimization with Individual Stocks')
    ax.set_xlabel('annualised volatility')
    ax.set_ylabel('annualised returns')
    plt.legend(labelspacing=2)
    plt.show()