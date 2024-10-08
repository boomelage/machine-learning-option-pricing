#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:26:58 2024

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from settings import model_settings
ms = model_settings()
import time

def plot_volatility_surface(
        black_var_surface, K, T, title="", elev=30, azim=120):
    plt.rcParams['figure.figsize']=(15,7)
    K = K.astype(int)
    plot_maturities = np.sort(np.array(T,dtype=float)/365)
    plot_strikes = np.sort(K).astype(float)
    X, Y = np.meshgrid(plot_strikes, plot_maturities)
    Z = np.array([[
        black_var_surface.blackVol(y, x) for x in plot_strikes] 
        for y in plot_maturities])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=elev, azim=azim)  
    ax.set_title(title)
    surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ax.set_xlabel("Strikes", size=9)
    ax.set_ylabel("Maturities (Years)", size=9)
    ax.set_zlabel("Volatility", size=9)
    
    plt.show()
    plt.cla()
    plt.clf()
    return fig

def plot_term_structure(
        K,
        real_ts,
        est_ts,
        title = ""
        ):
    plt.rcParams['figure.figsize']=(6,4)
    # K = K.astype(int)
    fig, ax = plt.subplots()
    ax.plot(K, est_ts)
    ax.plot(K, real_ts, "o")
    ax.set_title(title)
    plt.show()
    plt.cla()
    plt.clf()
    return fig

def plot_rotate(black_var_surface,K,T,title="",elev=30):
    plt.rcParams['figure.figsize']=(15,7)
    plt.rcParams['figure.figsize']=(15,7)
    K = K.astype(int)
    plot_maturities = np.sort(np.array(T,dtype=float)/365)
    plot_strikes = np.sort(K).astype(float)
    X, Y = np.meshgrid(plot_strikes, plot_maturities)
    Z = np.array([[
        black_var_surface.blackVol(y, x) for x in plot_strikes] 
        for y in plot_maturities])
    
    azims = np.arange(0,360,30)

    for azim in azims:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=elev, azim=azim)  
        ax.set_title(title)
        surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        ax.set_xlabel("Strikes", size=9)
        ax.set_ylabel("Maturities (Years)", size=9)
        ax.set_zlabel("Volatility", size=9)
        fig.savefig(f'{int(time.time()*10)}.png')
        time.sleep(0.001)
        plt.show()
        plt.cla()
        plt.clf()
    return fig
